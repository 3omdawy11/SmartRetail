"""
Consumes the Kafka stream produced by producer.py, runs LSTM surge inference and
price propagation via PricingEngine (see pricing_engine.py - the same logic already
validated offline in run_dry_run.py), fires vendor alerts via notifier.py, and writes
simulation/output/price_history.csv when the producer's END sentinel arrives.

Preloads the item -> base_price catalog from the same local CSV the producer reads,
so every item (including related items that haven't streamed their own row yet) has
a known starting price from the first message onward.
"""
import csv
import json
import os

import notifier
from pricing_engine import load_model_and_config, PricingEngine, write_log_and_summary

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = "smart-retail-stream"
GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "smart-retail-consumer")
INPUT_CSV = "data/processed/raw_sample.csv"
OUTPUT_CSV = "simulation/output/price_history.csv"


def load_catalog():
    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["event_date"], int(r["itemid"])))

    catalog = {}
    for row in rows:
        itemid = int(row["itemid"])
        if itemid not in catalog:
            catalog[itemid] = float(row["base_price"])
    start_date = rows[0]["event_date"]
    return catalog, start_date


def main():
    from confluent_kafka import Consumer

    model, config, relationships = load_model_and_config("models")
    catalog_base_prices, start_date = load_catalog()
    engine = PricingEngine(model, config, relationships, catalog_base_prices, start_date)

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([TOPIC])

    print(f"Consuming from '{TOPIC}' ({BOOTSTRAP_SERVERS})... waiting for messages.")
    rows_processed = 0
    try:
        while True:
            msg = consumer.poll(30.0)
            if msg is None:
                print("No message received in 30s - producer may not be running. Stopping.")
                break
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            if msg.key() == b"__END__":
                print("Received END sentinel - stream complete.")
                break

            row = json.loads(msg.value())
            new_entries = engine.process_row(row)
            rows_processed += 1

            if new_entries and new_entries[0]["trigger_reason"] == "surge":
                surge_entry = new_entries[0]
                related_changes = new_entries[1:]
                notifier.notify_surge(
                    itemid=surge_entry["itemid"],
                    event_date=surge_entry["event_date"],
                    confidence=surge_entry["confidence"],
                    old_price=surge_entry["old_price"],
                    new_price=surge_entry["new_price"],
                    related_changes=related_changes,
                )

            if rows_processed % 500 == 0:
                print(f"  ...{rows_processed:,} rows processed")
    finally:
        consumer.close()

    print(f"\nRows processed: {rows_processed:,}")
    if rows_processed == 0:
        print("No rows consumed - did the producer run? (Also check: a previous "
              "consumer run may have already committed past all messages in this "
              "consumer group. Restart Redpanda or use a fresh KAFKA_GROUP_ID env var.) "
              "Not overwriting price_history.csv.")
        return

    write_log_and_summary(engine, OUTPUT_CSV)


if __name__ == "__main__":
    main()
