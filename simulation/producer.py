"""
Streams data/processed/raw_sample.csv row-by-row into the Kafka topic, in
chronological order, simulating the test-window clickstream arriving live.
Sends a sentinel END message when done so the consumer knows to stop.
"""
import csv
import json
import os
import time

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = "smart-retail-stream"
INPUT_CSV = "data/processed/raw_sample.csv"
STREAM_DELAY_SECONDS = float(os.environ.get("STREAM_DELAY_SECONDS", "0.02"))


def delivery_callback(err, msg):
    if err is not None:
        print(f"Delivery failed for key {msg.key()}: {err}")


def reset_topic():
    """Deletes and recreates the topic so each producer run starts clean - the
    demo is stateful across runs otherwise (old rows and END sentinels from a
    previous run would still be sitting in the topic)."""
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})

    if TOPIC in admin.list_topics(timeout=10).topics:
        for topic, future in admin.delete_topics([TOPIC]).items():
            try:
                future.result(timeout=10)
            except Exception as e:
                print(f"  (delete_topics for '{topic}': {e})")
        time.sleep(2)   # give the broker a moment to fully remove it before recreating

    for topic, future in admin.create_topics([NewTopic(TOPIC, num_partitions=1, replication_factor=1)]).items():
        try:
            future.result(timeout=10)
        except Exception as e:
            print(f"  (create_topics for '{topic}': {e})")


def main():
    reset_topic()

    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["event_date"], int(r["itemid"])))

    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

    print(f"Streaming {len(rows):,} rows to topic '{TOPIC}' ({BOOTSTRAP_SERVERS})...")
    for i, row in enumerate(rows, start=1):
        producer.produce(
            TOPIC,
            key=row["itemid"],
            value=json.dumps(row),
            callback=delivery_callback,
        )
        producer.poll(0)
        if STREAM_DELAY_SECONDS:
            time.sleep(STREAM_DELAY_SECONDS)
        if i % 500 == 0:
            print(f"  ...{i:,}/{len(rows):,} rows sent")

    producer.produce(TOPIC, key="__END__", value=json.dumps({"type": "END"}))
    producer.flush(10)
    print(f"Done. {len(rows):,} rows + END sentinel sent to '{TOPIC}'.")


if __name__ == "__main__":
    main()
