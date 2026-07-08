"""
Runs the pricing engine directly over data/processed/raw_sample.csv, in date order,
with no Kafka/Docker involved. This is the fast way to validate the surge-detection +
price-propagation logic itself before wrapping it in the Kafka transport - also usable
as a standalone way to generate simulation/output/price_history.csv without Docker.
"""
import csv

from pricing_engine import load_model_and_config, PricingEngine, write_log_and_summary

INPUT_CSV = "data/processed/raw_sample.csv"
OUTPUT_CSV = "simulation/output/price_history.csv"


def main():
    model, config, relationships = load_model_and_config("models")

    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["event_date"], int(r["itemid"])))

    catalog_base_prices = {}
    for row in rows:
        itemid = int(row["itemid"])
        if itemid not in catalog_base_prices:
            catalog_base_prices[itemid] = float(row["base_price"])
    start_date = rows[0]["event_date"]

    print(f"Rows: {len(rows):,}  Items: {len(catalog_base_prices):,}  Start date: {start_date}")

    engine = PricingEngine(model, config, relationships, catalog_base_prices, start_date)

    for row in rows:
        engine.process_row(row)

    write_log_and_summary(engine, OUTPUT_CSV)


if __name__ == "__main__":
    main()
