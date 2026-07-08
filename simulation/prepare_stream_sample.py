"""
Regenerates data/processed/raw_sample.csv for the Phase 2 streaming demo.

Notebook 1's original raw_sample.csv (`test_df.orderBy(event_date, itemid).limit(2000)`)
only captured the *first* calendar day of the test window - one row per item, no history
at all. The LSTM needs a 7-day rolling window per item before it can predict anything, so
that file could never trigger a surge during streaming.

This script builds a proper sample locally from the full test/ Parquet (already on disk,
no Spark/Colab needed): it selects items with enough history to form at least one LSTM
window, prioritised by view volume (more views -> a more plausible surge candidate), plus
their related items from relationships.json (so price propagation has something to show),
and keeps every available row for those items across the full test date range.
"""
import json

import pandas as pd

TEST_PARQUET = "data/processed/test"
RELATIONSHIPS_JSON = "models/relationships.json"
OUTPUT_CSV = "data/processed/raw_sample.csv"

WINDOW_SIZE = 7          # must match lstm_config.json's window_size
TOP_N_PRIMARY = 100      # number of "primary" items selected by view volume


def main():
    df = pd.read_parquet(TEST_PARQUET)

    with open(RELATIONSHIPS_JSON) as f:
        relationships = json.load(f)
    related_lookup = {int(k): v for k, v in relationships.items()}

    per_item_days = df.groupby("itemid").size()
    eligible = set(per_item_days[per_item_days >= WINDOW_SIZE].index)
    candidates = eligible & set(related_lookup.keys())

    views_by_item = df.groupby("itemid")["daily_views"].sum()
    ranked = views_by_item.loc[list(candidates)].sort_values(ascending=False)
    primary_items = set(ranked.head(TOP_N_PRIMARY).index)

    related_items = set()
    for itemid in primary_items:
        for entry in related_lookup[itemid]:
            related_items.add(entry["related"])

    all_items = primary_items | related_items

    sample = (
        df[df["itemid"].isin(all_items)]
        .sort_values(["event_date", "itemid"])
        .reset_index(drop=True)
    )

    sample.to_csv(OUTPUT_CSV, index=False)

    print(f"Primary items (>= {WINDOW_SIZE} days history, ranked by views): {len(primary_items):,}")
    print(f"Related items pulled in from relationships.json     : {len(related_items):,}")
    print(f"Total items in sample                                : {len(all_items):,}")
    print(f"Total rows written                                   : {len(sample):,}")
    print(f"Date range                                           : {sample['event_date'].min()} -> {sample['event_date'].max()}")
    print(f"Written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
