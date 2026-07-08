"""
Reads simulation/output/price_history.csv and charts price trajectories over the
simulated stream: for each of the most active surging items, its own price line plus
the related items it bumped (via relationships.json), with surge points annotated.
Also produces one overview chart of surge activity across the simulated timeline.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = "simulation/output/price_history.csv"
CHARTS_DIR = "simulation/output/charts"
TOP_N_ITEMS = 6


def build_pct_change_series(df, itemid, end_date):
    """Step-function series of % change from base price, sorted by seq, extended
    flat out to end_date so every line reaches the same right edge instead of
    stopping wherever that item's log happened to end."""
    item_log = df[df["itemid"] == itemid].sort_values("seq")
    base_price = item_log.iloc[0]["new_price"]   # first entry is always the "initial" row

    dates = pd.to_datetime(item_log["event_date"]).tolist()
    pct_change = ((item_log["new_price"] - base_price) / base_price * 100).tolist()

    if dates[-1] < end_date:
        dates.append(end_date)
        pct_change.append(pct_change[-1])

    return dates, pct_change, base_price


def plot_item_chart(df, source_itemid, end_date, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    dates, pct_change, base_price = build_pct_change_series(df, source_itemid, end_date)
    ax.plot(dates, pct_change, drawstyle="steps-post",
            label=f"item {source_itemid} (surging, base ${base_price:.2f})",
            color="tab:red", linewidth=2.5, zorder=3)

    surges = df[(df["itemid"] == source_itemid) & (df["trigger_reason"] == "surge")]
    for _, row in surges.iterrows():
        surge_date = pd.to_datetime(row["event_date"])
        surge_pct = (row["new_price"] - base_price) / base_price * 100
        ax.axvline(surge_date, color="tab:red", linestyle="--", alpha=0.3, zorder=1)
        ax.annotate(
            f"{row['confidence']:.0%}",
            xy=(surge_date, surge_pct),
            xytext=(0, 8), textcoords="offset points",
            fontsize=8, color="tab:red", ha="center",
        )

    related_ids = df[
        (df["source_item"] == source_itemid) & (df["trigger_reason"] == "related")
    ]["itemid"].unique()

    for related_id in related_ids[:5]:   # cap overlay lines so the chart stays readable
        r_dates, r_pct, r_base = build_pct_change_series(df, related_id, end_date)
        ax.plot(r_dates, r_pct, drawstyle="steps-post",
                label=f"related item {related_id} (base ${r_base:.2f})",
                linewidth=1.2, alpha=0.7, zorder=2)

    ax.set_title(f"Price change from base - item {source_itemid} and its related items")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price change from base (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_surge_timeline(df, out_path):
    surges = df[df["trigger_reason"] == "surge"].copy()
    surges["event_date"] = pd.to_datetime(surges["event_date"])
    daily_counts = surges.groupby("event_date").size()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(daily_counts.index, daily_counts.values, color="tab:orange")
    ax.set_title("Surge events per day across the simulated stream")
    ax.set_xlabel("Date")
    ax.set_ylabel("Surge count")
    ax.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    df = pd.read_csv(INPUT_CSV)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    surge_counts = df[df["trigger_reason"] == "surge"]["itemid"].value_counts()
    top_items = surge_counts.head(TOP_N_ITEMS).index.tolist()
    end_date = pd.to_datetime(df["event_date"]).max()

    print(f"Charting top {len(top_items)} items by surge count: {top_items}")
    for itemid in top_items:
        out_path = f"{CHARTS_DIR}/item_{itemid}.png"
        plot_item_chart(df, itemid, end_date, out_path)
        print(f"  wrote {out_path}")

    timeline_path = f"{CHARTS_DIR}/surge_timeline.png"
    plot_surge_timeline(df, timeline_path)
    print(f"  wrote {timeline_path}")

    print(f"\nAll charts written to: {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
