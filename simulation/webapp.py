"""
Web dashboard for the SmartRetail dynamic pricing simulation.

Runs PricingEngine directly over raw_sample.csv (the same path already verified in
run_dry_run.py) rather than acting as a live Kafka consumer itself - the CLI
producer/consumer pair already demonstrates the real Kafka mechanics, and duplicating
that inside a Streamlit script (which reruns on every widget interaction) would
reintroduce the consumer-group offset-commit bug that was just fixed there. The
dashboard is a view over the same price-change log either path produces.
"""
import csv
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing_engine import load_model_and_config, PricingEngine

INPUT_CSV = "data/processed/raw_sample.csv"
RELATED_ITEMS_PER_FOCUS = 5   # caps overlay lines so the chart stays readable

st.set_page_config(page_title="SmartRetail — Dynamic Pricing", layout="wide")


@st.cache_resource
def get_model_and_config():
    return load_model_and_config("models")


def load_rows_and_catalog():
    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["event_date"], int(r["itemid"])))

    catalog = {}
    for row in rows:
        itemid = int(row["itemid"])
        if itemid not in catalog:
            catalog[itemid] = float(row["base_price"])
    start_date = rows[0]["event_date"]
    return rows, catalog, start_date


def run_simulation(price_bump_pct, related_bump_scale, cooldown_days, price_cap_multiplier):
    model, config, relationships = get_model_and_config()
    rows, catalog, start_date = load_rows_and_catalog()

    # Keeps the same ratio as PricingEngine's defaults (related cap = 1.25 when the
    # primary cap is 1.5, i.e. related items get half the primary's headroom above 1.0)
    related_price_cap_multiplier = 1 + (price_cap_multiplier - 1) * 0.5

    engine = PricingEngine(
        model, config, relationships, catalog, start_date,
        price_bump_pct=price_bump_pct,
        related_bump_scale=related_bump_scale,
        cooldown_days=cooldown_days,
        price_cap_multiplier=price_cap_multiplier,
        related_price_cap_multiplier=related_price_cap_multiplier,
    )
    for row in rows:
        engine.process_row(row)

    return pd.DataFrame(engine.log)


st.title("SmartRetail — Dynamic Pricing Simulation")
st.caption(
    "Replays the test-window stream through the LSTM surge detector and the "
    "relationship-graph price propagation, in date order. The same logic and "
    "results as the Kafka producer/consumer pair, viewed here as a dashboard."
)

with st.sidebar:
    st.header("Pricing policy")
    price_bump_pct = st.slider("Surge price bump (%)", 5, 25, 10) / 100
    related_bump_scale = st.slider("Related item bump scale", 0.0, 1.0, 0.5, step=0.05)
    cooldown_days = st.slider("Cooldown (days between re-bumps)", 0, 7, 3)
    price_cap_multiplier = st.slider("Price cap (x base price)", 1.1, 2.0, 1.5, step=0.05)
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Streaming raw_sample.csv through the model..."):
        st.session_state["log_df"] = run_simulation(
            price_bump_pct, related_bump_scale, cooldown_days, price_cap_multiplier
        )

if "log_df" not in st.session_state:
    st.info("Click **Run Simulation** in the sidebar to stream the test set and see price changes.")
    st.stop()

log_df = st.session_state["log_df"]

# ── KPIs ─────────────────────────────────────────────────────────────────────
n_items = log_df["itemid"].nunique()
n_surges = (log_df["trigger_reason"] == "surge").sum()
n_related = (log_df["trigger_reason"] == "related").sum()
final_prices = log_df.sort_values("seq").groupby("itemid").last()
base_prices = log_df.sort_values("seq").groupby("itemid").first()["new_price"]
pct_changes = (final_prices["new_price"] - base_prices) / base_prices * 100
n_capped = (pct_changes >= (price_cap_multiplier - 1) * 100 - 0.1).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Items streamed", f"{n_items:,}")
col2.metric("Surge events", f"{n_surges:,}")
col3.metric("Related bumps", f"{n_related:,}")
col4.metric("Items at price cap", f"{n_capped:,}")

# ── Chart: price change over time for selected items ────────────────────────
st.subheader("Price change over time")

surge_counts = log_df[log_df["trigger_reason"] == "surge"]["itemid"].value_counts()
default_items = surge_counts.head(5).index.tolist()
all_items = sorted(log_df["itemid"].unique())

selected_items = st.multiselect(
    "Items to chart (defaults to the 5 most-surged items)",
    options=all_items,
    default=default_items,
)

if selected_items:
    fig = go.Figure()
    end_date = pd.to_datetime(log_df["event_date"]).max()

    for itemid in selected_items:
        item_log = log_df[log_df["itemid"] == itemid].sort_values("seq")
        base_price = item_log.iloc[0]["new_price"]
        dates = pd.to_datetime(item_log["event_date"]).tolist()
        pct = ((item_log["new_price"] - base_price) / base_price * 100).tolist()
        if dates[-1] < end_date:
            dates.append(end_date)
            pct.append(pct[-1])

        fig.add_trace(go.Scatter(
            x=dates, y=pct, mode="lines", name=f"item {itemid}",
            line_shape="hv",   # step function: matches the price log's discrete jumps
        ))

        surges = item_log[item_log["trigger_reason"] == "surge"]
        if not surges.empty:
            surge_dates = pd.to_datetime(surges["event_date"])
            surge_pct = (surges["new_price"] - base_price) / base_price * 100
            fig.add_trace(go.Scatter(
                x=surge_dates, y=surge_pct, mode="markers", showlegend=False,
                marker=dict(size=8, symbol="star"),
                text=[f"confidence {c:.0%}" for c in surges["confidence"]],
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price change from base (%)",
        height=500, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Select at least one item to plot.")

# ── Table: current prices ────────────────────────────────────────────────────
st.subheader("Current prices")

table = pd.DataFrame({
    "itemid": final_prices.index,
    "base_price": base_prices.values,
    "current_price": final_prices["new_price"].values,
    "pct_change": pct_changes.values,
    "last_trigger": final_prices["trigger_reason"].values,
    "last_event_date": final_prices["event_date"].values,
}).sort_values("pct_change", ascending=False).reset_index(drop=True)

st.dataframe(
    table,
    use_container_width=True,
    height=420,
    column_config={
        "base_price": st.column_config.NumberColumn("Base price", format="$%.2f"),
        "current_price": st.column_config.NumberColumn("Current price", format="$%.2f"),
        "pct_change": st.column_config.NumberColumn("Change", format="%.2f%%"),
    },
)
