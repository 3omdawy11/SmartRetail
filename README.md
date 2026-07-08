# RetailPrice: Predictive Demand Forecasting & Real-Time Dynamic Pricing Engine

An end-to-end e-commerce optimization pipeline built on the Retailrocket clickstream dataset. It combines offline batch machine learning (demand forecasting) with a real-time streaming engine (traffic surge detection and dynamic price adjustment). The system is split into two phases: heavy computation on Google Colab using PySpark, and a lightweight local simulation using plain Python and Kafka.

---

## Project Structure

```
smart_retail/
├── colab_notebooks/
│   ├── 1_data_preprocessing.ipynb   ← ETL (DONE)
│   ├── 2_train_baseline_ml.ipynb    ← GBT demand forecasting (DONE)
│   ├── 3_train_lstm_dl.ipynb        ← LSTM surge detection (DONE)
│   └── 4_build_graphframes.ipynb    ← Product relationship graph (DONE)
├── data/
│   ├── raw/                         ← Drop raw CSVs here (gitignored)
│   └── processed/                   ← ETL outputs land here (gitignored)
├── models/
│   ├── baseline_config.json
│   ├── lstm_model.pth
│   ├── lstm_config.json
│   └── relationships.json
├── simulation/
│   ├── prepare_stream_sample.py
│   ├── pricing_engine.py
│   ├── run_dry_run.py
│   ├── producer.py
│   ├── consumer_engine.py
│   ├── notifier.py
│   ├── visualize_results.py
│   ├── webapp.py
│   └── output/                      ← price_history.csv, charts/ (gitignored)
├── docker-compose.yml                ← local Redpanda (Kafka-API) broker
├── requirements.txt
└── README.md
```

---

## Dataset

Source: [Retailrocket Recommender System Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

| File | Description |
|---|---|
| `events.csv` | ~2.75M clickstream rows — views, add-to-carts, transactions |
| `item_properties_part1.csv` | Item metadata part 1 (~11M rows) |
| `item_properties_part2.csv` | Item metadata part 2 (~9.2M rows) |
| `category_tree.csv` | Category hierarchy |

---

## Notebook 1 — ETL (Completed)

### What was done

The raw Retailrocket CSVs were ingested and cleaned using PySpark on Google Colab. The following transformations were applied:

- Unix millisecond timestamps converted to calendar dates and decomposed into `day_of_week`, `day_of_month`, `month`, `week_of_year`
- Both item property files unioned and the most recent `categoryid` extracted per item
- A deterministic `base_price` engineered per SKU using `hash(itemid)` mapped to the range [5.00, 500.00] — Retailrocket anonymises real prices
- Raw clickstream events aggregated into one row per `(itemid, date)` with daily counts for transactions, views, add-to-carts, and unique visitors
- Conversion ratio features computed: `view_to_cart_ratio` and `cart_to_purchase_ratio`
- 7-day and 30-day rolling sales velocity features built using Spark range-based Window functions — strictly backward-looking, no future leakage
- A strict **chronological 80/20 split on calendar dates** applied — the boundary date sits at the 80th percentile of all dates in the dataset

### Outputs written to `data/processed/`

| Output | Description |
|---|---|
| `train/` (Parquet) | 80% earliest data — used for model training |
| `test/` (Parquet) | 20% latest data — used for evaluation and Kafka streaming |
| `raw_sample.csv` | First 2000 rows of the test slice — for local Phase 2 sanity tests |

### Full column list

| Column | Type | Description |
|---|---|---|
| `itemid` | int | Product identifier (SKU) |
| `event_date` | date | Calendar date |
| `day_of_week` | int | 1 = Sunday, 7 = Saturday |
| `day_of_month` | int | Day number within the month |
| `month` | int | Month number |
| `week_of_year` | int | ISO week number |
| `categoryid` | int | Most recent product category (0 = unknown) |
| `base_price` | double | Deterministic price in [5.00, 500.00] |
| `daily_transactions` | int | **Target variable** — purchases that day |
| `daily_views` | int | Page views that day |
| `daily_addtocarts` | int | Add-to-cart events that day |
| `daily_unique_visitors` | int | Distinct visitors that day |
| `view_to_cart_ratio` | double | `addtocarts / views` |
| `cart_to_purchase_ratio` | double | `transactions / addtocarts` |
| `velocity_7d` | double | 7-day rolling avg of daily transactions |
| `velocity_30d` | double | 30-day rolling avg of daily transactions |
| `view_velocity_7d` | double | 7-day rolling avg of daily views |

---

## Notebook 2 — GBT Demand Forecasting (Completed)

### What this notebook will do

Train a single global `GBTRegressor` to predict `daily_transactions` for any SKU on any future date. One model covers all products simultaneously using `StringIndexer` to encode item and category identifiers — no individual model per product.

### What to expect from the data

When you load `data/processed/train/` you will see all the columns listed above. The key things to understand before touching the model:

- **Target column** is `daily_transactions` — this is what the model predicts
- `daily_views`, `daily_addtocarts`, `view_to_cart_ratio`, and `cart_to_purchase_ratio` **cannot be used directly as features** — they reflect what happened during the day, not before it. At prediction time (start of day) you do not have these values yet

### Required preprocessing before training

**Step 1 — Add lag features**

Replace today's view and cart counts with yesterday's values using a lag window:

```python
lag_window = Window.partitionBy("itemid").orderBy("event_date")

df = df.withColumn("views_lag1",     F.lag("daily_views", 1).over(lag_window))
       .withColumn("addtocart_lag1", F.lag("daily_addtocarts", 1).over(lag_window))
```

**Step 2 — Drop columns that would cause leakage**

```python
cols_to_drop = ["daily_views", "daily_addtocarts", "view_to_cart_ratio", "cart_to_purchase_ratio"]
```

**Step 3 — Encode categorical columns**

`itemid` and `categoryid` are integers but the model must treat them as categories, not continuous numbers:

```python
StringIndexer(inputCol="itemid",     outputCol="itemid_index")
StringIndexer(inputCol="categoryid", outputCol="categoryid_index")
```

**Step 4 — Drop null lag rows**

The very first date per item will have null lag values (no yesterday exists). Drop these before training.


**Target Column Sparse issue**
Note: the target column daily_transactions has most of its values zeros which will make the model you use lazy and still get high accuracy so we need to work only on items that have transactions before in order to have meaningful rows

```python
# Keep only SKUs that made at least one sale in the training window
items_with_sales = train_df.filter(F.col("daily_transactions") > 0) \
                            .select("itemid").distinct()

train_df = train_df.join(items_with_sales, on="itemid", how="inner")
```

### Final feature set going into the GBT

```
itemid_index, categoryid_index, base_price,
day_of_week, day_of_month, month, week_of_year,
velocity_7d, velocity_30d, view_velocity_7d,
views_lag1, addtocart_lag1
```

### Output to save

- `models/baseline_config.json` — the trained model parameters and feature list, kept as a documentation artifact (Phase 2 has no Spark, so this model never runs live there)

### Results

| Metric | Value |
|---|---|
| RMSE | 0.1793 |
| MAE | 0.0455 |
| R² | 0.2998 |

**Known limitation — zero-inflation.** 96.65% of test rows have zero actual transactions, since most SKU-days simply have no sale. The model learns which items are likely to sell (`velocity_7d` dominates feature importance) but not how much — on the 1,516 test rows with a real sale, error jumps to RMSE 0.83 / MAE 0.72 against mostly-1 true values. A `log1p` target transform was tried and measured; it left R²/RMSE essentially unchanged, because `log1p` corrects right-skew, not zero-inflation, and is nearly linear over mostly-{0,1} values. The correct fix would be a two-stage hurdle model (classify "sells today?" then regress magnitude on the positive rows only) — out of scope for this baseline pass. Accepted as-is: this is an academic portfolio project, the baseline's purpose is to demonstrate the pipeline and the diagnostic process, and the LSTM (notebook 3) is the actual real-time workhorse for the Phase 2 streaming simulation.

---

## Notebook 3 — LSTM Surge Detection (Completed)

### What this notebook will do

Train an LSTM network to classify whether a sequence of recent daily activity represents a **normal browsing pattern** or an **exponential traffic surge**. This is a binary classification task, not a regression. The output is a surge probability per time window that the Phase 2 consumer engine uses to trigger dynamic price adjustments.

### What to expect from the data

The LSTM does not work with flat tabular rows like the GBT. It needs **sequences** — a sliding window of N consecutive days per item stacked into a 3D tensor of shape `[samples, time_steps, features]`.

### Required preprocessing before training

**Step 1 — Engineer the surge label**

A day is labelled a surge if its view count exceeds 2 standard deviations above that item's historical mean:

```python
item_stats = train_df.groupBy("itemid").agg(
    F.mean("daily_views").alias("mean_views"),
    F.stddev("daily_views").alias("std_views")
)

df = df.join(item_stats, on="itemid") \
       .withColumn("is_surge", 
           F.when(F.col("daily_views") > 
                 (F.col("mean_views") + 2 * F.col("std_views")), 1
           ).otherwise(0)
       )
```

**Step 2 — Select sequence features**

Only these columns go into the LSTM input tensor — keep it tight to avoid noise:

```
daily_views, daily_addtocarts, velocity_7d, day_of_week, base_price
```

**Step 3 — Normalise features**

Scale all numeric features to [0, 1] using min-max scaling fitted only on the train set. Apply the same scaler to the test set — do not refit on test.

**Step 4 — Build sliding windows**

Use a window of 7 time steps (7 days of history) with a stride of 1. For each item, slide across its sorted daily rows to produce labelled samples:

```
days [1–7]  → label from day 7  → sample 1
days [2–8]  → label from day 8  → sample 2
days [3–9]  → label from day 9  → sample 3
```

The label for the window is the `is_surge` value of the **last day** in that window.

**Step 5 — Final tensor shape**

```python
X_train shape: [n_samples, 7, 5]   # 7 time steps, 5 features
y_train shape: [n_samples]          # binary: 0 or 1
```

### Output to save

- `models/lstm_model.pth` — trained LSTM weights loaded by `consumer_engine.py` in Phase 2
- `models/lstm_config.json` — feature order, per-feature min/max (for normalising live stream data identically), architecture, surge-label rule, and decision threshold. Phase 2 has no Spark, so this file is the only place those stats are recorded.

### Results

| Metric | Value |
|---|---|
| Accuracy | 0.8963 |
| Precision | 0.4480 |
| Recall | 0.7009 |
| F1 | 0.5466 |

Confusion matrix on 55,105 test windows (8.92% true surge rate): **3,444 true positives, 1,470 false negatives, 4,244 false positives, 45,947 true negatives.** Recall > precision is by design — `BCEWithLogitsLoss(pos_weight=...)` deliberately biases the loss toward catching surges (missing one means no price adjustment happens when it should), at the cost of more false alarms (55% of raised alerts). The label itself is a `daily_views > mean + 2σ` threshold, and `daily_views` is one of the 5 sequence inputs, so the model leans heavily on the current day's view count — the same dynamic as `velocity_7d` dominating the GBT's feature importances.

---

## Notebook 4 — Product Relationship Graph (Completed)

### What this notebook does

Builds a weighted co-occurrence graph of products from raw `events.csv` (not notebook 1's daily aggregation — session structure lives at the individual-event level). Restricts to `addtocart`/`transaction` events (a much stronger "these items are related" signal than views), sessionizes per visitor with a 30-minute inactivity gap, and caps basket size at 20 items before generating co-occurrence pairs via a `session_id` self-join — safe because each join key only matches within one small, capped basket, unlike a lifetime-history self-join which would blow up quadratically.

`relationships.json` is derived directly from the weighted edges DataFrame, so it doesn't depend on `GraphFrame`/PageRank succeeding — GraphFrames ships a separate JVM package per Spark/Scala build that can't be verified outside Colab, so the notebook's actual deliverable is structured to survive that step failing. PageRank runs on top as the graph-algorithm showcase (surfaces items structurally central to co-purchase behavior, not just individually popular ones).

### Output to save

- `models/relationships.json` — top-5 related items per SKU with co-occurrence weight, keyed by `str(itemid)`. Phase 2 scales a related item's price bump proportionally to its weight rather than applying a flat adjustment to every related item.

---

## Phase 2 — Local Streaming Simulation

Runs entirely in plain Python — no Spark or Hadoop required locally. All pieces below were built and verified end-to-end against a real local Kafka broker (Redpanda via Docker), not just written and handed off.

### Fixing `raw_sample.csv` first

Notebook 1's original `raw_sample.csv` (`test_df.orderBy(event_date, itemid).limit(2000)`) only captured the *first* calendar day of the test window — one row per item, no history at all. The LSTM needs a 7-day rolling window per item before it can predict anything, so that file could never trigger a surge during streaming.

`simulation/prepare_stream_sample.py` regenerates it locally from the full `data/processed/test/` Parquet (no Spark needed — it's already on disk): selects items with ≥7 days of history (so LSTM windows can form), ranked by view volume, plus their related items from `relationships.json` (so price propagation has something to show), keeping every available row for those items across the full 28-day test window.

```bash
python simulation/prepare_stream_sample.py
```

Produces ~415 items / ~5,500 rows spanning the full test date range, versus the original 2,000 rows / 1 day.

### Price-adjustment logic

`simulation/pricing_engine.py` holds the model-agnostic, Kafka-agnostic core (`PricingEngine`): per-item rolling window, LSTM inference, and price propagation through `relationships.json`. It's shared by the offline dry run and the live Kafka consumer so the logic is identical either way.

- **On a surge** (`sigmoid(logits) >= decision_threshold` from `lstm_config.json`): the item's price bumps by `price_bump_pct` (10%).
- **Related items** (from `relationships.json`) get a smaller bump, scaled by `related_bump_scale` (0.5) and by their co-occurrence weight relative to the strongest related item.
- **Guardrails against runaway compounding** — an item that keeps re-triggering would otherwise bump every single day it stays "hot," and related bumps stack on top of each other. A `cooldown_days` (3) suppresses re-bumping the same item too soon, and a hard price cap (`price_cap_multiplier` = 1.5× base for direct surges, 1.25× for related items) bounds the outcome regardless. Verified empirically: on the regenerated sample, 505 surge events / 1,159 related bumps produced clean step trajectories that cap out at exactly +50%/+25%, not unbounded hockey sticks.
- Feature vectors are built by iterating `lstm_config.json`'s `sequence_features` list **in order** — never hardcoded — so the tensor's feature axis can't silently drift from what the model was trained on.
- **A detected surge is always logged, even at the price cap.** An earlier version only logged a `"surge"` entry when the price actually moved — but an item already sitting at its cap (pushed there by *related* bumps from other items' surges) would then have a real, model-detected surge silently vanish: no log entry, no chart marker, no alert. Fixed so surge *detection* is always recorded regardless of whether there's headroom left to move the price.

### Running it

**Offline (no Docker, fastest way to inspect the logic):**

```bash
python simulation/run_dry_run.py
```

**Live, through Kafka (matches the project's "streaming" framing):**

```bash
docker compose up -d       # starts a local Redpanda broker on localhost:9092
```

Then, in **two separate terminals** (the consumer waits for messages, so start it first):

```bash
# terminal 1
python simulation/consumer_engine.py

# terminal 2
python simulation/producer.py
```

Each producer run resets the Kafka topic first, so re-running the demo always starts clean. If you run the consumer a second time *without* a fresh producer run, it correctly reports 0 rows consumed and refuses to overwrite `price_history.csv` with empty results — that guard was tested, not assumed.

Both paths write `simulation/output/price_history.csv` (every price change: timestamp, item, old→new price, trigger reason, source item, confidence, co-occurrence weight) and produce identical results — confirmed by running both against the same sample.

| Script | Role |
|---|---|
| `simulation/prepare_stream_sample.py` | Regenerates a proper multi-day `raw_sample.csv` locally from the full test Parquet |
| `simulation/pricing_engine.py` | Shared surge-detection + price-propagation logic (`PricingEngine`), Kafka-agnostic |
| `simulation/run_dry_run.py` | Runs the pricing engine directly over `raw_sample.csv`, no Docker needed |
| `simulation/producer.py` | Streams `raw_sample.csv` row-by-row into the Kafka topic, chronologically |
| `simulation/consumer_engine.py` | Consumes the Kafka stream, runs the pricing engine, fires alerts, writes the price log |
| `simulation/notifier.py` | Prints a console alert on every surge; sends real SMTP email too if `.env` has `SMTP_HOST`/`PORT`/`USER`/`PASSWORD`/`ALERT_RECIPIENT` set — falls back to console-only if not configured or if sending fails |
| `simulation/visualize_results.py` | Charts each top surging item's price trajectory (as % change from base, so items with different price scales are comparable) alongside its related items, plus a surge-count timeline |
| `simulation/webapp.py` | Streamlit dashboard: interactive Plotly chart + sortable price table over the same pricing logic |

### Charts

```bash
python simulation/visualize_results.py
```

Writes to `simulation/output/charts/`: one PNG per top surging item (its own price line plus its related items', surge points annotated with model confidence) and one `surge_timeline.png` showing surge activity across the simulated 28-day window.

### Web dashboard

```bash
streamlit run simulation/webapp.py
```

A "Run Simulation" button, sliders for the pricing policy (surge bump %, related bump scale, cooldown, price cap), an interactive Plotly chart of price change over time for selected items (star markers = surge points, hover for model confidence), KPI counters, and a sortable table of every item's current price.

Deliberately **not** a live Kafka consumer — it drives `PricingEngine` directly over `raw_sample.csv` (the same path as `run_dry_run.py`), rather than duplicating the Kafka consumer inside a Streamlit script that reruns on every widget interaction, which would reintroduce the consumer-group offset-commit bug that `consumer_engine.py`'s guard exists to catch. The CLI producer/consumer pair is the place to see the literal Kafka mechanics; this dashboard is a view over the same output.

Install local dependencies with:

```bash
pip install -r requirements.txt
```
