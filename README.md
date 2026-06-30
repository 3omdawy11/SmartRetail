# RetailPrice: Predictive Demand Forecasting & Real-Time Dynamic Pricing Engine

An end-to-end e-commerce optimization pipeline built on the Retailrocket clickstream dataset. It combines offline batch machine learning (demand forecasting) with a real-time streaming engine (traffic surge detection and dynamic price adjustment). The system is split into two phases: heavy computation on Google Colab using PySpark, and a lightweight local simulation using plain Python and Kafka.

---

## Project Structure

```
smart_retail/
├── colab_notebooks/
│   ├── 1_data_preprocessing.ipynb   ← ETL (DONE)
│   ├── 2_train_baseline_ml.ipynb    ← GBT demand forecasting (NEXT)
│   ├── 3_train_lstm_dl.ipynb        ← LSTM surge detection
│   └── 4_build_graphframes.ipynb    ← Product relationship graph
├── data/
│   ├── raw/                         ← Drop raw CSVs here (gitignored)
│   └── processed/                   ← ETL outputs land here (gitignored)
├── models/
│   ├── baseline_config.json
│   └── lstm_model.pth
├── simulation/
│   ├── producer.py
│   ├── consumer_engine.py
│   └── notifier.py
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

## Notebook 2 — GBT Demand Forecasting (Up Next)

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

- `models/baseline_config.json` — the trained model parameters and feature list for use in Phase 2 inference

---

## Notebook 3 — LSTM Surge Detection (Up Next)

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

---

## Phase 2 — Local Streaming Simulation

> Documentation will be added after notebooks 2, 3, and 4 are complete.

Runs entirely in plain Python — no Spark or Hadoop required locally.

| Script | Role |
|---|---|
| `simulation/producer.py` | Reads `raw_sample.csv` row-by-row and pushes to a local Kafka topic |
| `simulation/consumer_engine.py` | Consumes Kafka stream, runs LSTM inference, applies price adjustments using `relationships.json` |
| `simulation/notifier.py` | Fires SMTP email alerts to vendors on surge detection |

Install local dependencies with:

```bash
pip install -r requirements.txt
```
