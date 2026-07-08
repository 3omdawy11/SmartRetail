"""
Core surge-detection + price-adjustment logic, shared by the offline dry run
(run_dry_run.py) and the live Kafka consumer (consumer_engine.py). Kept Kafka-free
so the app logic can be tested as a plain function over a CSV before any broker
is involved.
"""
import csv
import json
import os
from collections import deque

import torch
import torch.nn as nn


class SurgeLSTM(nn.Module):
    """Must match colab_notebooks/3_train_lstm_dl.ipynb exactly - we're loading its weights."""

    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden).squeeze(-1)


def load_model_and_config(models_dir="models"):
    with open(f"{models_dir}/lstm_config.json") as f:
        config = json.load(f)

    model = SurgeLSTM(**config["hyperparameters"])
    state_dict = torch.load(f"{models_dir}/lstm_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with open(f"{models_dir}/relationships.json") as f:
        relationships = json.load(f)

    return model, config, relationships


def build_feature_vector(row, config):
    """Iterates config['sequence_features'] in order - the tensor's feature axis
    must match training exactly, so this list is never hardcoded elsewhere."""
    vec = []
    for feat in config["sequence_features"]:
        stats = config["feature_min_max"][feat]
        lo, hi = stats["min"], stats["max"]
        val = float(row[feat])
        rng = hi - lo
        norm = 0.0 if rng == 0 else min(1.0, max(0.0, (val - lo) / rng))
        vec.append(norm)
    return vec


class PricingEngine:
    """
    Stateful per-item surge detection + price propagation.

    Guardrails against runaway compounding (an item that keeps re-triggering would
    otherwise bump every single day it stays "hot", and related bumps stack on top):
      - cooldown_days: once an item surges, it can't surge-bump again for N of its
        own subsequent rows (does not block other items).
      - price_cap_multiplier / related_price_cap_multiplier: price can never exceed
        that multiple of the item's own base_price, regardless of how many surges fire.
    """

    def __init__(self, model, config, relationships, catalog_base_prices, start_date,
                 price_bump_pct=0.10, related_bump_scale=0.5, cooldown_days=3,
                 price_cap_multiplier=1.5, related_price_cap_multiplier=1.25):
        self.model = model
        self.config = config
        self.relationships = relationships
        self.window_size = config["window_size"]
        self.decision_threshold = config["decision_threshold"]

        self.price_bump_pct = price_bump_pct
        self.related_bump_scale = related_bump_scale
        self.cooldown_days = cooldown_days
        self.price_cap_multiplier = price_cap_multiplier
        self.related_price_cap_multiplier = related_price_cap_multiplier

        self.base_prices = dict(catalog_base_prices)
        self.prices = dict(catalog_base_prices)
        self.windows = {}          # itemid -> deque[list[float]]
        self.cooldowns = {}        # itemid -> int (rows remaining before next surge bump allowed)
        self.seq = 0
        self.log = []

        for itemid, base_price in catalog_base_prices.items():
            self._append_log(start_date, itemid, None, base_price, "initial", itemid, None, None)

    def _append_log(self, event_date, itemid, old_price, new_price, trigger_reason,
                     source_item, confidence, weight):
        self.seq += 1
        self.log.append({
            "seq": self.seq,
            "event_date": event_date,
            "itemid": itemid,
            "old_price": round(old_price, 2) if old_price is not None else None,
            "new_price": round(new_price, 2),
            "pct_change": round((new_price - old_price) / old_price * 100, 2) if old_price else 0.0,
            "trigger_reason": trigger_reason,
            "source_item": source_item,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "weight": weight,
        })

    def process_row(self, row):
        """row: dict with itemid, event_date, and config['sequence_features'] columns.
        Returns the list of new log entries produced by this row (empty if none)."""
        itemid = int(row["itemid"])
        event_date = row["event_date"]
        entries_before = len(self.log)

        if itemid not in self.prices:
            self.prices[itemid] = float(row["base_price"])
            self.base_prices[itemid] = float(row["base_price"])
            self._append_log(event_date, itemid, None, self.prices[itemid], "initial", itemid, None, None)

        if self.cooldowns.get(itemid, 0) > 0:
            self.cooldowns[itemid] -= 1

        window = self.windows.setdefault(itemid, deque(maxlen=self.window_size))
        window.append(build_feature_vector(row, self.config))

        if len(window) == self.window_size:
            x = torch.tensor([list(window)], dtype=torch.float32)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(x)).item()

            if prob >= self.decision_threshold and self.cooldowns.get(itemid, 0) == 0:
                self._apply_surge(itemid, event_date, prob)
                self.cooldowns[itemid] = self.cooldown_days

        return self.log[entries_before:]

    def _apply_surge(self, itemid, event_date, confidence):
        old_price = self.prices[itemid]
        base_price = self.base_prices[itemid]
        cap = base_price * self.price_cap_multiplier
        new_price = min(old_price * (1 + self.price_bump_pct), cap)

        # Always log a detected surge, even if the price is already at its cap
        # (e.g. pushed there by related-item bumps from other items' surges) -
        # otherwise a genuinely-detected surge silently vanishes: no log entry,
        # no chart marker, no alert, just because there was no headroom left.
        self.prices[itemid] = new_price
        self._append_log(event_date, itemid, old_price, new_price, "surge", itemid, confidence, None)

        related_entries = self.relationships.get(str(itemid), [])
        if not related_entries:
            return
        max_weight = max(e["weight"] for e in related_entries)

        for entry in related_entries:
            related_id = entry["related"]
            weight = entry["weight"]
            if related_id not in self.prices:
                # Related item never streamed its own row (outside the sample) - fall
                # back to its own base_price if known, otherwise skip (no price to bump).
                continue

            related_old = self.prices[related_id]
            related_base = self.base_prices[related_id]
            related_cap = related_base * self.related_price_cap_multiplier
            bump = self.price_bump_pct * self.related_bump_scale * (weight / max_weight)
            related_new = min(related_old * (1 + bump), related_cap)

            if related_new > related_old:
                self.prices[related_id] = related_new
                self._append_log(event_date, related_id, related_old, related_new,
                                  "related", itemid, confidence, weight)


def write_log_and_summary(engine, output_csv):
    """Writes the engine's price-change log to CSV and prints a run summary.
    Shared by run_dry_run.py and consumer_engine.py so both report identically."""
    reason_counts = {}
    for entry in engine.log:
        reason_counts[entry["trigger_reason"]] = reason_counts.get(entry["trigger_reason"], 0) + 1

    print(f"Initial items       : {reason_counts.get('initial', 0):,}")
    print(f"Surge events        : {reason_counts.get('surge', 0):,}")
    print(f"Related-bump events : {reason_counts.get('related', 0):,}")
    print(f"Total log entries   : {len(engine.log):,}")

    capped = [
        itemid for itemid, price in engine.prices.items()
        if price >= engine.base_prices[itemid] * engine.price_cap_multiplier * 0.999
    ]
    print(f"Items that hit the price cap: {len(capped):,}")

    price_changes = sorted(
        (
            (itemid, engine.base_prices[itemid], price,
             (price / engine.base_prices[itemid] - 1) * 100)
            for itemid, price in engine.prices.items()
        ),
        key=lambda t: -t[3],
    )
    print("\nTop 10 items by total price increase:")
    for itemid, base, final, pct in price_changes[:10]:
        print(f"  item {itemid:<8} base={base:>7.2f}  final={final:>7.2f}  change={pct:+6.2f}%")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(engine.log[0].keys()))
        writer.writeheader()
        writer.writerows(engine.log)
    print(f"\nWritten to: {output_csv}")
