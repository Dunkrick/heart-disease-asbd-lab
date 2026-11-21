#!/usr/bin/env python3
"""Mine frequent itemsets with Dynamic Itemset Counting and verify with mlxtend."""

import csv
import importlib.util
import itertools
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "transactions_sel.csv"
MIN_SUP = 0.3
MIN_CONF = 0.7
INTERVAL = 10
SENSITIVITY_M = (2, 10)
SENSITIVITY_SUP = (0.2, 0.3, 0.4)
FONT_COLORS = {
    "dashed_circle": (215, 133, 40),
    "dashed_box": (70, 145, 195),
    "solid_box": (55, 150, 95),
    "solid_circle": (190, 83, 83),
}
STATE_ORDER = ("dashed_circle", "dashed_box", "solid_box", "solid_circle")
STATE_LABELS = ("DASHED CIRCLE", "DASHED BOX", "SOLID BOX", "SOLID CIRCLE")


def read_transactions(path):
    """Read comma-joined transactions; path names a one-column `items` CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        boundary = re.compile(r",(?=(?:age|trestbps|chol|thalach|oldpeak|sex|cp|fbs|restecg|exang|slope|disease)=)")
        return [tuple(item for item in boundary.split(row["items"]) if item)
                for row in csv.DictReader(stream)]


def state_count(states, label):
    """Count itemsets in one state; states maps tuples to state records, label is state name."""
    return sum(record["state"] == label for itemset, record in states.items() if itemset)


def candidate_supersets(states):
    """Generate untracked immediate supersets whose every immediate subset is a square."""
    square_names = {"dashed_box", "solid_box"}
    by_size = {}
    for itemset, record in states.items():
        if itemset and record["state"] in square_names:
            by_size.setdefault(len(itemset), []).append(itemset)
    generated = set()
    for size, itemsets in by_size.items():
        for first, second in itertools.combinations(itemsets, 2):
            union = tuple(sorted(set(first) | set(second)))
            if len(union) != size + 1 or union in states:
                continue
            if all(tuple(subset) in states and states[tuple(subset)]["state"] in square_names
                   for subset in itertools.combinations(union, size)):
                generated.add(union)
    return generated


def run_dic(transactions, min_support, interval):
    """Count itemsets dynamically; transactions, threshold, and interval control scan schedule."""
    total = len(transactions)
    minimum_count = min_support * total
    items = sorted({item for transaction in transactions for item in transaction})
    states = {(): {"state": "solid_box", "count": total, "seen": total}}
    for item in items:
        states[(item,)] = {"state": "dashed_circle", "count": 0, "seen": 0}
    pointer, total_read, intervals = 0, 0, []
    while any(record["state"].startswith("dashed") for record in states.values()):
        read_this_interval = 0
        for _ in range(interval):
            transaction = set(transactions[pointer])
            pointer = (pointer + 1) % total
            total_read += 1
            read_this_interval += 1
            for itemset, record in list(states.items()):
                if not record["state"].startswith("dashed") or record["seen"] >= total:
                    continue
                if set(itemset).issubset(transaction):
                    record["count"] += 1
                record["seen"] += 1
                if record["state"] == "dashed_circle" and record["count"] >= minimum_count:
                    record["state"] = "dashed_box"
                if record["seen"] == total:
                    record["state"] = "solid_box" if record["count"] >= minimum_count else "solid_circle"
        for candidate in candidate_supersets(states):
            states[candidate] = {"state": "dashed_circle", "count": 0, "seen": 0}
        intervals.append({
            "interval": len(intervals) + 1,
            "transactions_read": read_this_interval,
            "dashed_circle": state_count(states, "dashed_circle"),
            "dashed_box": state_count(states, "dashed_box"),
            "solid_box": state_count(states, "solid_box"),
            "solid_circle": state_count(states, "solid_circle"),
            "total_read": total_read,
        })
        if len(intervals) > 100000:
            raise RuntimeError("DIC did not converge within 100000 intervals")
    frequent = {itemset: record["count"] / total for itemset, record in states.items()
                if itemset and record["state"] == "solid_box"}
    return frequent, intervals, total_read, states


def make_rules(frequent, minimum_confidence):
    """Generate association rules; frequent maps itemsets to support and threshold filters confidence."""
    rules = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        items = set(itemset)
        for size in range(1, len(itemset)):
            for antecedent_tuple in itertools.combinations(itemset, size):
                antecedent = frozenset(antecedent_tuple)
                consequent = frozenset(items - antecedent)
                antecedent_support = frequent.get(tuple(sorted(antecedent)))
                consequent_support = frequent.get(tuple(sorted(consequent)))
                if not antecedent_support or not consequent_support:
                    continue
                confidence = support / antecedent_support
                if confidence >= minimum_confidence:
                    rules.append({
                        "antecedent": ",".join(sorted(antecedent)),
                        "consequent": ",".join(sorted(consequent)),
                        "support": support,
                        "confidence": confidence,
                        "lift": confidence / consequent_support,
                    })
    return rules


def write_itemsets(frequent, path):
    """Write itemset text and relative support; frequent maps tuples to support values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["itemset", "support"])
        for itemset, support in sorted(frequent.items(), key=lambda entry: (len(entry[0]), entry[0])):
            writer.writerow([",".join(itemset), f"{support:.12g}"])


def write_rules(rules, path):
    """Write rules and support/confidence/lift; rules is a list of rule dictionaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["antecedent", "consequent", "support", "confidence", "lift"])
        writer.writeheader()
        for rule in rules:
            writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value
                             for key, value in rule.items()})


def verify_mlxtend(transactions, frequent, minimum_support):
    """Compare itemsets and supports to mlxtend Apriori; transactions/frequent/threshold are paired inputs."""
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import apriori
    except ImportError as error:
        raise RuntimeError("mlxtend and pandas required for requested DIC verification") from error
    vocabulary = sorted({item for transaction in transactions for item in transaction})
    transaction_sets = [set(transaction) for transaction in transactions]
    frame = pd.DataFrame([{item: item in transaction for item in vocabulary} for transaction in transaction_sets])
    output = apriori(frame, min_support=minimum_support, use_colnames=True)
    reference = {tuple(sorted(itemset)): float(support) for support, itemset in zip(output["support"], output["itemsets"])}
    missing = sorted(set(reference) - set(frequent), key=lambda item: (len(item), item))
    extra = sorted(set(frequent) - set(reference), key=lambda item: (len(item), item))
    support_diff = [(itemset, frequent[itemset], reference[itemset]) for itemset in set(frequent) & set(reference)
                    if not math.isclose(frequent[itemset], reference[itemset], rel_tol=0, abs_tol=1e-12)]
    passed = not missing and not extra and not support_diff
    if passed:
        print(f"PASS: DIC matches mlxtend apriori ({len(reference)} itemsets; supports identical)")
    else:
        print("FAIL: DIC/mlxtend differences")
        print("missing from DIC:", missing)
        print("extra in DIC:", extra)
        print("support differences:", support_diff)
    return passed


def load_raster():
    """Load PNG raster primitives from cleaner; existing module owns 300-DPI encoder."""
    spec = importlib.util.spec_from_file_location("heart_cleaner", ROOT / "src" / "01_clean.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FONT.update({
        "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
        "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
        ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    })
    return module.Raster


def draw_progress(intervals, path):
    """Plot counts in all four DIC states; intervals are logged boundaries and path is PNG destination."""
    Raster = load_raster()
    image = Raster(1600, 950)
    image.text(85, 25, "DIC ITEMSET STATES OVER INTERVALS M=10", scale=3)
    image.text(25, 105, "ITEMSET COUNT", scale=1)
    image.text(760, 865, "INTERVAL", scale=2)
    x0, y0, x1, y1 = 130, 170, 1450, 800
    maximum = max(max(record[state] for record in intervals) for state in STATE_ORDER)
    maximum = max(maximum, 1)
    for tick in range(6):
        fraction = tick / 5
        yy = int(y1 - fraction * (y1 - y0))
        image.line(x0, yy, x1, yy, (220, 225, 230))
        image.text(45, yy - 4, str(round(maximum * fraction)), scale=1)
    tick_indices = sorted(set([0, len(intervals) // 4, len(intervals) // 2, (3 * len(intervals)) // 4, len(intervals) - 1]))
    for index in tick_indices:
        xx = int(x0 + index * (x1 - x0) / max(1, len(intervals) - 1))
        image.line(xx, y1, xx, y1 + 10, (25, 35, 45))
        label = str(intervals[index]["interval"])
        image.text(xx - len(label) * 3, y1 + 14, label, scale=1)
    image.line(x0, y0, x0, y1, (25, 35, 45))
    image.line(x0, y1, x1, y1, (25, 35, 45))
    for state in STATE_ORDER:
        points = []
        for index, record in enumerate(intervals):
            x = int(x0 + index * (x1 - x0) / max(1, len(intervals) - 1))
            y = int(y1 - record[state] * (y1 - y0) / maximum)
            points.append((x, y))
        for first, second in zip(points, points[1:]):
            image.line(first[0], first[1], second[0], second[1], FONT_COLORS[state])
        for x, y in points:
            image.rect(x - 3, y - 3, x + 4, y + 4, FONT_COLORS[state])
    for index, state in enumerate(STATE_ORDER):
        yy = 835 + index * 25
        image.rect(1080, yy, 1098, yy + 15, FONT_COLORS[state])
        image.text(1110, yy, STATE_LABELS[index], scale=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def apriori_pass_count(frequent):
    """Estimate classic Apriori scans from maximum frequent size; frequent is itemset/support mapping."""
    return max((len(itemset) for itemset in frequent), default=1)


def levelwise_apriori_reads(transactions, min_support):
    """Estimate classic Apriori level scans times N; inputs are transactions and minimum support."""
    frequent, _, _, _ = run_dic(transactions, min_support, INTERVAL)
    return apriori_pass_count(frequent) * len(transactions)


def sensitivity_table(transactions):
    """Run DIC parameter sweeps; transactions is shared base and returned rows are printable metrics."""
    rows = []
    for interval in SENSITIVITY_M:
        frequent, log, reads, _ = run_dic(transactions, MIN_SUP, interval)
        rows.append(["M", interval, reads, len(log), len(frequent), len(make_rules(frequent, MIN_CONF))])
    for support in SENSITIVITY_SUP:
        frequent, log, reads, _ = run_dic(transactions, support, INTERVAL)
        rows.append(["min_sup", support, reads, len(log), len(frequent), len(make_rules(frequent, MIN_CONF))])
    return rows


def main():
    """Mine selected ARM transactions, verify supports, save rules, and report sensitivity."""
    transactions = read_transactions(INPUT)
    frequent, intervals, total_read, states = run_dic(transactions, MIN_SUP, INTERVAL)
    interval_path = ROOT / "out" / "tables" / "dic_intervals.csv"
    interval_path.parent.mkdir(parents=True, exist_ok=True)
    with interval_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)
    itemset_path = ROOT / "out" / "tables" / "dic_itemsets.csv"
    write_itemsets(frequent, itemset_path)
    rules = make_rules(frequent, MIN_CONF)
    rules.sort(key=lambda rule: (-rule["lift"], -rule["confidence"], -rule["support"], rule["antecedent"], rule["consequent"]))
    write_rules(rules, ROOT / "out" / "rules" / "dic_rules.csv")
    for disease in ("disease=yes", "disease=no"):
        selected = [rule for rule in rules if rule["consequent"] == disease]
        write_rules(selected[:10], ROOT / "out" / "rules" / f"dic_rules_{disease.split('=')[1]}.csv")
        print(f"Top rules with consequent {disease}:")
        if not selected:
            print("(0 rules)")
        for rule in selected[:10]:
            print(f"{rule['antecedent']} -> {rule['consequent']} | support={rule['support']:.6f} confidence={rule['confidence']:.6f} lift={rule['lift']:.6f}")
    draw_progress(intervals, ROOT / "out" / "figs" / "dic_progress.png")
    support_pass = verify_mlxtend(transactions, frequent, MIN_SUP)
    maximum_size = max((len(itemset) for itemset in frequent), default=1)
    passes = apriori_pass_count(frequent)
    print(f"DIC transactions read: {total_read}; Apriori passes × N: {passes} × {len(transactions)} = {passes * len(transactions)}")
    print("Frequent itemsets by size:", dict(sorted(Counter(len(itemset) for itemset in frequent).items())))
    print("Top 10 rules by lift:")
    for rule in rules[:10]:
        print(f"{rule['antecedent']} -> {rule['consequent']} | support={rule['support']:.6f} confidence={rule['confidence']:.6f} lift={rule['lift']:.6f}")
    sensitivity = sensitivity_table(transactions)
    print("Sensitivity: parameter | value | transactions_read | intervals | itemsets | rules")
    for row in sensitivity:
        print(" | ".join(str(value) for value in row))
    print("Read comparison: min_sup | DIC reads | Apriori level-wise passes x N")
    for support in SENSITIVITY_SUP:
        frequent_at_support, _, dic_reads, _ = run_dic(transactions, support, INTERVAL)
        apriori_reads = apriori_pass_count(frequent_at_support) * len(transactions)
        print(f"{support} | {dic_reads} | {apriori_pass_count(frequent_at_support)} x {len(transactions)} = {apriori_reads}")
        if support == 0.2:
            rules_at_support = make_rules(frequent_at_support, MIN_CONF)
            rules_at_support.sort(key=lambda rule: (-rule["lift"], -rule["confidence"], -rule["support"]))
            disease_no = [rule for rule in rules_at_support if rule["consequent"] == "disease=no"][:10]
            write_rules(disease_no, ROOT / "out" / "rules" / "dic_rules_disease_no_sup_0.2.csv")
            print("Top rules with consequent disease=no at min_sup=0.2:")
            if not disease_no:
                print("(0 rules)")
            for rule in disease_no:
                print(f"{rule['antecedent']} -> {rule['consequent']} | support={rule['support']:.6f} confidence={rule['confidence']:.6f} lift={rule['lift']:.6f}")
    print("DIC progress intervals (M=10):", len(intervals), "final states:",
          {state: state_count(states, state) for state in STATE_ORDER})
    if not support_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
