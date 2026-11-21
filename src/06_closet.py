#!/usr/bin/env python3
"""Mine frequent closed patterns with recursive CLOSET-style conditional projection."""

import csv
import itertools
import math
import re
import signal
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "transactions_sel.csv"
MIN_SUP = 0.3
MIN_CONF = 0.7

def sweep_timeout_handler(signum, frame):
    """Raise a bounded-run timeout; signal/frame are passed by operating system alarm callback."""
    raise TimeoutError("CLOSET sweep run exceeded 120 seconds")


def load_transactions(path):
    """Read comma-joined transaction items; path points to ARM transactions."""
    with path.open(newline="", encoding="utf-8") as stream:
        boundary = re.compile(r",(?=(?:age|trestbps|chol|thalach|oldpeak|sex|cp|fbs|restecg|exang|slope|disease)=)")
        return [tuple(item for item in boundary.split(row["items"]) if item)
                for row in csv.DictReader(stream)]


def support_count(transactions, itemset):
    """Count transactions containing itemset; transactions are rows and itemset is a frozenset."""
    return sum(itemset.issubset(transaction) for transaction in transactions)


def support_map_from_apriori(transactions, minimum_support):
    """Run mlxtend Apriori to get reference supports; inputs are transactions and support threshold."""
    import pandas as pd
    from mlxtend.frequent_patterns import apriori
    vocabulary = sorted({item for transaction in transactions for item in transaction})
    matrix = [{item: item in transaction for item in vocabulary} for transaction in transactions]
    result = apriori(pd.DataFrame(matrix), min_support=minimum_support, use_colnames=True)
    return {frozenset(itemset): float(support) for support, itemset in zip(result["support"], result["itemsets"])}


def is_subsumed(itemset, support, found):
    """Check same-support superset pruning; itemset/support are candidate and found maps closed candidates."""
    candidate = frozenset(itemset)
    return any(candidate < superset and math.isclose(support, other_support, abs_tol=1e-12, rel_tol=0)
               for superset, other_support in found.items())


def record_closed_candidate(itemset, support_count_value, total, found):
    """Store candidate unless equal-support closed superset subsumes it; arguments carry support records."""
    key = frozenset(itemset)
    support = support_count_value / total
    if is_subsumed(key, support, found):
        return
    for prior in list(found):
        if prior < key and math.isclose(found[prior], support, abs_tol=1e-12, rel_tol=0):
            del found[prior]
    found[key] = support


def closet_mine(transactions, minimum_support):
    """Mine closed itemsets by descending f-list, conditional DB projection, merging, and support pruning."""
    total = len(transactions)
    threshold = math.ceil(minimum_support * total - 1e-12)
    global_counts = Counter(item for transaction in transactions for item in set(transaction))
    f_list = sorted((item for item, count in global_counts.items() if count >= threshold),
                    key=lambda item: (-global_counts[item], item))
    rank = {item: index for index, item in enumerate(f_list)}
    filtered = [tuple(sorted((item for item in transaction if item in rank), key=rank.get))
                for transaction in transactions]
    closed = {}

    def mine(prefix, projected_db, local_f_list):
        """Mine one projected database; prefix is suffix pattern, projected_db rows precede suffix items."""
        if not projected_db:
            return
        local_counts = Counter(item for transaction in projected_db for item in set(transaction))
        frequent_local = [item for item in local_f_list
                          if local_counts[item] >= threshold and item in local_counts]
        common = [item for item in frequent_local if local_counts[item] == len(projected_db)]
        merged_prefix = frozenset(set(prefix) | set(common))
        if merged_prefix:
            record_closed_candidate(merged_prefix, len(projected_db), total, closed)
        if common:
            common_set = set(common)
            projected_db = [tuple(item for item in transaction if item not in common_set)
                            for transaction in projected_db]
            local_counts = Counter(item for transaction in projected_db for item in set(transaction))
            frequent_local = [item for item in frequent_local if item not in common_set]
        for item in reversed(frequent_local):
            item_count = local_counts[item]
            child_prefix = frozenset(set(merged_prefix) | {item})
            record_closed_candidate(child_prefix, item_count, total, closed)
            conditional_db = []
            for transaction in projected_db:
                if item in transaction:
                    index = transaction.index(item)
                    conditional_db.append(transaction[:index])
            if not conditional_db:
                continue
            conditional_counts = Counter(value for transaction in conditional_db for value in set(transaction))
            conditional_f_list = [value for value in frequent_local if value != item and
                                  conditional_counts[value] >= threshold]
            mine(child_prefix, conditional_db, conditional_f_list)

    mine(frozenset(), filtered, f_list)
    closed = {itemset: support for itemset, support in closed.items()
              if not any(itemset < superset and math.isclose(support, other_support, abs_tol=1e-12, rel_tol=0)
                         for superset, other_support in closed.items())}
    return closed, global_counts


def reference_closed(frequent):
    """Filter Apriori supports to closed sets; frequent maps itemsets to support."""
    return {itemset: support for itemset, support in frequent.items()
            if not any(itemset < superset and math.isclose(support, other_support, abs_tol=1e-12, rel_tol=0)
                       for superset, other_support in frequent.items())}


def verify_closed(transactions, actual, minimum_support):
    """Compare CLOSET output with closed Apriori reference; transactions and actual share support threshold."""
    frequent = support_map_from_apriori(transactions, minimum_support)
    expected = reference_closed(frequent)
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    support_diff = [(itemset, actual[itemset], expected[itemset]) for itemset in set(actual) & set(expected)
                    if not math.isclose(actual[itemset], expected[itemset], abs_tol=1e-12, rel_tol=0)]
    if not missing and not extra and not support_diff:
        print(f"PASS: CLOSET matches Apriori closed-set reference ({len(expected)} closed itemsets; supports identical)")
        return frequent, True
    print("FAIL: CLOSET/reference differences")
    print("missing:", [sorted(value) for value in sorted(missing, key=lambda x: (len(x), sorted(x)))])
    print("extra:", [sorted(value) for value in sorted(extra, key=lambda x: (len(x), sorted(x)))])
    print("support differences:", support_diff)
    return frequent, False


def write_itemsets(itemsets, path):
    """Save closed itemsets and support; itemsets is mapping and path is CSV destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["itemset", "support"])
        for itemset, support in sorted(itemsets.items(), key=lambda entry: (len(entry[0]), sorted(entry[0]))):
            writer.writerow([",".join(sorted(itemset)), f"{support:.12g}"])


def make_closed_rules(closed, transactions, minimum_confidence):
    """Generate rules from closed unions; transactions support arbitrary antecedent/consequent subsets."""
    rules = []
    for itemset, support in closed.items():
        if len(itemset) < 2:
            continue
        ordered = sorted(itemset)
        for size in range(1, len(itemset)):
            for antecedent_tuple in itertools.combinations(ordered, size):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_support = support_count(transactions, antecedent) / len(transactions)
                consequent_support = support_count(transactions, consequent) / len(transactions)
                confidence = support / antecedent_support if antecedent_support else 0
                if confidence >= minimum_confidence and consequent_support:
                    rules.append({"antecedent": ",".join(sorted(antecedent)),
                                  "consequent": ",".join(sorted(consequent)), "support": support,
                                  "confidence": confidence, "lift": confidence / consequent_support})
    unique = {(rule["antecedent"], rule["consequent"]): rule for rule in rules}
    return sorted(unique.values(), key=lambda rule: (-rule["lift"], -rule["confidence"], rule["antecedent"]))


def write_rules(rules, path):
    """Write rule metrics; rules is ordered sequence of dictionaries and path is output CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["antecedent", "consequent", "support", "confidence", "lift"])
        writer.writeheader()
        for rule in rules:
            writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value
                             for key, value in rule.items()})


def sweep_supports(transactions):
    """Compare CLOSET and Apriori counts over required thresholds; transactions are selected ARM rows."""
    rows = []
    for minimum_support in (0.1, 0.15, 0.2, 0.3, 0.05, 0.03, 0.02):
        signal.signal(signal.SIGALRM, sweep_timeout_handler)
        signal.alarm(120)
        try:
            closed, _ = closet_mine(transactions, minimum_support)
            frequent, passed = verify_closed(transactions, closed, minimum_support)
            all_rules = make_closed_rules(frequent, transactions, MIN_CONF)
            closed_rules = make_closed_rules(closed, transactions, MIN_CONF)
            size_counts = Counter(len(itemset) for itemset in closed)
            size_text = ";".join(f"{size}:{count}" for size, count in sorted(size_counts.items()))
            rows.append({"min_sup": minimum_support, "frequent": len(frequent), "closed": len(closed),
                         "compression_ratio": len(frequent) / len(closed) if closed else 0,
                         "rules": len(all_rules), "closed_rules": len(closed_rules),
                         "closed_by_size": size_text, "reference": "PASS" if passed else "FAIL"})
            if minimum_support == 0.15:
                print("Top 10 closed rules by lift at min_sup=0.15:")
                for rule in closed_rules[:10]:
                    print(f"{rule['antecedent']} -> {rule['consequent']} | support={rule['support']:.6f} confidence={rule['confidence']:.6f} lift={rule['lift']:.6f}")
        except TimeoutError:
            with (ROOT / "out" / "log.md").open("a", encoding="utf-8") as stream:
                stream.write(f"\nCLOSET sweep skipped | min_sup={minimum_support} | 120 second cap reached.\n")
            rows.append({"min_sup": minimum_support, "frequent": "SKIP", "closed": "SKIP",
                         "compression_ratio": "SKIP", "rules": "SKIP", "closed_rules": "SKIP",
                         "closed_by_size": "SKIP", "reference": "SKIP"})
            print(f"SKIP: CLOSET min_sup={minimum_support}; 120 second cap reached")
        finally:
            signal.alarm(0)
    return rows


def main():
    """Mine, verify, and save CLOSET closed sets and rule metrics; constants define required thresholds."""
    transactions = load_transactions(INPUT)
    closed, _ = closet_mine(transactions, MIN_SUP)
    frequent, passed = verify_closed(transactions, closed, MIN_SUP)
    write_itemsets(closed, ROOT / "out" / "tables" / "closet_itemsets.csv")
    rules = make_closed_rules(closed, transactions, MIN_CONF)
    write_rules(rules, ROOT / "out" / "rules" / "closed_rules.csv")
    total_count, closed_count = len(frequent), len(closed)
    ratio = total_count / closed_count if closed_count else 0.0
    print(f"All frequent itemsets: {total_count}; closed itemsets: {closed_count}; closed rules: {len(rules)}")
    print(f"Compression ratio (all/closed): {ratio:.6f}")
    if not passed:
        raise SystemExit(1)
    sweep = sweep_supports(transactions)
    sweep_path = ROOT / "out" / "tables" / "closet_sweep.csv"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with sweep_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    print("CLOSET sweep: min_sup | frequent | closed | compression_ratio | rules | closed_rules | closed_by_size | reference")
    for row in sweep:
        print(" | ".join(str(row[key]) for key in row))


if __name__ == "__main__":
    main()
