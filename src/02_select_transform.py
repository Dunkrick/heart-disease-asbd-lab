#!/usr/bin/env python3
"""Rank heart features and generate ARM and distance representations."""

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "clean.csv"
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
SENSITIVITY = ["trestbps", "chol", "thalach", "oldpeak", "fbs", "restecg", "exang"]


def read_csv(path):
    """Load CSV records and headers; path selects source file for repeatable processing."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames)


def mean(values):
    """Return arithmetic mean; values is non-empty numeric sequence."""
    return sum(values) / len(values)


def variance(values):
    """Return population variance; values is non-empty numeric sequence."""
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def pearson(xs, ys):
    """Compute Pearson correlation; xs and ys are equal-length numeric sequences."""
    mx, my = mean(xs), mean(ys)
    dx, dy = [value - mx for value in xs], [value - my for value in ys]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else 0.0


def ranks(values):
    """Assign average ranks for ties; values is numeric sequence for Spearman correlation."""
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def beta_fraction(a, b, x):
    """Evaluate incomplete-beta continued fraction; a/b/x are positive shape and unit-interval inputs."""
    tiny, epsilon, limit = 1e-300, 3e-14, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for step in range(1, limit + 1):
        m2 = 2 * step
        coefficient = step * (b - step) * x / ((qam + m2) * (a + m2))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + step) * (qab + step) * x / ((a + m2) * (qap + m2))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def regularized_beta(x, a, b):
    """Compute I_x(a,b) for Student/F tails; x is in [0,1], a and b positive."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * beta_fraction(a, b, x) / a
    return 1 - front * beta_fraction(b, a, 1 - x) / b


def beta_survival(x, a, b):
    """Compute 1-I_x(a,b) without cancellation; x is in [0,1], a/b are positive."""
    if x <= 0:
        return 1.0
    if x >= 1:
        return 0.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x >= (a + 1) / (a + b + 2):
        return front * beta_fraction(b, a, 1 - x) / b
    return 1 - front * beta_fraction(a, b, x) / a


def gamma_q(a, x):
    """Compute regularized upper incomplete gamma for chi-square p-values; a>0, x>=0."""
    if x <= 0:
        return 1.0
    log_factor = -x + a * math.log(x) - math.lgamma(a)
    epsilon, limit = 3e-14, 500
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(limit):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * epsilon:
                break
        return min(1.0, max(0.0, 1 - total * math.exp(log_factor)))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    result = d
    for index in range(1, limit + 1):
        coefficient = -index * (index - a)
        b += 2
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        result *= delta
        if abs(delta - 1) < epsilon:
            break
    return min(1.0, max(0.0, result * math.exp(log_factor)))


def two_sided_t_p(t_value, degrees):
    """Compute two-sided Student t p-value; t_value is statistic and degrees is df."""
    x = degrees / (degrees + t_value * t_value)
    return regularized_beta(x, degrees / 2, 0.5)


def f_survival(f_value, numerator_df, denominator_df):
    """Compute upper F-tail probability; f statistic and dfs define null distribution."""
    x = numerator_df * f_value / (numerator_df * f_value + denominator_df)
    return beta_survival(x, numerator_df / 2, denominator_df / 2)


def numeric_bins(values, cuts=5):
    """Create equal-frequency bins for empirical MI; values are numeric, cuts sets bin count."""
    sorted_values = sorted(values)
    boundaries = [sorted_values[round((len(values) - 1) * index / cuts)] for index in range(1, cuts)]
    return [sum(value > boundary for boundary in boundaries) for value in values]


def mutual_information(feature_values, labels):
    """Compute empirical discrete MI in nats; feature values and labels are equal-length categories."""
    joint = Counter(zip(feature_values, labels))
    feature_counts, label_counts = Counter(feature_values), Counter(labels)
    total = len(labels)
    return sum((count / total) * math.log(count * total / (feature_counts[x] * label_counts[y]))
               for (x, y), count in joint.items())


def categorical_test(values, labels):
    """Compute Pearson chi-square, p-value, and Cramer's V; values/labels are categorical sequences."""
    categories, targets = sorted(set(values)), sorted(set(labels))
    table = {category: Counter() for category in categories}
    for category, target in zip(values, labels):
        table[category][target] += 1
    target_counts = Counter(labels)
    total = len(labels)
    statistic = 0.0
    for category in categories:
        row_total = sum(table[category].values())
        for target in targets:
            expected = row_total * target_counts[target] / total
            if expected:
                statistic += (table[category][target] - expected) ** 2 / expected
    degrees = max(1, (len(categories) - 1) * (len(targets) - 1))
    p_value = gamma_q(degrees / 2, statistic / 2)
    denominator = total * max(1, min(len(categories) - 1, len(targets) - 1))
    cramer_v = math.sqrt(statistic / denominator) if denominator else 0.0
    return statistic, p_value, cramer_v


def numeric_test(values, labels):
    """Compute point-biserial r and two-sided p; inputs are numeric values and binary labels."""
    binary = [int(label) for label in labels]
    correlation = pearson(values, binary)
    degrees = len(values) - 2
    if abs(correlation) >= 1:
        p_value = 0.0
    else:
        t_value = correlation * math.sqrt(degrees / (1 - correlation * correlation))
        p_value = two_sided_t_p(abs(t_value), degrees)
    return correlation, p_value


def effect_tier(effect_size):
    """Assign effect-size tier; effect_size is absolute r or Cramer's V."""
    if effect_size >= 0.30:
        return "strong"
    if effect_size >= 0.15:
        return "moderate"
    return "weak"


def feature_effect(feature, values, labels):
    """Return absolute association effect size; feature type selects r or Cramer's V test."""
    if feature in NUMERIC:
        correlation, _ = numeric_test([float(value) for value in values], labels)
        return abs(correlation)
    return categorical_test(values, labels)[2]


def write_transactions(path, transactions):
    """Write one comma-joined item row per transaction; path and transaction lists define output."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["items"])
        writer.writerows([[",".join(items)] for items in transactions])


def write_onehot(path, transactions):
    """Write item incidence without an ID feature; path and transactions determine columns/rows."""
    vocabulary = sorted({item for items in transactions for item in items})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(vocabulary)
        for items in transactions:
            item_set = set(items)
            writer.writerow([int(item in item_set) for item in vocabulary])
    return vocabulary


def encode_level(column, value):
    """Map coded categorical value to readable ARM item; column/value identify codebook entry."""
    mappings = {
        "sex": {"0": "F", "1": "M"},
        "cp": {"1": "typical_angina", "2": "atypical_angina", "3": "non_anginal_pain", "4": "asymptomatic"},
        "fbs": {"0": "no", "1": "yes"},
        "restecg": {"0": "normal", "1": "st_t_abnormality", "2": "lv_hypertrophy"},
        "exang": {"0": "no", "1": "yes"},
        "slope": {"1": "upsloping", "2": "flat", "3": "downsloping"},
    }
    return mappings[column].get(str(value), str(value))


def arm_items(row, thalach_edges):
    """Create transaction item strings; row is clean record and edges define thalach tertiles."""
    age, pressure, cholesterol = float(row["age"]), float(row["trestbps"]), float(row["chol"])
    pulse, oldpeak = float(row["thalach"]), float(row["oldpeak"])
    if age < 45:
        age_level = "<45"
    elif age < 55:
        age_level = "45-54"
    elif age < 65:
        age_level = "55-64"
    else:
        age_level = "65+"
    pressure_level = "<120" if pressure < 120 else "120-139" if pressure < 140 else ">=140"
    chol_level = "<200" if cholesterol < 200 else "200-239" if cholesterol < 240 else ">=240"
    pulse_level = "low" if pulse <= thalach_edges[0] else "middle" if pulse <= thalach_edges[1] else "high"
    if oldpeak < 0:
        oldpeak_level = "<0"
    elif oldpeak == 0:
        oldpeak_level = "0"
    elif oldpeak < 2:
        oldpeak_level = "(0,2)"
    else:
        oldpeak_level = ">=2"
    items = [f"age={age_level}", f"trestbps={pressure_level}", f"chol={chol_level}",
             f"thalach={pulse_level}", f"oldpeak={oldpeak_level}"]
    for column in CATEGORICAL:
        value = str(row[column])
        if column == "slope" and value == "-1":
            continue
        items.append(f"{column}={encode_level(column, value)}")
    items.append("disease=yes" if int(row["num_bin"]) else "disease=no")
    return items


def save_ranking(rows, path):
    """Write feature ranking rows; path is destination and rows contain requested report fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", "type", "stat", "p", "MI", "effect_size", "tier",
                         "n_used", "effect_size_observed_only", "n_observed", "keep/weak"])
        writer.writerows(rows)


def run():
    """Run feature ranking and both transformations; read clean data and write requested artifacts."""
    rows, fields = read_csv(INPUT)
    flags, _ = read_csv(ROOT / "data" / "imputed_flags.csv")
    if len(flags) != len(rows):
        raise ValueError("imputed_flags.csv row count must match clean.csv")
    labels = [int(row["num_bin"]) for row in rows]
    numeric_values = {column: [float(row[column]) for row in rows] for column in NUMERIC}
    ranking = []
    effect_sizes = {}
    tiers = {}
    n_used = len(rows)
    for column in NUMERIC:
        values = numeric_values[column]
        correlation, p_value = numeric_test(values, labels)
        effect_size = abs(correlation)
        effect_sizes[column] = effect_size
        tiers[column] = effect_tier(effect_size)
        mi = mutual_information(numeric_bins(values), labels)
        observed_indices = [index for index, flag in enumerate(flags) if flag[f"{column}_imputed"] == "False"] if column in SENSITIVITY else []
        observed_effect = feature_effect(column, [values[index] for index in observed_indices],
                                         [labels[index] for index in observed_indices]) if observed_indices else ""
        ranking.append([column, "numeric", f"point_biserial_r={correlation:.6g}", f"{p_value:.8g}",
                        f"{mi:.8g}", f"{effect_size:.8g}", tiers[column], n_used,
                        f"{observed_effect:.8g}" if observed_indices else "", len(observed_indices) if column in SENSITIVITY else "",
                        "weak" if tiers[column] == "weak" else "keep"])
    categorical_values = {column: [row[column] for row in rows] for column in CATEGORICAL}
    for column in CATEGORICAL:
        statistic, p_value, cramer_v = categorical_test(categorical_values[column], labels)
        effect_size = cramer_v
        effect_sizes[column] = effect_size
        tiers[column] = effect_tier(effect_size)
        mi = mutual_information(categorical_values[column], labels)
        observed_indices = [index for index, flag in enumerate(flags) if flag[f"{column}_imputed"] == "False"] if column in SENSITIVITY else []
        observed_effect = feature_effect(column, [categorical_values[column][index] for index in observed_indices],
                                         [labels[index] for index in observed_indices]) if observed_indices else ""
        ranking.append([column, "categorical", f"chi2={statistic:.6g};Cramers_V={cramer_v:.6g}",
                        f"{p_value:.8g}", f"{mi:.8g}", f"{effect_size:.8g}", tiers[column], n_used,
                        f"{observed_effect:.8g}" if observed_indices else "", len(observed_indices) if column in SENSITIVITY else "",
                        "weak" if tiers[column] == "weak" else "keep"])
    ranking_path = ROOT / "out" / "tables" / "feature_ranking.csv"
    save_ranking(ranking, ranking_path)
    print("FEATURE_RANKING")
    with ranking_path.open(encoding="utf-8") as stream:
        print(stream.read().rstrip())
    spearman_pairs = []
    for index, first in enumerate(NUMERIC):
        for second in NUMERIC[index + 1:]:
            rho = pearson(ranks(numeric_values[first]), ranks(numeric_values[second]))
            if abs(rho) > 0.7:
                spearman_pairs.append((first, second, rho))
    print("Spearman matrix:")
    print("feature," + ",".join(NUMERIC))
    for first in NUMERIC:
        print(first + "," + ",".join(f"{pearson(ranks(numeric_values[first]), ranks(numeric_values[second])):.6g}"
                                      for second in NUMERIC))
    print("Spearman pairs |rho|>0.7:", spearman_pairs)

    pulse_values = numeric_values["thalach"]
    thalach_edges = [statistics.quantiles(pulse_values, n=3, method="inclusive")[0],
                     statistics.quantiles(pulse_values, n=3, method="inclusive")[1]]
    print("Clinical bin edges: age=45,55,65; trestbps=120,140; chol=200,240; "
          f"thalach tertiles={thalach_edges[0]:.6g},{thalach_edges[1]:.6g}; oldpeak=0,2")
    transactions = [arm_items(row, thalach_edges) for row in rows]
    data_dir = ROOT / "data"
    write_transactions(data_dir / "transactions.csv", transactions)
    vocabulary = write_onehot(data_dir / "transactions_onehot.csv", transactions)
    average_items = sum(map(len, transactions)) / len(transactions)
    frequent_counts = {support: sum(count / len(transactions) >= support
                                    for count in Counter(item for items in transactions for item in items).values())
                       for support in (0.3, 0.2, 0.4)}
    print(f"Average items per transaction: {average_items:.6f}")
    print("Frequent 1-itemset counts:", frequent_counts)

    model_fields = []
    model_field_feature = {}
    model_rows = [[] for _ in rows]
    for column in NUMERIC:
        values = numeric_values[column]
        center = mean(values)
        scale = math.sqrt(variance(values))
        feature = f"X_{column}"
        model_fields.append(feature)
        model_field_feature[feature] = column
        for index, value in enumerate(values):
            model_rows[index].append((value - center) / scale if scale else 0.0)
    for column in CATEGORICAL:
        levels = sorted(set(categorical_values[column]), key=lambda value: (float(value), value))
        for level in levels:
            feature_name = f"X_{column}__{level}"
            model_fields.append(feature_name)
            model_field_feature[feature_name] = column
            for index, value in enumerate(categorical_values[column]):
                model_rows[index].append(int(value == level))
    model_path = data_dir / "model_matrix.csv"
    with model_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(model_fields + ["y"])
        writer.writerows([*values, labels[index]] for index, values in enumerate(model_rows))

    selected_features = [feature for feature in NUMERIC + CATEGORICAL if tiers[feature] in ("strong", "moderate")]
    selected_transactions = [[item for item in items
                              if item.split("=", 1)[0] in selected_features or item.startswith("disease=")]
                             for items in transactions]
    write_transactions(data_dir / "transactions_sel.csv", selected_transactions)
    selected_vocabulary = write_onehot(data_dir / "transactions_sel_onehot.csv", selected_transactions)
    selected_model_fields = [field for field in model_fields if model_field_feature[field] in selected_features]
    selected_model_indices = [model_fields.index(field) for field in selected_model_fields]
    with (data_dir / "model_matrix_sel.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(selected_model_fields + ["y"])
        writer.writerows([[values[index] for index in selected_model_indices] + [labels[row_index]]
                          for row_index, values in enumerate(model_rows)])
    selected_item_counts = Counter(item for items in selected_transactions for item in items)
    selected_frequent = {support: sum(count / len(selected_transactions) >= support
                                       for count in selected_item_counts.values())
                         for support in (0.2, 0.3, 0.4)}
    selected_lengths = [len(items) for items in selected_transactions]
    print("Selected features:", selected_features)
    print("Selected transaction items per row:",
          f"average={sum(selected_lengths) / len(selected_lengths):.6f}; min={min(selected_lengths)}; max={max(selected_lengths)}")
    print("Selected frequent 1-itemset counts:", selected_frequent)
    full_model_columns = model_fields + ["y"]
    onehot_columns = vocabulary
    print("model_matrix.csv df.columns:", full_model_columns)
    print("transactions_onehot.csv df.columns:", onehot_columns)
    model_x = [column for column in full_model_columns if column != "y"]
    forbidden = {"num", "num_bin", "source", "index", "transaction_id"}
    leaks = [column for column in model_x if column in forbidden]
    print("model_matrix X leakage check:", "PASS" if not leaks else f"FAIL {leaks}", "(y=num_bin)")
    item_leaks = [column for column in onehot_columns if column in forbidden]
    print("transactions_onehot X field check:", "PASS" if not item_leaks else f"FAIL {item_leaks}",
          "(disease=yes/no retained as ARM target items)")

    log_path = ROOT / "out" / "log.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Preprocessing log\n"
    marker = "\n## Phase 2 selection and transformation\n"
    log = existing.split(marker, 1)[0].rstrip()
    decisions = [
        "", "## Phase 2 selection and transformation", "",
        "action | affected | reason | alternative",
        f"Feature tests and tiers | {len(ranking)} features tested; selected tiers: {', '.join(selected_features)} | Use one point-biserial test for each numeric feature and chi-square/Cramer's V for categorical features; MI uses quantile-discretized numeric values; p<0.05 alone can flag small effects as significant at n={n_used}, so selected set uses effect tiers | Select by p-value alone.",
        "Effect tiers | strong >=0.30; moderate >=0.15 and <0.30; weak <0.15 | Use effect magnitude to select features; thresholds assumed by student | Use p<0.05 alone.",
        f"Observed-only sensitivity | Recomputed effects for {', '.join(SENSITIVITY)} using rows whose matching imputation flag is False | Show association on observed values only | Use imputed rows for all estimates.",
        f"Spearman screening | {len(spearman_pairs)} pairs exceed |rho|>0.7 | Flag strong numeric associations for review | Remove correlated feature now; deferred.",
        f"ARM discretization | {len(transactions)} transactions; {len(vocabulary)} item levels | Use age<45/45-54/55-64/65+; trestbps<120/120-139/>=140; chol<200/200-239/>=240; pooled thalach tertiles at {thalach_edges[0]:.6g},{thalach_edges[1]:.6g}; oldpeak 0/(0,2)/>=2; encode below-zero oldpeak separately to preserve observed negatives; omit slope=-1 | Drop unbinned negatives or include missing slope as an item.",
        f"ARM one-hot | {len(transactions)} rows × {len(vocabulary)} item columns; no transaction index | Make binary item incidence available for mining; disease tokens remain ARM outcome items | Keep only comma-joined item list or add an index.",
        f"Selected matrices | {len(rows)} rows; {len(selected_model_fields)} model X columns; {len(selected_vocabulary)} ARM items | Keep strong/moderate features only while retaining y and disease ARM items | Drop weak features from full matrices.",
        f"Distance matrix | {len(rows)} rows × {len(model_fields) + 1} columns | Z-score numeric features; one-hot categorical features including slope=-1; exclude num, num_bin, source from X and place num_bin in y | Include source or drop missing slope level.",
    ]
    log_path.write_text(log + "\n" + "\n".join(decisions) + "\n", encoding="utf-8")
    print("Saved shapes:")
    for name, header, count in (("transactions.csv", ["items"], len(transactions)),
                                ("transactions_onehot.csv", vocabulary, len(transactions)),
                                ("model_matrix.csv", model_fields + ["y"], len(rows)),
                                ("transactions_sel.csv", ["items"], len(selected_transactions)),
                                ("transactions_sel_onehot.csv", selected_vocabulary, len(selected_transactions)),
                                ("model_matrix_sel.csv", selected_model_fields + ["y"], len(rows))):
        print(f"{name}: ({count}, {len(header)})")


if __name__ == "__main__":
    run()
