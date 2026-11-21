#!/usr/bin/env python3
"""Create requested descriptive heart-disease figures and raw-number summaries."""

import csv
import importlib.util
import math
import random
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out" / "figs"
SEED = 42
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
PAIR_FEATURES = ["age", "thalach", "oldpeak", "trestbps"]
BLUE = (44, 125, 170)
RED = (218, 93, 74)
PALETTE = [(44, 125, 170), (218, 93, 74), (70, 157, 103), (172, 111, 178), (220, 165, 60)]
plt.rcParams["xaxis.labellocation"] = "center"
plt.rcParams["yaxis.labellocation"] = "center"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 100, "savefig.dpi": 300})
WHITE = (255, 255, 255)
INK = (28, 39, 51)
GRID = (218, 224, 230)


def load_rows(path):
    """Read CSV records; path identifies a requested data artifact."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row, column):
    """Convert one clean field to float; row/column identify an imputed numeric value."""
    return float(row[column])


def qtile(values, fraction):
    """Return interpolated quantile; values is numeric sequence and fraction is in [0,1]."""
    ordered = sorted(values)
    place = (len(ordered) - 1) * fraction
    low = int(place)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (place - low)


def average(values):
    """Return mean; values is a non-empty numeric sequence."""
    return sum(values) / len(values)


def pearson(xs, ys):
    """Compute Pearson correlation; xs and ys are equal-length numeric vectors."""
    mx, my = average(xs), average(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator else 0.0


def ranks(values):
    """Assign average ranks to ties; values is a numeric sequence for Spearman rho."""
    result = [0.0] * len(values)
    pairs = sorted(enumerate(values), key=lambda pair: pair[1])
    start = 0
    while start < len(pairs):
        stop = start + 1
        while stop < len(pairs) and pairs[stop][1] == pairs[start][1]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index, _ in pairs[start:stop]:
            result[index] = rank
        start = stop
    return result


def spearman(xs, ys):
    """Compute Spearman rank correlation; xs and ys are numeric sequences."""
    return pearson(ranks(xs), ranks(ys))


def render_engine():
    """Load existing PNG raster primitives; the cleaner module owns dependency-free encoding."""
    spec = importlib.util.spec_from_file_location("heart_cleaner", ROOT / "src" / "01_clean.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FONT.update({
        "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
        "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
        "<": ("00010", "00100", "01000", "10000", "01000", "00100", "00010"),
        ">": ("01000", "00100", "00010", "00001", "00010", "00100", "01000"),
        "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
        ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
        ",": ("00000", "00000", "00000", "00000", "00110", "00100", "01000"),
        "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
        "'": ("00100", "00100", "00000", "00000", "00000", "00000", "00000"),
        "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
        "*": ("00000", "10101", "01110", "11111", "01110", "10101", "00000"),
    })
    return module.Raster


Raster = render_engine()


def chart(title, xlabel, ylabel, width=1500, height=950):
    """Create titled white canvas and labelled axes; title/labels and dimensions define figure."""
    image = Raster(width, height)
    image.text(max(20, (width - len(title.upper()) * 12) // 2), 25, title.upper(), color=INK, scale=2)
    image.text(max(20, (width - len(xlabel.upper()) * 12) // 2), height - 68, xlabel.upper(), color=INK, scale=2)
    image.text(20, 105, ylabel.upper(), color=INK, scale=2)
    return image


def save(image, name):
    """Save one 300-DPI PNG; image is raster and name is plot filename."""
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name)


def axes(image, box, x_ticks=None, y_ticks=None):
    """Draw plot frame/grid and optional ticks; box is x0,y0,x1,y1 pixel bounds."""
    x0, y0, x1, y1 = box
    for value, label in (y_ticks or []):
        y = int(y1 - value * (y1 - y0))
        image.line(x0, y, x1, y, GRID)
        image.text(max(24, x0 - 58), y - 5, str(label), color=INK, scale=1)
    image.line(x0, y0, x0, y1, INK)
    image.line(x0, y1, x1, y1, INK)
    for fraction, label in (x_ticks or []):
        x = int(x0 + fraction * (x1 - x0))
        image.line(x, y1, x, y1 + 6, INK)
        image.text(x - len(str(label)) * 3, y1 + 12, str(label), color=INK, scale=1)


def dot(image, x, y, color, radius=5):
    """Draw filled circular mark; x/y is center, color RGB, radius in pixels."""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                image.pixel(x + dx, y + dy, color)


def scale(value, low, high, start, end):
    """Map data value to pixel span; low/high are domain and start/end are pixel bounds."""
    return start + (value - low) / (high - low) * (end - start) if high != low else (start + end) / 2


def legend(image, entries, x, y):
    """Draw compact color legend; entries are (label, RGB color) pairs at pixel origin."""
    for index, (label, color) in enumerate(entries):
        yy = y + index * 30
        image.rect(x, yy, x + 20, yy + 18, color)
        image.text(x + 28, yy + 2, label.upper(), color=INK, scale=1)


def draw_pie(rows, stats):
    """Render num class pie and record class counts; rows contain clean targets, stats receives raw counts."""
    counts = Counter(int(row["num"]) for row in rows)
    image = chart("num (0-4) distribution", "class label", "patient count", 1400, 900)
    cx, cy, radius = 470, 480, 280
    total = len(rows)
    cumulative = 0
    starts = []
    for klass in range(5):
        starts.append((cumulative / total * 2 * math.pi, (cumulative + counts[klass]) / total * 2 * math.pi))
        cumulative += counts[klass]
    for y in range(cy - radius, cy + radius):
        for x in range(cx - radius, cx + radius):
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy > radius * radius:
                continue
            angle = math.atan2(dy, dx) % (2 * math.pi)
            for index, (begin, end) in enumerate(starts):
                if begin <= angle < end:
                    image.pixel(x, y, PALETTE[index])
                    break
    legend(image, [(f"{k}: {counts[k]}", PALETTE[k]) for k in range(5)], 850, 250)
    stats.append(("p01", "num_counts", ", ".join(str(counts[k]) for k in range(5))))
    save(image, "p01_num_pie.png")


def draw_histogram(rows, stats):
    """Render overlaid age histograms by num_bin; rows are clean records and stats collects values."""
    image = chart("age by num_bin overlay", "age (years)", "patient count")
    box = (130, 170, 1270, 780)
    age_bins = list(range(25, 81, 5))
    counts = {label: [0] * (len(age_bins) - 1) for label in (0, 1)}
    for row in rows:
        age, label = int(float(row["age"])), int(row["num_bin"])
        index = min(len(counts[label]) - 1, max(0, (age - age_bins[0]) // 5))
        counts[label][index] += 1
    maximum = max(max(values) for values in counts.values())
    axes(image, box, [(i / (len(age_bins) - 1), age_bins[i]) for i in range(0, len(age_bins), 2)],
         [(fraction, round(maximum * fraction)) for fraction in (0, .25, .5, .75, 1)])
    lane = (box[2] - box[0]) / (len(age_bins) - 1)
    for index in range(len(age_bins) - 1):
        x0 = int(box[0] + index * lane + 4)
        y0 = int(scale(counts[0][index], 0, maximum, box[3], box[1]))
        y1 = int(scale(counts[1][index], 0, maximum, box[3], box[1]))
        image.rect(x0, y0, int(box[0] + (index + 1) * lane - 4), box[3], BLUE)
        overlay_width = max(4, int(lane * .58))
        image.rect(x0 + (int(lane) - overlay_width) // 2, y1,
                   x0 + (int(lane) + overlay_width) // 2, box[3], RED)
    legend(image, [("num_bin=0", BLUE), ("num_bin=1", RED)], 1300, 240)
    medians = [statistics.median([float(row["age"]) for row in rows if int(row["num_bin"]) == value]) for value in (0, 1)]
    class_counts = [sum(int(row["num_bin"]) == value for row in rows) for value in (0, 1)]
    stats.append(("p02", "age_medians_by_num_bin", f"{medians[0]:g}, {medians[1]:g}; n={class_counts[0]}, {class_counts[1]}"))
    save(image, "p02_age_histogram.png")


def draw_dotplot(rows, stats):
    """Render half-unit oldpeak stacks with one mark per ten patients; rows/stats supply data and summary."""
    counts = Counter(math.floor(float(row["oldpeak"]) * 2 + .5) / 2 for row in rows)
    values = sorted(counts)
    maximum_dots = max(counts[value] // 10 for value in values)
    image = chart("oldpeak dot plot; one dot = 10 patients", "oldpeak rounded to 0.5", "dots (10 patients each)")
    box = (140, 170, 1350, 780)
    axes(image, box, [(i / max(1, len(values) - 1), f"{values[i]:g}") for i in range(0, len(values), max(1, len(values) // 12))],
         [(fraction, round(maximum_dots * fraction)) for fraction in (0, .25, .5, .75, 1)])
    for index, value in enumerate(values):
        count_dots = counts[value] // 10
        x = scale(index, 0, max(1, len(values) - 1), box[0], box[2])
        for level in range(count_dots):
            y = scale(level + 1, 0, maximum_dots + 1, box[3], box[1])
            dot(image, int(x), int(y), BLUE, 5)
    common = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:4]
    stats.append(("p03", "oldpeak_rounded_0.5_top4_counts", ", ".join(f"{value:g}:{count}" for value, count in common)))
    save(image, "p03_oldpeak_dotplot.png")


def draw_stem_leaf(rows, stats):
    """Render seed-42 sample age stem-and-leaf text; rows supply patients and stats records sample values."""
    sample = random.Random(SEED).sample(rows, 80)
    stems = defaultdict(list)
    ages = [int(float(row["age"])) for row in sample]
    for age in ages:
        stems[age // 10].append(age % 10)
    image = Raster(1250, 850)
    image.text(75, 35, "AGE STEM AND LEAF - RANDOM SAMPLE OF 80", color=INK, scale=2)
    image.text(80, 110, "STEM | LEAVES", color=INK, scale=2)
    for index, stem in enumerate(range(min(stems), max(stems) + 1)):
        leaves = " ".join(str(value) for value in sorted(stems[stem]))
        image.text(110, 175 + index * 82, f"{stem} | {leaves}", color=INK, scale=2)
    stats.append(("p04", "sample80_age_min_median_max", f"{min(ages)}, {statistics.median(ages):g}, {max(ages)}"))
    save(image, "p04_age_stem_leaf.png")


def draw_scatter(rows, stats):
    """Render age/thalach scatter by num_bin; rows are clean patient values and stats records raw correlations."""
    image = chart("age vs thalach by num_bin", "age (years)", "maximum heart rate")
    box = (140, 170, 1280, 780)
    xvals = [float(row["age"]) for row in rows]
    yvals = [float(row["thalach"]) for row in rows]
    axes(image, box, [(i / 5, 25 + i * 11) for i in range(6)], [(i / 5, 60 + i * 32) for i in range(6)])
    for row in rows:
        x = scale(float(row["age"]), 25, 80, box[0], box[2])
        y = scale(float(row["thalach"]), 60, 220, box[3], box[1])
        dot(image, int(x), int(y), RED if int(row["num_bin"]) else BLUE, 3)
    legend(image, [("num_bin=0", BLUE), ("num_bin=1", RED)], 1300, 250)
    values = {}
    for label in (0, 1):
        subset = [row for row in rows if int(row["num_bin"]) == label]
        values[label] = spearman([float(row["age"]) for row in subset], [float(row["thalach"]) for row in subset])
    counts = [sum(int(row["num_bin"]) == label for row in rows) for label in (0, 1)]
    stats.append(("p05", "age_thalach_spearman_by_num_bin", f"{values[0]:.4f}, {values[1]:.4f}; n={counts[0]}, {counts[1]}"))
    save(image, "p05_age_thalach_scatter.png")


def draw_boxplot(image, values, center, box, low, high, color):
    """Draw one Tukey boxplot; values define sample and center/box/limits/color define placement."""
    q1, med, q3 = qtile(values, .25), qtile(values, .5), qtile(values, .75)
    spread = q3 - q1
    whisker_low = min(value for value in values if value >= q1 - 1.5 * spread)
    whisker_high = max(value for value in values if value <= q3 + 1.5 * spread)
    x0, y0, x1, y1 = box
    y = lambda value: int(scale(value, low, high, y1, y0))
    image.line(center, y(whisker_low), center, y(whisker_high), color)
    image.line(center - 18, y(whisker_low), center + 18, y(whisker_low), color)
    image.line(center - 18, y(whisker_high), center + 18, y(whisker_high), color)
    image.rect(center - 28, y(q3), center + 28, y(q1), color)
    image.line(center - 28, y(med), center + 28, y(med), INK)


def draw_boxes(rows, stats):
    """Render 3 numeric Tukey panels by num_bin; rows are clean records and stats stores group medians."""
    columns = ["thalach", "oldpeak", "age"]
    image = chart("thalach, oldpeak, age by num_bin", "num_bin group", "feature value", 1650, 1000)
    medians = {}
    for index, column in enumerate(columns):
        left, right = 100 + index * 510, 570 + index * 510
        top, bottom = 160, 780
        box = (left, top, right, bottom)
        vals = [float(row[column]) for row in rows]
        low, high = min(vals), max(vals)
        pad = max(1, (high - low) * .05)
        low, high = low - pad, high + pad
        image.text(left + 100, 95, column.upper(), color=INK, scale=2)
        axes(image, box, [], [])
        for label, color, center in ((0, BLUE, left + 140), (1, RED, left + 330)):
            group = [float(row[column]) for row in rows if int(row["num_bin"]) == label]
            draw_boxplot(image, group, center, box, low, high, color)
            image.text(center - 24, bottom + 18, f"BIN {label}", color=INK, scale=1)
            medians[(column, label)] = statistics.median(group)
    stats.append(("p06", "thalach_median_bin0,thalach_median_bin1,oldpeak_median_bin0,oldpeak_median_bin1,age_overall_median",
                  f"{medians[(columns[0], 0)]:g}, {medians[(columns[0], 1)]:g}, "
                  f"{medians[(columns[1], 0)]:g}, {medians[(columns[1], 1)]:g}, "
                  f"{statistics.median(float(row['age']) for row in rows):g}"))
    save(image, "p06_boxplots_by_num_bin.png")


def kernel_density(values, points, bandwidth):
    """Compute Gaussian KDE values; sample values and query points share numeric units."""
    normalizer = len(values) * bandwidth * math.sqrt(2 * math.pi)
    return [sum(math.exp(-.5 * ((point - value) / bandwidth) ** 2) for value in values) / normalizer
            for point in points]


def draw_violin(rows, stats):
    """Render KDE violins of thalach by cp; rows supply data and stats records group medians."""
    image = chart("thalach violin by cp", "", "maximum heart rate")
    box = (150, 170, 1320, 780)
    category_labels = ["1 typical", "2 atypical", "3 non-anginal", "4 asymptomatic"]
    panel_width = (box[2] - box[0]) / 4
    x_ticks = [((i + 0.5) / 4, category_labels[i]) for i in range(4)]
    axes(image, box, x_ticks, [(i / 5, 60 + i * 32) for i in range(6)])
    points = [float(v) for v in range(60, 221, 2)]
    group_medians = []
    for cp in range(1, 5):
        vals = [float(row["thalach"]) for row in rows if int(row["cp"]) == cp]
        bandwidth = max(2.0, 1.06 * statistics.pstdev(vals) * len(vals) ** (-.2))
        density = kernel_density(vals, points, bandwidth)
        max_density = max(density)
        center = int(box[0] + (cp - 1 + 0.5) * panel_width)
        for index, point in enumerate(points):
            half_width = max(1, int(65 * density[index] / max_density))
            yy = int(scale(point, 60, 220, box[3], box[1]))
            image.line(center - half_width, yy, center + half_width, yy, PALETTE[(cp - 1) % len(PALETTE)])
        group_medians.append(statistics.median(vals))
    stats.append(("p07", "thalach_median_by_cp_1_to_4", ", ".join(f"{value:g}" for value in group_medians)))
    xlabel = "CHEST PAIN CODE"
    box_center = (box[0] + box[2]) // 2
    image.text(box_center - len(xlabel) * 6, image.height - 68, xlabel, color=INK, scale=2)
    save(image, "p07_thalach_violin_by_cp.png")


def draw_heatmap(rows, stats, observed_n):
    """Render numeric Spearman matrix on observed-chol rows; rows/stats/n define matrix and summary."""
    columns = NUMERIC + ["num_bin"]
    data = {column: [int(row[column]) if column == "num_bin" else float(row[column]) for row in rows]
            for column in columns}
    matrix = [[spearman(data[a], data[b]) for b in columns] for a in columns]
    image = chart(f"Spearman heatmap - observed rows only, n={observed_n}", "", "numeric feature", 2100, 1800)
    left, top, cell = 500, 350, 220
    for index, column in enumerate(columns):
        image.text(left + index * cell + 18, top - 45, column.upper(), color=INK, scale=2)
        image.text(80, top + index * cell + 48, column.upper(), color=INK, scale=2)
        for j, rho in enumerate(matrix[index]):
            x0, y0 = left + j * cell, top + index * cell
            if rho < 0:
                color = (int(245 + 20 * rho), int(245 + 115 * rho), 245)
            else:
                color = (245, int(245 - 115 * rho), int(245 - 20 * rho))
            image.rect(x0, y0, x0 + cell - 3, y0 + cell - 3, color)
            image.text(x0 + 25, y0 + 52, f"{rho:.2f}", color=INK, scale=2)
    pairs = [("age", "num_bin"), ("trestbps", "num_bin"), ("chol", "num_bin"), ("thalach", "num_bin"), ("oldpeak", "num_bin")]
    stats.append(("p08", "spearman_with_num_bin", ", ".join(f"{a}:{spearman(data[a], data[b]):.4f}" for a, b in pairs)))
    xlabel = "NUMERIC FEATURE"
    image.text(image.width // 2 - len(xlabel) * 6, image.height - 68, xlabel, color=INK, scale=2)
    save(image, "p08_spearman_heatmap.png")


def draw_rate_panel(image, rows, column, levels, left, title):
    """Draw 100-percent no/yes bars for a grouping field; rows, levels, left and title select panel."""
    image.text(left + 80, 145, title.upper(), color=INK, scale=2)
    x0, y0, x1, y1 = left + 50, 220, left + 680, 770
    axes(image, (x0, y0, x1, y1), [(i / max(1, len(levels) - 1), str(levels[i])) for i in range(len(levels))],
         [(i / 4, f"{i * 25}%") for i in range(5)])
    rates = []
    width = 80
    for index, level in enumerate(levels):
        group = [row for row in rows if str(row[column]) == str(level)]
        positive = sum(int(row["num_bin"]) for row in group)
        rate = positive / len(group) if group else 0
        rates.append(rate)
        cx = int(scale(index, 0, max(1, len(levels) - 1), x0 + 90, x1 - 90))
        y_split = int(scale(rate, 0, 1, y1, y0))
        image.rect(cx - width // 2, y_split, cx + width // 2, y1, RED)
        image.rect(cx - width // 2, y0, cx + width // 2, y_split, BLUE)
    return rates


def draw_stacked_bars(rows, stats):
    """Render disease-rate bars by cp and sex; rows supply groups and stats records cp rates."""
    image = chart("100% stacked disease rate by cp and sex", "group level", "percent", 1700, 950)
    cp_rates = draw_rate_panel(image, rows, "cp", [1, 2, 3, 4], 70, "by cp")
    draw_rate_panel(image, rows, "sex", ["0", "1"], 800, "by sex")
    legend(image, [("num_bin=0", BLUE), ("num_bin=1", RED)], 1280, 820)
    stats.append(("p09", "disease_rate_by_cp_1_to_4", ", ".join(f"{rate:.4f}" for rate in cp_rates)))
    save(image, "p09_disease_rate_cp_sex.png")


def draw_source_counts(rows, stats):
    """Render num_bin counts by source; rows provide labels and stats stores source-positive counts."""
    sources = sorted({row["source"] for row in rows})
    image = chart("num_bin count by source", "source", "patient count")
    box = (180, 180, 1270, 770)
    counts = {source: [sum(row["source"] == source and int(row["num_bin"]) == label for row in rows)
                       for label in (0, 1)] for source in sources}
    maximum = max(max(value) for value in counts.values())
    axes(image, box, [],
         [(i / 5, round(maximum * i / 5)) for i in range(6)])
    for index, source in enumerate(sources):
        center = scale(index, 0, len(sources) - 1, box[0] + 110, box[2] - 110)
        for label, color, dx in ((0, BLUE, -30), (1, RED, 30)):
            y = scale(counts[source][label], 0, maximum, box[3], box[1])
            image.rect(int(center + dx - 24), int(y), int(center + dx + 24), box[3], color)
        source_label = source.replace("processed.", "").replace(".data", "").upper()
        image.text(int(center - len(source_label) * 3), box[3] + 12, source_label, color=INK, scale=1)
    legend(image, [("num_bin=0", BLUE), ("num_bin=1", RED)], 1300, 260)
    stats.append(("p10", "num_bin_1_counts_by_source",
                  ", ".join(f"{source.replace('processed.', '').replace('.data', '').upper()}={counts[source][1]}"
                            for source in sources)))
    save(image, "p10_num_bin_by_source.png")


def draw_pairplot(rows, stats):
    """Render four-feature pair matrix with class hue; rows supply data and stats stores raw correlations."""
    width, height, left, top, cell = 1800, 1800, 170, 130, 390
    image = Raster(width, height)
    image.text(140, 25, "PAIR PLOT: AGE THALACH OLDPEAK TRESTBPS BY NUM_BIN", color=INK, scale=2)
    domains = {column: [float(row[column]) for row in rows] for column in PAIR_FEATURES}
    pair_stats = []
    for row_index, ycol in enumerate(PAIR_FEATURES):
        for col_index, xcol in enumerate(PAIR_FEATURES):
            x0, y0 = left + col_index * cell, top + row_index * cell
            box = (x0 + 25, y0 + 20, x0 + cell - 15, y0 + cell - 50)
            image.line(box[0], box[1], box[0], box[3], INK)
            image.line(box[0], box[3], box[2], box[3], INK)
            if row_index == col_index:
                for label, color in ((0, BLUE), (1, RED)):
                    vals = [float(row[xcol]) for row in rows if int(row["num_bin"]) == label]
                    edges = [min(domains[xcol]) + i * (max(domains[xcol]) - min(domains[xcol])) / 8 for i in range(9)]
                    hist = [0] * 8
                    for value in vals:
                        hist[min(7, max(0, int((value - edges[0]) / (edges[-1] - edges[0]) * 8)))] += 1
                    max_hist = max(hist)
                    lane = (box[2] - box[0]) / 8
                    for i, count in enumerate(hist):
                        hh = (box[3] - box[1]) * count / max_hist if max_hist else 0
                        inset = lane * .2 if label else 0
                        image.rect(int(box[0] + i * lane + inset), int(box[3] - hh),
                                   int(box[0] + (i + 1) * lane - 2 - inset), box[3], color)
            else:
                xvalues, yvalues = domains[xcol], domains[ycol]
                for row in rows:
                    x = scale(float(row[xcol]), min(xvalues), max(xvalues), box[0], box[2])
                    y = scale(float(row[ycol]), min(yvalues), max(yvalues), box[3], box[1])
                    dot(image, int(x), int(y), RED if int(row["num_bin"]) else BLUE, 2)
            if col_index == 0:
                image.text(10, y0 + cell // 2, ycol.upper(), color=INK, scale=1)
            if row_index == len(PAIR_FEATURES) - 1:
                image.text(x0 + 65, y0 + cell - 28, xcol.upper(), color=INK, scale=1)
    for first, second in (("age", "thalach"), ("age", "oldpeak"), ("age", "trestbps"),
                          ("thalach", "oldpeak"), ("thalach", "trestbps")):
        pair_stats.append(f"{first}-{second}:{spearman(domains[first], domains[second]):.4f}")
    stats.append(("p11", "spearman_pair_values", ", ".join(pair_stats)))
    save(image, "p11_pair_plot.png")


def draw_kde(rows, stats):
    """Render Gaussian KDE panels for age and thalach by num_bin; rows/stats hold values and summaries."""
    image = chart("KDE age and thalach by num_bin", "value", "density", 1600, 1000)
    medians = {}
    for panel, column in enumerate(("thalach", "age")):
        x0, x1, y0, y1 = 120 + panel * 750, 760 + panel * 750, 190, 780
        image.text(x0 + 190, 115, column.upper(), color=INK, scale=2)
        axis_box = (x0, y0, x1, y1)
        axes(image, axis_box, [], [(i / 4, f"{i/4:.2f}") for i in range(5)])
        combined = [float(row[column]) for row in rows]
        low, high = min(combined), max(combined)
        points = [low + (high - low) * index / 100 for index in range(101)]
        maximum = 0
        curves = {}
        for label in (0, 1):
            vals = [float(row[column]) for row in rows if int(row["num_bin"]) == label]
            bandwidth = max(.25, 1.06 * statistics.pstdev(vals) * len(vals) ** (-.2))
            curves[label] = kernel_density(vals, points, bandwidth)
            maximum = max(maximum, max(curves[label]))
            medians[(column, label)] = statistics.median(vals)
        for label, color in ((0, BLUE), (1, RED)):
            coords = [(int(scale(point, low, high, x0, x1)),
                       int(scale(density, 0, maximum, y1, y0))) for point, density in zip(points, curves[label])]
            for a, b in zip(coords, coords[1:]):
                image.line(a[0], a[1], b[0], b[1], color)
    legend(image, [("num_bin=0", BLUE), ("num_bin=1", RED)], 1340, 820)
    stats.append(("p12", "medians_thalach_age_by_num_bin",
                  ", ".join(f"{column}:{medians[(column, label)]:g}" for column in ("thalach", "age") for label in (0, 1))))
    save(image, "p12_kde_age_thalach.png")


def draw_parallel(rows, flags, stats):
    """Render 200 sampled min-max parallel profiles; rows are observed-chol cases, flags select completeness."""
    sample = random.Random(SEED).sample(rows, min(200, len(rows)))
    image = chart(f"parallel coordinates - observed rows only, n={len(rows)}; sample={len(sample)}",
                  "", "min-max scaled value", 1700, 1000)
    box = (150, 190, 1510, 780)
    axes(image, box, [(i / (len(NUMERIC) - 1), NUMERIC[i]) for i in range(len(NUMERIC))],
         [(i / 5, f"{i/5:.1f}") for i in range(6)])
    domains = {column: [float(row[column]) for row in rows] for column in NUMERIC}
    scaled = []
    for row in sample:
        points = []
        for index, column in enumerate(NUMERIC):
            values = domains[column]
            x = scale(index, 0, len(NUMERIC) - 1, box[0], box[2])
            y = scale(float(row[column]), min(values), max(values), box[3], box[1])
            points.append((int(x), int(y)))
        color = RED if int(row["num_bin"]) else BLUE
        scaled.append([scale(float(row[column]), min(domains[column]), max(domains[column]), 0, 1) for column in NUMERIC])
        for first, second in zip(points, points[1:]):
            image.line(first[0], first[1], second[0], second[1], color)
    legend(image, [("num_bin=0", BLUE), ("num_bin=1", RED)], 1300, 820)
    labels = [int(row["num_bin"]) for row in sample]
    stats.append(("p13", "sample200_class_counts_scaled_min_max", f"{len(sample)}, {labels.count(0)}, {labels.count(1)}, 0, 1"))
    xlabel = "NUMERIC FEATURE"
    image.text(image.width // 2 - len(xlabel) * 6, image.height - 68, xlabel, color=INK, scale=2)
    save(image, "p13_parallel_coordinates.png")


def write_stats(stats):
    """Write only plot labels and raw numeric summaries; stats contains precomputed number strings."""
    path = ROOT / "out" / "tables" / "plot_stats.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Plot statistics", ""]
    for plot_id, label, values in stats:
        lines.extend([f"## {plot_id}", f"- {label}: {values}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_style_axis(ax, numeric_x=True, numeric_y=True):
    """Apply readable axis ticks and font to one Matplotlib axis; flags identify continuous dimensions."""
    ax.tick_params(axis="both", labelsize=9)
    if numeric_x:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    if numeric_y:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))


def save_matplotlib(fig, filename):
    """Save one tight 300-DPI figure under out/figs; fig and filename identify output."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)


def label_panel(ax, label):
    """Add required (a)-(d) panel label in axes coordinates; ax and label select target panel."""
    ax.text(-0.08, 1.04, f"({label})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def make_stem_text(values):
    """Format age values as stem-and-leaf text; values is sampled integer age sequence."""
    stems = defaultdict(list)
    for value in values:
        age = int(value)
        stems[age // 10].append(age % 10)
    lines = ["Stem | Leaves"]
    for stem, leaves in sorted(stems.items()):
        ordered = sorted(leaves)
        for offset in range(0, len(ordered), 14):
            label = f"{stem:>2}" if offset == 0 else "  "
            lines.append(f"{label}   | {' '.join(map(str, ordered[offset:offset + 14]))}")
    return "\n".join(lines)


def build_matplotlib_figures():
    """Rebuild individual and composite plots with Matplotlib/Seaborn using clean and raw project inputs."""
    clean = pd.read_csv(DATA / "clean.csv")
    raw = pd.read_csv(DATA / "raw_combined.csv")
    flags = pd.read_csv(DATA / "imputed_flags.csv")
    d = clean.copy()
    d["disease_label"] = d["num_bin"].map({0: "No disease", 1: "Disease"})
    d["sex_label"] = d["sex"].map({0: "Female", 1: "Male"})
    cp_names = {1: "1 typical", 2: "2 atypical", 3: "3 non-anginal", 4: "4 asymptomatic"}
    d["cp_label"] = d["cp"].map(cp_names)
    source_names = {"processed.cleveland.data": "Cleveland", "processed.hungarian.data": "Hungarian",
                    "processed.switzerland.data": "Switzerland", "processed.va.data": "VA"}
    d["source_label"] = d["source"].map(source_names)
    palette = {"No disease": "#287DAA", "Disease": "#DA5D4A"}
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    short = {"age": "Age (years)", "trestbps": "Resting blood pressure (mmHg)", "chol": "Cholesterol (mg/dL)",
             "thalach": "Max heart rate (bpm)", "oldpeak": "ST depression", "num_bin": "Disease target"}
    stats = []
    observed_chol = d.loc[~flags["chol_imputed"].astype(bool)].copy()

    # p01 pie: multiclass target with count and percentage labels.
    counts = d["num"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [f"Level {level} (n={count})" for level, count in counts.items()]
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax.set_title("Disease severity distribution (levels 0-4)")
    save_matplotlib(fig, "p01_num_pie.png")
    stats.append(("p01", "num_counts", ", ".join(str(int(v)) for v in counts.values)))

    # p02 overlay histogram.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(data=d, x="age", hue="disease_label", hue_order=["No disease", "Disease"],
                 palette=palette, bins=18, element="step", stat="count", common_norm=False, ax=ax)
    ax.get_legend().set_title("Disease status")
    ax.set(xlabel=short["age"], ylabel="Patients"); ax.set_title("Age by disease target")
    plot_style_axis(ax); save_matplotlib(fig, "p02_age_histogram.png")
    stats.append(("p02", "age_medians_by_disease", str(d.groupby("disease_label").age.median().to_dict())))

    # p03 dot display: each plotted dot represents ten patients; remainder appears in annotation.
    oldpeak_rounded = (d["oldpeak"] * 2).round() / 2
    dot_counts = oldpeak_rounded.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs, ys = [], []
    for value, count in dot_counts.items():
        dots = int(count) // 10
        xs.extend([value] * dots); ys.extend(range(1, dots + 1))
        ax.annotate(f"n={int(count)}", (value, dots + .3), ha="center", fontsize=7)
    ax.scatter(xs, ys, s=20, color="#287DAA", alpha=.85)
    ax.set(xlabel="ST depression (rounded to 0.5)", ylabel="Dots (10 patients each)")
    ax.set_title("ST depression dot plot"); plot_style_axis(ax); save_matplotlib(fig, "p03_oldpeak_dotplot.png")
    stats.append(("p03", "oldpeak_counts_rounded_0.5", str({float(k): int(v) for k, v in dot_counts.items()})))

    # p04 stem-and-leaf image, fixed random sample of 80.
    sample_ages = d.sample(n=min(80, len(d)), random_state=SEED)["age"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.axis("off")
    ax.text(.04, .94, "Age stem-and-leaf, random sample n=80", transform=ax.transAxes, va="top", fontsize=11)
    ax.text(.10, .82, make_stem_text(sample_ages), transform=ax.transAxes, va="top", family="DejaVu Sans Mono", fontsize=9)
    save_matplotlib(fig, "p04_age_stem_leaf.png")
    stats.append(("p04", "sample80_age_min_median_max", f"{min(sample_ages)}, {statistics.median(sample_ages):g}, {max(sample_ages)}"))

    # p05 scatter.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.scatterplot(data=d, x="age", y="thalach", hue="disease_label", hue_order=["No disease", "Disease"],
                    palette=palette, alpha=.65, s=28, ax=ax)
    ax.get_legend().set_title("Disease status")
    ax.set(xlabel=short["age"], ylabel=short["thalach"]); ax.set_title("Age vs. maximum heart rate")
    plot_style_axis(ax); save_matplotlib(fig, "p05_age_thalach_scatter.png")

    # p06 three numeric boxplot panels.
    fig, axes_array = plt.subplots(1, 3, figsize=(9, 4.5))
    for ax, column in zip(axes_array, ["thalach", "oldpeak", "age"]):
        sns.boxplot(data=d, x="disease_label", y=column, hue="disease_label", hue_order=["No disease", "Disease"],
                    order=["No disease", "Disease"], palette=palette, legend=False, ax=ax)
        ax.set(xlabel="Disease status", ylabel=short.get(column, column)); plot_style_axis(ax, numeric_x=False)
    fig.suptitle("Numeric features by disease status"); save_matplotlib(fig, "p06_boxplots_by_num_bin.png")

    # p07 ordered categorical violin plot with ticks centered at category centers.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cp_order = [cp_names[level] for level in (1, 2, 3, 4)]
    sns.violinplot(data=d, x="cp_label", y="thalach", order=cp_order, inner="quartile", cut=0, color="#59A6C5", ax=ax)
    ax.set_xticks(range(4), cp_order); ax.set(xlabel="Chest-pain category", ylabel=short["thalach"])
    ax.set_title("Maximum heart rate by chest-pain category"); plot_style_axis(ax, numeric_x=False)
    save_matplotlib(fig, "p07_thalach_violin_by_cp.png")
    stats.append(("p07", "thalach_medians_by_cp", str(d.groupby("cp_label", sort=False).thalach.median().to_dict())))

    # p08 observed-chol Spearman heatmap with n in title.
    heat_cols = numeric + ["num_bin"]
    corr = observed_chol[heat_cols].corr(method="spearman")
    corr.index = [short.get(v, v) for v in corr.index]; corr.columns = [short.get(v, v) for v in corr.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True,
                cbar=True, annot_kws={"size": 9}, ax=ax)
    ax.set_title(f"Spearman correlation (observed rows only, n={len(observed_chol)})")
    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", labelrotation=38)
    for tick in ax.get_xticklabels(): tick.set_ha("right")
    save_matplotlib(fig, "p08_spearman_heatmap.png")
    stats.append(("p08", "rho_with_disease_observed_chol", str(corr["Disease target"].round(3).to_dict())))

    # p09 100% stacked bars for chest pain and sex.
    fig, axes_array = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, column, category_order, title in ((axes_array[0], "cp_label", [cp_names[i] for i in range(1,5)], "By chest-pain category"),
                                               (axes_array[1], "sex_label", ["Female", "Male"], "By sex")):
        table = pd.crosstab(d[column], d["disease_label"], normalize="index").reindex(category_order)
        table = table.reindex(columns=["No disease", "Disease"], fill_value=0)
        table.plot(kind="bar", stacked=True, color=[palette["No disease"], palette["Disease"]], ax=ax)
        ax.set(title=title, xlabel="", ylabel="Share of patients"); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
        ax.legend(title="Disease status", fontsize=7); ax.tick_params(axis="x", rotation=25, labelsize=8)
        plot_style_axis(ax, numeric_x=False)
    fig.suptitle("Disease distribution by group"); save_matplotlib(fig, "p09_disease_rate_cp_sex.png")

    # p10 count bar by source.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.countplot(data=d, x="source_label", hue="disease_label", hue_order=["No disease", "Disease"],
                  palette=palette, ax=ax)
    ax.get_legend().set_title("Disease status")
    ax.set(xlabel="Data source", ylabel="Patients"); ax.set_title("Disease target counts by source")
    ax.tick_params(axis="x", rotation=15); plot_style_axis(ax, numeric_x=False)
    save_matplotlib(fig, "p10_num_bin_by_source.png")

    # p11 pair plot with visible class legend.
    pair_data = d[PAIR_FEATURES + ["disease_label"]].rename(columns={"age":"Age (years)", "thalach":"Max heart rate (bpm)",
        "oldpeak":"ST depression", "trestbps":"Resting blood pressure (mmHg)"})
    pair = sns.pairplot(pair_data, hue="disease_label", hue_order=["No disease", "Disease"],
                        palette=palette, corner=True, diag_kind="kde", plot_kws={"s": 16, "alpha": .55}, height=1.65)
    pair.fig.suptitle("Pair plot by disease status", y=1.02)
    pair._legend.set_title("Disease status")
    for ax in pair.axes.flat:
        if ax is not None:
            ax.tick_params(labelsize=9)
    save_matplotlib(pair.fig, "p11_pair_plot.png")

    # p12 KDE panels for heart rate and age.
    fig, axes_array = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, column in zip(axes_array, ["thalach", "age"]):
        sns.kdeplot(data=d, x=column, hue="disease_label", hue_order=["No disease", "Disease"],
                    palette=palette, fill=False, common_norm=False, ax=ax)
        ax.get_legend().set_title("Disease status")
        ax.set(xlabel=short[column], ylabel="Density", title=short[column]); plot_style_axis(ax)
    fig.suptitle("KDE by disease status"); save_matplotlib(fig, "p12_kde_age_thalach.png")

    # p13 min-max parallel coordinates, stratified random sample.
    parallel_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    parallel_rows = observed_chol.sample(n=min(200, len(observed_chol)), random_state=SEED).copy()
    scaled = parallel_rows[parallel_cols].apply(lambda col: (col-col.min())/(col.max()-col.min() or 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in palette.items():
        subset = scaled.loc[parallel_rows.disease_label.eq(label)]
        for row in subset.to_numpy():
            ax.plot(range(len(parallel_cols)), row, color=color, alpha=.12, linewidth=.7)
    ax.set_xticks(range(len(parallel_cols)), [short[column] for column in parallel_cols], rotation=20, ha="right")
    ax.set(xlabel="Feature", ylabel="Min-max scaled value", title=f"Parallel coordinates, n={len(parallel_rows)}")
    ax.set_ylim(0, 1); ax.set_yticks(np.linspace(0, 1, 6)); ax.legend(handles=[plt.Line2D([0],[0],color=color,label=label) for label,color in palette.items()])
    plot_style_axis(ax, numeric_x=False); save_matplotlib(fig, "p13_parallel_coordinates.png")

    # Missingness by source and numeric before/after boxplots.
    fields = [column for column in raw.columns if column != "source"]
    missing = raw.groupby("source")[fields].apply(lambda group: group.isna().mean().mul(100)).reset_index().melt("source", var_name="feature", value_name="missing_percent")
    missing["source_label"] = missing.source.map(source_names)
    feature_names = {"age":"Age", "sex":"Sex", "cp":"Chest-pain category", "trestbps":"Resting blood pressure",
                     "chol":"Cholesterol", "fbs":"Fasting blood sugar", "restecg":"Resting ECG",
                     "thalach":"Max heart rate", "exang":"Exercise-induced angina", "oldpeak":"ST depression",
                     "slope":"ST slope", "ca":"Coronary vessel count", "thal":"Thal test", "num":"Disease severity"}
    missing["feature"] = missing.feature.map(feature_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=missing, x="feature", y="missing_percent", hue="source_label", ax=ax)
    ax.get_legend().set_title("Cohort")
    ax.set(xlabel="Feature", ylabel="Missing values (%)", title="Missingness by source")
    ax.tick_params(axis="x", rotation=45); plot_style_axis(ax, numeric_x=False); save_matplotlib(fig, "missingness_bar.png")

    before = raw[numeric].assign(stage="Before (raw)").melt("stage", var_name="feature", value_name="value")
    after = clean[numeric].assign(stage="After (imputed)").melt("stage", var_name="feature", value_name="value")
    both = pd.concat([before, after], ignore_index=True)
    fig, axes_array = plt.subplots(1, 4, figsize=(9, 4.5))
    for ax, column in zip(axes_array, numeric):
        sns.boxplot(data=both[both.feature.eq(column)], x="stage", y="value", hue="stage", legend=False, ax=ax,
                    palette={"Before (raw)":"#287DAA", "After (imputed)":"#DA5D4A"})
        ax.set(xlabel="", ylabel=short.get(column,column), title=short.get(column,column)); ax.tick_params(axis="x", rotation=20)
        plot_style_axis(ax, numeric_x=False)
    fig.suptitle("Numeric values before and after imputation (raw zeros retained)")
    save_matplotlib(fig, "boxplots_before_after.png")

    # Composite fig_A: pie, histogram, KDE, and dot plot.
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 7.3))
    ax=axs[0,0]; ax.pie(counts.values, labels=[f"Level {level}" for level in counts.index], autopct="%1.0f%%", startangle=90, textprops={"fontsize":6}); ax.set_title("Disease severity")
    ax=axs[0,1]; sns.histplot(data=d,x="age",hue="disease_label",hue_order=["No disease","Disease"],palette=palette,element="step",ax=ax); ax.set(xlabel="Age (years)",ylabel="Patients"); ax.get_legend().set_title("Disease status")
    ax=axs[1,0]
    for label,color in palette.items(): sns.kdeplot(data=d[d.disease_label.eq(label)],x="thalach",color=color,label=label,ax=ax)
    ax.set(xlabel=short["thalach"],ylabel="Density"); ax.legend(fontsize=7)
    ax=axs[1,1]; ax.scatter(xs,ys,s=8,color="#287DAA"); ax.set(xlabel="ST depression (0.5 units)",ylabel="Dots (10 patients each)")
    for ax,label in zip(axs.flat,"abcd"):
        label_panel(ax,label)
        if ax is not axs[0,0]: plot_style_axis(ax,numeric_x=True,numeric_y=True)
    fig.tight_layout(); save_matplotlib(fig,"fig_A_univariate.png")

    # Composite fig_B: box, violin, stem text, stacked group bars.
    fig, axs = plt.subplots(2,2,figsize=(6.5,7.3))
    sns.boxplot(data=d,x="disease_label",y="thalach",hue="disease_label",hue_order=["No disease","Disease"],order=["No disease","Disease"],palette=palette,legend=False,ax=axs[0,0]); axs[0,0].set(xlabel="Disease status",ylabel=short["thalach"])
    compact_cp = ["1 typical", "2 atyp.", "3 non-ang.", "4 asympt."]
    sns.violinplot(data=d,x="cp_label",y="thalach",order=cp_order,cut=0,color="#59A6C5",ax=axs[0,1]); axs[0,1].set(xlabel="Chest-pain category",ylabel=short["thalach"]); axs[0,1].set_xticks(range(4),compact_cp,rotation=15,ha="right",fontsize=7)
    axs[1,0].axis("off"); axs[1,0].text(.02,.98,make_stem_text(sample_ages),family="DejaVu Sans Mono",fontsize=8,va="top",transform=axs[1,0].transAxes); axs[1,0].set_title("Age stem-and-leaf, n=80")
    combined_order = cp_order + ["Female", "Male"]
    group_rows = pd.concat([d.assign(plot_group=d.cp_label), d.assign(plot_group=d.sex_label)], ignore_index=True)
    stacked = pd.crosstab(pd.Categorical(group_rows.plot_group,categories=combined_order),group_rows.disease_label,normalize="index").reindex(columns=["No disease","Disease"],fill_value=0)
    stacked.plot(kind="bar",stacked=True,color=[palette["No disease"],palette["Disease"]],ax=axs[1,1]); axs[1,1].set(xlabel="CP category / sex",ylabel="Patient share",title="CP and sex stacked"); axs[1,1].yaxis.set_major_formatter(mticker.PercentFormatter(1)); axs[1,1].set_ylim(0,1.17); axs[1,1].set_xticks(range(6),["typical","atyp.","non-ang.","asympt.","Female","Male"],rotation=25,ha="right",fontsize=7); axs[1,1].legend(title=None,loc="upper right",bbox_to_anchor=(1,.99),fontsize=6,ncol=2,frameon=False)
    for ax,label in zip(axs.flat,"abcd"): label_panel(ax,label); plot_style_axis(ax,numeric_x=False)
    fig.tight_layout(); save_matplotlib(fig,"fig_B_group_comparison.png")

    # Composite fig_C: scatter, pair excerpt, heatmap, parallel coordinates.
    fig, axs = plt.subplots(2,2,figsize=(6.5,7.3))
    sns.scatterplot(data=d,x="age",y="thalach",hue="disease_label",hue_order=["No disease","Disease"],palette=palette,s=10,alpha=.6,legend=False,ax=axs[0,0]); axs[0,0].set(xlabel="Age (years)",ylabel=short["thalach"])
    ax=axs[0,1]; sns.scatterplot(data=d.sample(min(250,len(d)),random_state=SEED),x="age",y="oldpeak",hue="disease_label",hue_order=["No disease","Disease"],palette=palette,s=8,alpha=.5,legend=False,ax=ax); ax.set(xlabel="Age (years)",ylabel="ST depression",title="Pair plot excerpt")
    compact_corr = observed_chol[heat_cols].corr(method="spearman")
    compact_labels = ["Age", "BP", "Chol", "HR", "ST", "Disease"]
    compact_corr.index = compact_labels; compact_corr.columns = compact_labels
    sns.heatmap(compact_corr,cmap="coolwarm",vmin=-1,vmax=1,annot=True,fmt=".2f",cbar=True,annot_kws={"size":7},ax=axs[1,0]); axs[1,0].set_title(f"Spearman, observed only, n={len(observed_chol)}"); axs[1,0].tick_params(labelsize=9,axis="y"); axs[1,0].tick_params(labelsize=8,axis="x",rotation=35)
    ax=axs[1,1]
    for label,color in palette.items():
        subset=scaled.loc[parallel_rows.disease_label.eq(label)]
        for row in subset.to_numpy(): ax.plot(range(len(parallel_cols)),row,color=color,alpha=.1,linewidth=.5)
    ax.set_xticks(range(len(parallel_cols)),["Age","BP","Chol","HR","Oldpeak"],rotation=30,ha="right"); ax.set(xlabel="Feature",ylabel="Scaled 0-1"); ax.set_ylim(0,1)
    for ax,label in zip(axs.flat,"abcd"): label_panel(ax,label); plot_style_axis(ax,numeric_x=ax in (axs[0,0],axs[0,1]),numeric_y=ax is not axs[1,0])
    fig.tight_layout(); save_matplotlib(fig,"fig_C_relations.png")

    # Composite fig_D: source counts and missingness panels.
    fig, axs = plt.subplots(2,2,figsize=(6.5,7.3))
    sns.countplot(data=d,x="source_label",hue="disease_label",hue_order=["No disease","Disease"],palette=palette,ax=axs[0,0]); axs[0,0].set(xlabel="Source",ylabel="Patients",title="Counts by source"); axs[0,0].tick_params(axis="x",rotation=25); axs[0,0].get_legend().set_title("Disease status")
    sns.barplot(data=missing,x="feature",y="missing_percent",hue="source_label",ax=axs[0,1],legend=False); axs[0,1].set(xlabel="Feature",ylabel="Missing (%)",title="Missingness by source"); axs[0,1].tick_params(axis="x",rotation=75,labelsize=9)
    by_source = d.groupby(["source_label","disease_label"]).size().unstack(fill_value=0).reindex(columns=["No disease","Disease"])
    by_source.plot(kind="bar",stacked=True,color=[palette["No disease"],palette["Disease"]],ax=axs[1,0]); axs[1,0].set(xlabel="Source",ylabel="Patients",title="Stacked disease counts"); axs[1,0].tick_params(axis="x",rotation=25); axs[1,0].legend(title="Disease status")
    overall_missing=raw[fields].isna().mean().mul(100).sort_values(ascending=False)
    overall_missing.index = [feature_names.get(value, value) for value in overall_missing.index]
    sns.barplot(x=overall_missing.index,y=overall_missing.values,color="#59A6C5",ax=axs[1,1]); axs[1,1].set(xlabel="Feature",ylabel="Missing (%)",title="Overall missingness"); axs[1,1].tick_params(axis="x",rotation=75,labelsize=9)
    for ax,label in zip(axs.flat,"abcd"): label_panel(ax,label); plot_style_axis(ax,numeric_x=False)
    fig.tight_layout(); save_matplotlib(fig,"fig_D_source_missingness.png")

    # Persist concise raw-number plot statistics.
    write_stats(stats)
    print(f"Matplotlib/Seaborn figures written: 13 individual plots + 4 composites + missingness + before/after; dpi=300; font=DejaVu Sans")


def build_scratch_p07_p08_p13():
    """Build raster p07, p08, and p13 figures."""
    rows = load_rows(DATA / "clean.csv")
    flags = load_rows(DATA / "imputed_flags.csv")
    stats = []
    draw_violin(rows, stats)
    observed_chol = [row for row, flag in zip(rows, flags) if flag["chol_imputed"] == "False"]
    draw_heatmap(observed_chol, stats, len(observed_chol))
    draw_parallel(observed_chol, flags, stats)


def main():
    """Run Matplotlib/Seaborn figure generation; project root and artifact paths define inputs."""
    build_matplotlib_figures()
    build_scratch_p07_p08_p13()


if __name__ == "__main__":
    main()
