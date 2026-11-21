#!/usr/bin/env python3
"""Clean raw heart data and write requested profiles and figures."""

import csv
import math
import statistics
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw_combined.csv"
NUMERIC = ["trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
MODE_IMPUTE = ["fbs", "restecg", "exang"]
DROP_COLS = ["ca", "thal"]
SOURCES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]


def read_rows(path):
    """Read CSV rows, mapping blank fields to None; path selects input file."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames)


def num(value):
    """Parse numeric CSV value; value is text or None; preserve missingness."""
    if value is None or value == "":
        return None
    return float(value)


def format_value(value):
    """Serialize clean value; value is numeric, categorical, or None."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def mode(values):
    """Return most frequent value with numeric tie break; values excludes missing."""
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], float(value)))


def median(values):
    """Return median of observed values; values excludes missing."""
    return statistics.median(values) if values else None


def quantile(values, fraction):
    """Return linearly interpolated quantile; values must be non-empty numbers."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def png_chunk(kind, payload):
    """Encode one PNG chunk; kind is four-byte type and payload is chunk bytes."""
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
}


class Raster:
    """Small RGB canvas for dependency-free 300-DPI chart PNGs."""

    def __init__(self, width, height, color=(255, 255, 255)):
        """Create blank canvas; width and height are pixels, color is RGB tuple."""
        self.width, self.height = width, height
        self.pixels = bytearray(color * (width * height))

    def pixel(self, x, y, color):
        """Set in-bounds pixel; x/y are coordinates and color is RGB tuple."""
        if 0 <= x < self.width and 0 <= y < self.height:
            start = (y * self.width + x) * 3
            self.pixels[start:start + 3] = bytes(color)

    def line(self, x0, y0, x1, y1, color):
        """Draw Bresenham line; endpoints are pixels and color is RGB tuple."""
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx - dy
        while True:
            self.pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice > -dy:
                error -= dy
                x0 += sx
            if twice < dx:
                error += dx
                y0 += sy

    def rect(self, x0, y0, x1, y1, color):
        """Fill rectangle; bounds are pixel coordinates and color is RGB tuple."""
        for y in range(max(0, y0), min(self.height, y1)):
            start, end = (y * self.width + max(0, x0)) * 3, (y * self.width + min(self.width, x1)) * 3
            if end > start:
                self.pixels[start:end] = bytes(color) * ((end - start) // 3)

    def text(self, x, y, label, color=(20, 30, 45), scale=2):
        """Draw compact uppercase bitmap text; label is string, x/y origin, scale pixel size."""
        cursor = x
        for char in label.upper():
            glyph = FONT.get(char)
            if glyph:
                for row, bits in enumerate(glyph):
                    for col, bit in enumerate(bits):
                        if bit == "1":
                            self.rect(cursor + col * scale, y + row * scale,
                                      cursor + (col + 1) * scale, y + (row + 1) * scale, color)
            cursor += 6 * scale

    def save(self, path):
        """Write RGB PNG with 300-DPI metadata; path is destination PNG file."""
        raw = b"".join(b"\x00" + bytes(self.pixels[y * self.width * 3:(y + 1) * self.width * 3])
                       for y in range(self.height))
        png = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
               + png_chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1))
               + png_chunk(b"IDAT", zlib.compress(raw)) + png_chunk(b"IEND", b""))
        path.write_bytes(png)


def write_missingness_figure(rows, columns, path):
    """Plot pre-imputation missing percent by source; rows/data and path define chart."""
    width, height = 1500, 700
    image = Raster(width, height)
    left, top, right, bottom = 100, 100, 1420, 570
    image.text(390, 24, "MISSINGNESS BY COLUMN AND SOURCE", scale=3)
    image.text(108, 78, "PERCENT MISSING", scale=2)
    image.text(710, 620, "COLUMN", scale=2)
    colors = [(40, 110, 180), (230, 130, 40), (50, 155, 100), (175, 80, 145)]
    for tick in range(0, 101, 20):
        y = bottom - int((bottom - top) * tick / 100)
        image.line(left, y, right, y, (220, 225, 230))
        image.text(42, y - 6, str(tick), scale=1)
    image.line(left, top, left, bottom, (25, 35, 45))
    image.line(left, bottom, right, bottom, (25, 35, 45))
    group_width = (right - left) // len(columns)
    bar_width = max(4, group_width // (len(SOURCES) + 2))
    for index, column in enumerate(columns):
        group_x = left + index * group_width
        for source_index, source in enumerate(SOURCES):
            values = [row[column] for row in rows if row["source"] == source]
            percent = 100 * sum(value is None or value == "" for value in values) / len(values) if values else 0
            bar_x = group_x + (source_index + 1) * bar_width
            y = bottom - int((bottom - top) * percent / 100)
            image.rect(bar_x, y, bar_x + bar_width - 2, bottom, colors[source_index])
        label = column.upper()
        image.text(group_x + 1, bottom + 16, label, scale=1)
    for index, source in enumerate(SOURCES):
        x = 120 + index * 320
        image.rect(x, 665, x + 24, 685, colors[index])
        image.text(x + 32, 666, source.replace("processed.", "").replace(".data", "").upper(), scale=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_boxplots(before, after, path):
    """Plot before/after distributions for requested numeric columns; lists/path are inputs."""
    width, height = 1600, 900
    image = Raster(width, height)
    image.text(460, 24, "NUMERIC FEATURES BEFORE AND AFTER IMPUTATION", scale=3)
    colors = [(60, 130, 190), (220, 120, 60)]
    panel_w, top, bottom = 370, 150, 760
    for index, column in enumerate(NUMERIC):
        left = 70 + index * 385
        right = left + panel_w - 25
        image.text(left + 70, 85, column.upper(), scale=2)
        observed = [num(row[column]) for row in before + after if num(row[column]) is not None]
        lo, hi = min(observed), max(observed)
        pad = max((hi - lo) * .05, 1)
        lo, hi = lo - pad, hi + pad
        def ypos(value):
            return bottom - int((bottom - top) * (value - lo) / (hi - lo))
        image.line(left + 35, top, left + 35, bottom, (30, 40, 50))
        image.line(left + 35, bottom, right, bottom, (30, 40, 50))
        for sample_index, dataset in enumerate((before, after)):
            vals = [num(row[column]) for row in dataset if num(row[column]) is not None]
            if not vals:
                continue
            q1, med, q3 = quantile(vals, .25), quantile(vals, .5), quantile(vals, .75)
            iqr = q3 - q1
            low = min(value for value in vals if value >= q1 - 1.5 * iqr)
            high = max(value for value in vals if value <= q3 + 1.5 * iqr)
            center = left + 115 + sample_index * 130
            image.line(center, ypos(low), center, ypos(high), colors[sample_index])
            image.line(center - 24, ypos(low), center + 24, ypos(low), colors[sample_index])
            image.line(center - 24, ypos(high), center + 24, ypos(high), colors[sample_index])
            image.rect(center - 34, ypos(q3), center + 35, ypos(q1), colors[sample_index])
            image.line(center - 34, ypos(med), center + 35, ypos(med), (20, 20, 20))
            image.text(center - 22, bottom + 20, "BEFORE" if sample_index == 0 else "AFTER", scale=1)
    image.text(725, 815, "VALUE", scale=2)
    image.text(720, 855, "BOXES SHOW Q1 MEDIAN Q3; WHISKERS USE 1.5 IQR", scale=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run():
    """Apply specified cleaning, report decisions, and write clean data and artifacts."""
    rows, fields = read_rows(INPUT)
    input_rows = [dict(row) for row in rows]
    input_row_count = len(rows)
    seen, unique_rows = set(), []
    duplicate_count = 0
    duplicate_sources = Counter()
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in seen:
            duplicate_count += 1
            duplicate_sources[row["source"]] += 1
        else:
            seen.add(key)
            unique_rows.append(row)
    print(f"Exact duplicate rows to drop: {duplicate_count}", flush=True)
    print("Duplicate rows by source:", dict(sorted(duplicate_sources.items())))
    rows = unique_rows
    initial_rows = len(rows)
    raw_box_before = [dict(row) for row in rows]
    reports = {
        "trestbps_zero": sum(num(row["trestbps"]) == 0 for row in rows if num(row["trestbps"]) is not None),
        "chol_zero": sum(num(row["chol"]) == 0 for row in rows if num(row["chol"]) is not None),
        "oldpeak_negative": sum(num(row["oldpeak"]) < 0 for row in rows if num(row["oldpeak"]) is not None),
        "thalach_outside_60_220": sum(not 60 <= num(row["thalach"]) <= 220 for row in rows if num(row["thalach"]) is not None),
    }
    for row in rows:
        if num(row["chol"]) == 0:
            row["chol"] = None
        if num(row["trestbps"]) == 0:
            row["trestbps"] = None
    for column in CATEGORICAL:
        for row in rows:
            if row[column] == "":
                row[column] = None
    for row in rows:
        for column in NUMERIC:
            if row[column] == "":
                row[column] = None
    imputed_flags = [{f"{column}_imputed": row[column] is None or row[column] == ""
                      for column in NUMERIC + MODE_IMPUTE} for row in rows]
    flag_path = ROOT / "data" / "imputed_flags.csv"
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    with flag_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[f"{column}_imputed" for column in NUMERIC + MODE_IMPUTE])
        writer.writeheader()
        writer.writerows(imputed_flags)
    # Include all source fields so high-missingness drop decisions remain visible.
    chart_columns = [field for field in fields if field != "source"]
    write_missingness_figure(rows, chart_columns, ROOT / "out" / "figs" / "missingness_bar.png")
    before_impute = [dict(row) for row in rows]
    before_nan = {field: sum(row.get(field) is None or row.get(field) == "" for row in input_rows)
                  for field in fields}
    for row in rows:
        if row["slope"] is None:
            row["slope"] = "-1"
    # IQR counts computed on observed values before imputation; natural extremes stay intact.
    outlier_counts = {}
    for column in NUMERIC:
        values = [num(row[column]) for row in rows if num(row[column]) is not None]
        q1, q3 = quantile(values, .25), quantile(values, .75)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outlier_counts[column] = sum(value < low or value > high for value in values)
    global_fills = {column: median([num(row[column]) for row in rows if num(row[column]) is not None])
                    for column in NUMERIC}
    for column in MODE_IMPUTE:
        vals = [row[column] for row in rows if row[column] is not None]
        global_fills[column] = mode(vals) if vals else None
    imputation_counts = defaultdict(int)
    imputation_counts_by_source = defaultdict(int)
    for column in NUMERIC + MODE_IMPUTE:
        for row in rows:
            if row[column] is None or row[column] == "":
                row[column] = global_fills[column]
                imputation_counts[column] += 1
                imputation_counts_by_source[(column, row["source"])] += 1
    clean_fields = [field for field in fields if field not in DROP_COLS] + ["num_bin"]
    for row in rows:
        row["num_bin"] = int(float(row["num"]) > 0)
    for column in CATEGORICAL:
        for row in rows:
            if row[column] is not None:
                row[column] = str(row[column])
    write_boxplots(raw_box_before, rows, ROOT / "out" / "figs" / "boxplots_before_after.png")
    after_nan = {field: sum(row.get(field) is None or row.get(field) == "" for row in rows)
                 for field in clean_fields}
    output = ROOT / "data" / "clean.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=clean_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in clean_fields})
    table_path = ROOT / "out" / "tables" / "clean_profile.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["column", "before_nan", "after_nan"])
        for field in before_nan:
            writer.writerow([field, before_nan[field], after_nan.get(field, "dropped")])
    log = [
        "# Phase 1 preprocessing log", "", "action | rows or cells affected | reason | alternative considered",
        f"Exact duplicate removal | {duplicate_count} rows dropped; {initial_rows} remain | Remove repeated records without changing unique observations | Keep duplicates and let them overweight estimates.",
        "Duplicate source attribution | " + "; ".join(f"{source}: {count}" for source, count in sorted(duplicate_sources.items())) + " | Identify source of every dropped duplicate | Ignore provenance of removed rows.",
        f"Sentinel checks | trestbps==0: {reports['trestbps_zero']}; chol==0: {reports['chol_zero']}; oldpeak<0: {reports['oldpeak_negative']}; thalach<60 or >220: {reports['thalach_outside_60_220']} rows | Report requested anomalies; convert only zero trestbps and zero chol to missing | Drop or cap values; rejected to preserve natural ranges.",
        f"Drop ca and thal | {sum(1 for r in rows if True)} rows retained; 2 columns removed | Both exceed 50% missing in raw file | Keep columns and impute; rejected due to high missingness.",
        f"Slope missing level | {sum(row['slope'] == '-1' for row in rows)} cells coded -1 | Preserve missingness as explicit categorical level | Drop rows or impute most common slope.",
        f"Categorical conversion | {len(rows) * len(CATEGORICAL)} cells treated as categorical; missing levels retained | Mark nominal codes as categories and slope as category | Treat codes as continuous numeric measurements.",
        f"Add num_bin | {len(rows)} cells created; num retained | Binary target indicates any disease | Replace multiclass num; rejected to retain original target.",
        "IQR outlier report | " + "; ".join(f"{col}: {count}" for col, count in outlier_counts.items()) + " | Flag 1.5×IQR extremes for review | Remove/cap extremes; rejected to preserve natural observations.",
        "Global median imputation | " + "; ".join(f"{col}: {imputation_counts[col]}" for col in NUMERIC) + " | Keep source out of features; per-source statistics would encode source and leak its disease prevalence (Switzerland is ~94% positive); Switzerland has no observed chol, so use global observed-value median | Per-source median or row deletion.",
        "Global mode imputation | " + "; ".join(f"{col}: {imputation_counts[col]}" for col in MODE_IMPUTE) + " | Keep source out of features to avoid source leakage; compute mode from observed values across rows | Per-source mode or row deletion.",
        "Imputation flags | " + str(sum(sum(flags.values()) for flags in imputed_flags)) + " cells flagged in row-aligned data/imputed_flags.csv | Preserve which source values were missing before filling | Add flags to clean.csv; rejected to keep requested feature file unchanged.",
        "Figures | 2 PNG files written at 300 DPI; boxplot before uses raw zeros and after uses imputed values | Show source missingness and requested numeric distributions | Other plots omitted per scope.",
        f"Save outputs | {len(rows)} rows × {len(clean_fields)} columns in clean.csv; NaN profile saved | Provide auditable cleaned dataset and missingness counts | Overwrite raw input; rejected to preserve source.",
    ]
    (ROOT / "out" / "log.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("Reported sentinel counts:", reports)
    print("IQR outlier counts:", outlier_counts)
    for column in ("chol", "fbs"):
        print(f"{column} imputed counts by source:", {source: imputation_counts_by_source[(column, source)] for source in SOURCES})
    print(f"Shape before cleaning: ({input_row_count}, {len(fields)})")
    print(f"Shape after cleaning: ({len(rows)}, {len(clean_fields)})")


if __name__ == "__main__":
    run()
