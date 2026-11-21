# Phase 1 preprocessing log

action | rows or cells affected | reason | alternative considered
Exact duplicate removal | 2 rows dropped; 918 remain | Remove repeated records without changing unique observations | Keep duplicates and let them overweight estimates.
Duplicate source attribution | processed.hungarian.data: 1; processed.va.data: 1 | Identify source of every dropped duplicate | Ignore provenance of removed rows.
Sentinel checks | trestbps==0: 1; chol==0: 172; oldpeak<0: 12; thalach<60 or >220: 0 rows | Report requested anomalies; convert only zero trestbps and zero chol to missing | Drop or cap values; rejected to preserve natural ranges.
Drop ca and thal | 918 rows retained; 2 columns removed | Both exceed 50% missing in raw file | Keep columns and impute; rejected due to high missingness.
Slope missing level | 307 cells coded -1 | Preserve missingness as explicit categorical level | Drop rows or impute most common slope.
Categorical conversion | 5508 cells treated as categorical; missing levels retained | Mark nominal codes as categories and slope as category | Treat codes as continuous numeric measurements.
Add num_bin | 918 cells created; num retained | Binary target indicates any disease | Replace multiclass num; rejected to retain original target.
IQR outlier report | trestbps: 27; chol: 23; thalach: 2; oldpeak: 16 | Flag 1.5×IQR extremes for review | Remove/cap extremes; rejected to preserve natural observations.
Global median imputation | trestbps: 60; chol: 201; thalach: 55; oldpeak: 62 | Keep source out of features; per-source statistics would encode source and leak its disease prevalence (Switzerland is ~94% positive); Switzerland has no observed chol, so use global observed-value median | Per-source median or row deletion.
Global mode imputation | fbs: 90; restecg: 2; exang: 55 | Keep source out of features to avoid source leakage; compute mode from observed values across rows | Per-source mode or row deletion.
Imputation flags | 525 cells flagged in row-aligned data/imputed_flags.csv | Preserve which source values were missing before filling | Add flags to clean.csv; rejected to keep requested feature file unchanged.
Figures | 2 PNG files written at 300 DPI; boxplot before uses raw zeros and after uses imputed values | Show source missingness and requested numeric distributions | Other plots omitted per scope.
Save outputs | 918 rows × 14 columns in clean.csv; NaN profile saved | Provide auditable cleaned dataset and missingness counts | Overwrite raw input; rejected to preserve source.

## Phase 2 selection and transformation

action | affected | reason | alternative
Feature tests and tiers | 11 features tested; selected tiers: age, thalach, oldpeak, sex, cp, exang, slope | Use one point-biserial test for each numeric feature and chi-square/Cramer's V for categorical features; MI uses quantile-discretized numeric values; p<0.05 alone can flag small effects as significant at n=918, so selected set uses effect tiers | Select by p-value alone.
Effect tiers | strong >=0.30; moderate >=0.15 and <0.30; weak <0.15 | Use effect magnitude to select features; thresholds assumed by student | Use p<0.05 alone.
Observed-only sensitivity | Recomputed effects for trestbps, chol, thalach, oldpeak, fbs, restecg, exang using rows whose matching imputation flag is False | Show association on observed values only | Use imputed rows for all estimates.
Spearman screening | 0 pairs exceed |rho|>0.7 | Flag strong numeric associations for review | Remove correlated feature now; deferred.
ARM discretization | 918 transactions; 35 item levels | Use age<45/45-54/55-64/65+; trestbps<120/120-139/>=140; chol<200/200-239/>=240; pooled thalach tertiles at 127,150; oldpeak 0/(0,2)/>=2; encode below-zero oldpeak separately to preserve observed negatives; omit slope=-1 | Drop unbinned negatives or include missing slope as an item.
ARM one-hot | 918 rows × 35 item columns; no transaction index | Make binary item incidence available for mining; disease tokens remain ARM outcome items | Keep only comma-joined item list or add an index.
Selected matrices | 918 rows; 15 model X columns; 24 ARM items | Keep strong/moderate features only while retaining y and disease ARM items | Drop weak features from full matrices.
Distance matrix | 918 rows × 23 columns | Z-score numeric features; one-hot categorical features including slope=-1; exclude num, num_bin, source from X and place num_bin in y | Include source or drop missing slope level.

Phase 08 benchmark runtime note | first Spark benchmark attempt failed because worker Python resolved to 3.14 while driver used 3.13; set PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON to the driver interpreter, reran successfully | environment mismatch; no fallback used.

Benchmark skipped | kmeans N=10000000 | available RAM=4205363200 bytes < 6442450944; no run started.

Benchmark skipped | kmeans N=10000000 | available RAM=4499259392 bytes < 6442450944; no run started.
