# Exploratory Data Analysis & Visualization Inferences

Derived directly from `out/tables/plot_stats.md`, `clean.csv`, and preprocessing audits.

---

### Plot 01: Multi-Class Target Distribution (`p01_num_pie.png`)
- **Observation:** Class 0 (absence of disease) forms the single largest category at 410 patients (44.7%), while mild-to-severe disease classes 1 through 4 account for 265, 108, 107, and 28 patients respectively.
- **Inference:** Although the multi-class spectrum exhibits right-tail sparsity in severe disease (Stage 4 represents only 3.1% of patients), binarizing into `num_bin` produces a well-balanced binary target: 410 disease-free (44.7%) versus 508 disease-present (55.3%).
- **Clinical/Analytical Relevance:** This justified collapsing stages into a binary target for classification and association rule mining without inducing severe class-imbalance skew.

---

### Plot 02: Age Distribution by Disease Status (`p02_age_histogram.png`)
- **Observation:** Disease-free patients (`num_bin=0`, $n=410$) have a median age of 51 years, whereas patients with heart disease (`num_bin=1`, $n=508$) have a median age of 57 years.
- **Inference:** The age histogram for positive cases shifts rightward by 6 years, with peak disease incidence concentrated in the 55–65 age bracket, while negative cases peak between 45 and 54.
- **Clinical/Analytical Relevance:** Age shows moderate positive correlation with cardiac risk ($r_s \approx 0.30$), supporting the clinical premise that arterial stiffening and plaque accumulation increase steadily past age 50.

---

### Plot 03: ST Depression (Oldpeak) Dot Plot (`p03_oldpeak_dotplot.png`)
- **Observation:** Oldpeak exhibits heavy zero-inflation, with 392 patients (42.7%) clustered exactly at $0.0\text{ mm}$, followed by peaks at $1.0\text{ mm}$ ($n=123$), $0.5\text{ mm}$ ($n=115$), and $2.0\text{ mm}$ ($n=100$).
- **Inference:** The distribution is markedly non-Gaussian with a long right tail extending to $6.2\text{ mm}$, representing severe exercise-induced myocardial ischemia.
- **Clinical/Analytical Relevance:** Discretizing oldpeak into three clinical bins ($0\text{ mm}$, $(0, 2)\text{ mm}$, and $\ge 2.0\text{ mm}$) properly isolates physiological normality from moderate and severe exercise-induced repolarization abnormalities.

---

### Plot 04: Stem-and-Leaf Age Display (`p04_age_stem_leaf.png`)
- **Observation:** In the representative $n=80$ seeded audit sample (seed 42), patient ages range from a minimum of 34 to a maximum of 74, with a median of 55.5 years.
- **Inference:** The sample mirrors the full cohort's bell-shaped dispersion around the 50s and 60s stem rows without isolated demographic outliers.
- **Clinical/Analytical Relevance:** Confirms that patient enrollment across all four hospital registries consistently focused on middle-aged to elderly adults presenting for angiography.

---

### Plot 05: Age vs. Maximum Heart Rate Scatter (`p05_age_thalach_scatter.png`)
- **Observation:** Maximum heart rate (`thalach`) correlates negatively with age in both groups, but the downward slope is steeper in healthy patients ($r_s = -0.3537$, $n=410$) than in disease patients ($r_s = -0.1640$, $n=508$).
- **Inference:** In cardiac patients, chronotropic incompetence and ischemic symptoms artificially cap maximum exertion across all adult ages, blunting the normal physiological age-heart rate gradient.
- **Clinical/Analytical Relevance:** `thalach` serves as a potent diagnostic discriminator, as disease cases exhibit suppressed maximum heart rates regardless of age tier.

---

### Plot 06: Grouped Tukey Boxplots (`p06_boxplots_by_num_bin.png`)
- **Observation:** Between healthy and diseased cohorts, median `thalach` drops from $150\text{ bpm}$ down to $130\text{ bpm}$, median `oldpeak` jumps from $0.0\text{ mm}$ to $1.0\text{ mm}$, while median age increases from 51 to 57 against an overall cohort median of 54.
- **Inference:** Interquartile ranges for `oldpeak` and `thalach` demonstrate marked separation with minimal overlap in their core quartiles across disease status.
- **Clinical/Analytical Relevance:** Confirms `thalach` and `oldpeak` as primary numerical separators, justifying their high ranking in mutual information and ANOVA screening.

---

### Plot 07: Heart Rate Violins by Chest Pain Category (`p07_thalach_violin_by_cp.png`)
- **Observation:** Median `thalach` across chest pain categories 1 to 4 is $145\text{ bpm}$ (typical angina), $150\text{ bpm}$ (atypical angina), $146\text{ bpm}$ (non-anginal), and drops sharply to $130\text{ bpm}$ for category 4 (asymptomatic).
- **Inference:** Categories 1 through 3 display broad, higher-rate violin profiles centered around $145\text{--}150\text{ bpm}$, whereas category 4 shows a distinctly lowered, bimodal density.
- **Clinical/Analytical Relevance:** Asymptomatic presentation (`cp=4`) frequently reflects advanced, "silent" ischemia or severe disease where exercise tolerance is severely impaired.

---

### Plot 08: Rank Correlation Heatmap (`p08_spearman_heatmap.png`)
- **Observation:** Evaluating complete-case observed cholesterol records shows the strongest rank correlations with `num_bin` are `oldpeak` ($r_s = +0.4688$) and `thalach` ($r_s = -0.3652$), followed by `age` ($+0.2967$), `trestbps` ($+0.1630$), and `chol` ($+0.1276$).
- **Inference:** Collinearity among predictors remains low to moderate across the board ($|r_s| < 0.35$ between all feature pairs), indicating minimal multi-collinearity redundancy.
- **Clinical/Analytical Relevance:** Resting blood pressure and total cholesterol have weak bivariate monotonic associations with angiographic disease presence compared to dynamic stress metrics.

---

### Plot 09: Disease Prevalence by Chest Pain & Sex (`p09_disease_rate_cp_sex.png`)
- **Observation:** Disease rate climbs dramatically across chest pain types: $13.9\%$ for atypical angina (`cp=2`), $35.5\%$ for non-anginal pain (`cp=3`), $43.5\%$ for typical angina (`cp=1`), and spikes to $79.0\%$ for asymptomatic patients (`cp=4`).
- **Inference:** Male patients display substantially higher disease prevalence than females across every chest pain category (overall male prevalence $>60\%$ vs female $<30\%$).
- **Clinical/Analytical Relevance:** The interaction of male sex and asymptomatic chest pain represents the highest risk profile for significant coronary artery stenosis in this cohort.

---

### Plot 10: Positive Case Counts Across Clinical Centers (`p10_num_bin_by_source.png`)
- **Observation:** Observed positive case counts vary significantly across clinical centers: Cleveland ($n=139$), Hungarian ($n=106$), Switzerland ($n=115$), and VA Long Beach ($n=148$).
- **Inference:** Proportional prevalence differs vastly by site: Switzerland recorded disease in nearly $93.5\%$ of its 123 patients, whereas the Hungarian registry had under $37\%$ positive cases.
- **Clinical/Analytical Relevance:** Confirms substantial institutional selection bias across datasets, highlighting why clinical center origin had to be excluded from predictive features to prevent label leakage.

---

### Plot 11: Feature Pairwise Scatter & Rank Associations (`p11_pair_plot.png`)
- **Observation:** Pairwise rank correlations confirm `age-thalach` ($r_s = -0.3286$), `age-oldpeak` ($r_s = +0.2758$), `age-trestbps` ($r_s = +0.2469$), `thalach-oldpeak` ($r_s = -0.1894$), and `thalach-trestbps` ($r_s = -0.0830$).
- **Inference:** Healthy patients cluster prominently in the high-`thalach`, zero-`oldpeak` quadrant, while diseased patients disperse into elevated `oldpeak` and low-`thalach` spaces.
- **Clinical/Analytical Relevance:** The lack of strict linear boundaries confirms the multi-factorial nature of coronary disease, validating non-linear distance and association rule approaches.

---

### Plot 12: Dual Kernel Density Estimation Panels (`p12_kde_age_thalach.png`)
- **Observation:** Density curves for `thalach` illustrate a clean modal shift from $150\text{ bpm}$ in healthy controls down to $130\text{ bpm}$ in cardiac cases, with healthy cases exhibiting higher kurtosis.
- **Inference:** The age KDE exhibits overlapping support but with a flattened, older plateau for disease cases ($55\text{--}65$ years) contrasted against a sharper healthy mode at $51$ years.
- **Clinical/Analytical Relevance:** The smooth probability densities corroborate the Tukey boxplot statistics, validating the tertile and quartile discretization boundaries used for ARM.

---

### Plot 13: Parallel Coordinates Profile & Imputation Sensitivity (`p13_parallel_coordinates.png`)
- **Observation:** In the 200-patient normalized profile (105 positive, 95 negative), disease cases systematically trace downward slopes on `thalach` and upward spikes on `oldpeak` and `trestbps`.
- **Inference & Honest Risk Assessment:** The global median imputation of cholesterol (201 cases imputed at the constant value of $223\text{ mg/dL}$) artificially collapses cholesterol variance for nearly a quarter of the dataset, particularly for the entire Swiss sub-cohort where cholesterol was unrecorded.
- **Clinical/Clustering Impact:** This artificial variance shrinkage poses a real risk of inducing an artificial clustering artifact in distance-based algorithms like K-Means; recognizing and reporting this artifact is critical for sound clinical machine learning validation.
