# UCI Heart Disease Lab

Run from repository root with Python 3.13 and Java 17:

1. `python src/01_clean.py`
2. `python src/02_select_transform.py`
3. `python src/03_plots.py`
4. `python src/04_dic.py`
5. `python src/05_models.py`
6. `python src/06_closet.py`
7. `python src/07_spark.py`
8. `python src/08_bench.py`

Phase outputs are written under `data/` and `out/`. Inputs are the four processed UCI Cleveland, Hungarian, Switzerland, and VA data files. Random seed: 42. Student-assumed ARM/DIC parameters: minimum support 0.3, minimum confidence 0.7, DIC interval M=10. Benchmark uses 2 runs, N=918/100000/1000000, Spark local[*], shuffle partitions 8, and 300-second per-run cap.

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, mlxtend, pyspark. Tested environment: Python 3.13, Java 17, PySpark 4.2.0, mlxtend 0.23.4.
