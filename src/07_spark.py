#!/usr/bin/env python3
"""Check Java and Spark runtime, then record local Spark startup metadata."""
import shutil
import os
import sys
import subprocess
import time
from pathlib import Path
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent

def main():
    """Validate Java first, create local[*] Spark with eight shuffle partitions, and save startup seconds."""
    java = shutil.which("java")
    if not java:
        raise RuntimeError("Java missing; Spark fallback: sklearn/NumPy-only benchmark")
    version = subprocess.run([java, "-version"], capture_output=True, text=True, check=True)
    print("JAVA", (version.stderr or version.stdout).splitlines()[0])
    started = time.perf_counter()
    spark = SparkSession.builder.master("local[*]").appName("heart-lab-benchmark").config("spark.sql.shuffle.partitions", "8").getOrCreate()
    seconds = time.perf_counter() - started
    spark.sparkContext.setLogLevel("ERROR")
    (ROOT / "out/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "out/tables/spark_startup.txt").write_text(f"startup_seconds={seconds:.6f}\nmaster={spark.sparkContext.master}\nshuffle_partitions=8\n", encoding="utf-8")
    print("SPARK_STARTUP_SECONDS", f"{seconds:.6f}", "MASTER", spark.sparkContext.master, "SHUFFLE_PARTITIONS", spark.conf.get("spark.sql.shuffle.partitions"))
    spark.stop()

if __name__ == "__main__": main()
