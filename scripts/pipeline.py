from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, floor, split, col, when, sum
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead, when, avg, last
import re
import os
import sys
import shutil
import kagglehub
from concurrent.futures import ThreadPoolExecutor

from dataset import dataset

spark = SparkSession.builder \
    .appName("TelemetryProcessing") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

path = kagglehub.dataset_download("coni57/f1-2020-race-data")

datasetClasses = {}

fileList = os.listdir(path)

for dataFile in fileList:
    if 'Telemetry' in dataFile:
        id = re.search(r'_(\d+)', dataFile).group(1)
        tempdf = spark.read.csv(path+"/"+dataFile, header=True, inferSchema=True)
        datasetClasses[id] = dataset(spark, tempdf)