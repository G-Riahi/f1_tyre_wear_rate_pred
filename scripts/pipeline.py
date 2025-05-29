from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, floor, split, col, when, sum
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead, when, avg, last
from datetime import datetime, timezone
import re
import os
import sys
import shutil
import kagglehub
from concurrent.futures import ThreadPoolExecutor

from dataset import dataset

# This script aims to create the ETL pipeline for the kaggle telemetry dataset
# Using Arirflow (which will be implemented after making sure that everything works like a clock)
# And trying to implement basic data engineering concepts (data security, gouvernance, orchestration...)
# Finally the datasets (for data anlysis and RNN training) will be stored in cloud data warehouse
# Between Azure Datalake and AWS S3

spark = SparkSession.builder \
    .appName("TelemetryProcessing") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

path = kagglehub.dataset_download("coni57/f1-2020-race-data")

datasetClasses = {}

fileList = os.listdir(path)

# Importing and creating telemtry dataset classes (and stocking them in a dictionary)
for dataFile in fileList:
    if 'Telemetry' in dataFile:
        id = re.search(r'_(\d+)', dataFile).group(1)
        tempdf = spark.read.csv(path+"/"+dataFile, header=True, inferSchema=True)
        datasetClasses[id] = dataset(spark, tempdf)

# Implementing data transformation functions (for data analysis at least for now)
for i in datasetClasses.keys():
    datasetClasses[i].transform()
    datasetClasses[i].imputation()

spark.stop()