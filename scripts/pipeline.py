from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, floor, split, col, when, sum
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead, when, avg, last
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import time
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

# 10 minutes for now, will be optimized

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
        tempdf = spark.read.csv(os.path.join(path, dataFile), header=True, inferSchema=True)
        datasetClasses[id] = dataset(spark, tempdf)

# Implementing data transformation functions, and storing locally the results

def transform_dataset(ds_obj):
    ds_obj.transform()
    ds_obj.imputation()

with ThreadPoolExecutor(max_workers=11) as executor:
    futures = []
    print("start transfomation!!")
    start = time.time()
    for gp_id, ds_obj in datasetClasses.items():
        futures.append(executor.submit(transform_dataset, ds_obj))
        
    for future in futures:
        future.result()
    end=time.time()
    print(f"it took {end-start:.2f} seconds")

def upload_dataset(gp_id, ds_obj, folder):
    ds_obj.datasetPBI(gp_id, f"{folder}/PBI")
    ds_obj.datasetDev(gp_id, f"{folder}/tempRNN")

folder_path = '/home/gesser/Desktop/f1_tyre_wear_rate_pred/data'

with ThreadPoolExecutor(max_workers=11) as executor: 
    futures = []
    print("start!!")
    start = time.time()
    for gp_id, ds_obj in datasetClasses.items():
        futures.append(executor.submit(upload_dataset, gp_id, ds_obj, folder_path))
        
    for future in futures:
        future.result()
    end=time.time()
    print(f"it took {end-start:.2f} seconds")

spark.stop()

base_dir = "tempRNN"
parquet_data = []
pattern = r"\d+"

#catch all details [driver number, session number, chunk number] so that we can define he file name with these caracterestics

for i in os.scandir(f"{folder_path}/{base_dir}"):
    if i.is_dir():
        #print(i.name)
        for j in os.scandir(i.path):
            if j.is_dir():
                #print('-------'+re.findall(pattern, str(j.name))[0])
                for k in os.scandir(j.path):
                    if k.is_dir():
                        #print('--'+re.findall(pattern, str(k.name))[0])
                        for l in os.scandir(k.path):
                            if l.is_file() and l.name.endswith(".snappy.parquet"):
                                parquet_data.append([i.name, re.findall(pattern, str(j.name))[0], re.findall(pattern, str(k.name))[0], l.path])

warehouse = "RNN"

def copy_file(file_path):
    shutil.copy2(file_path[3], f"{folder_path}/{warehouse}/{file_path[0]}_{file_path[1]}_{file_path[2]}.parquet"
)


os.makedirs(f"{folder_path}/{warehouse}", exist_ok=True)
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(copy_file, parquet_data)

shutil.rmtree(f"{folder_path}/{base_dir}")