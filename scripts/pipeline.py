from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, floor, split, col, when, sum
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead, when, avg, last, lit
from pyspark.sql.types import StructType
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time
import re
import os
import sys
import shutil
import random
import kagglehub
from concurrent.futures import ThreadPoolExecutor

from dataset import dataset

# This script aims to create the ETL pipeline for the kaggle telemetry dataset
# Using Arirflow (which will be implemented after making sure that everything works like a clock)
# And trying to implement basic data engineering concepts (data security, gouvernance, orchestration...)
# Finally the datasets (for data anlysis and RNN training) will be stored in cloud data warehouse
# Between Azure Datalake and AWS S3

# 10 minutes for now, will be optimized further after first deployment
jdbc_path = "/home/gesser/Desktop/postgres/postgresql-42.7.7.jar"
db_url = "jdbc:postgresql://localhost:5432/telemetry"
db_table = "telemetry"
username = "floppa"
password = "flopps"
write_mode = "overwrite"
connection_properties = {
    "user": username,
    "password": password,
    "driver": "org.postgresql.Driver"
}

datasetClasses = {}
sessionsAndDrivers= {}

spark = SparkSession.builder \
    .appName("TelemetryProcessing") \
    .config("spark.jars", jdbc_path) \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

path = kagglehub.dataset_download("coni57/f1-2020-race-data")


fileList = os.listdir(path)

start = time.time()
id_pattern = re.compile(r'_(\d+)')

def process_file(dataFile):
    match = id_pattern.search(dataFile)
    if not match:
        return None, None, None
    id = match.group(1)
    full_path = os.path.join(path, dataFile)
    tempdf = pd.read_csv(full_path, engine='c')
    if 'ParticipantData' in dataFile:
        return id, 'drivers', tempdf
    elif 'SessionData' in dataFile:
        return id, 'session', tempdf
    return None, None, None

start = time.time()
# Importing and creating telemtry dataset classes (and stocking them in a dictionary)
for dataFile in fileList:
    if 'Telemetry' in dataFile:
        id = re.search(r'_(\d+)', dataFile).group(1)
        tempdf = spark.read.csv(os.path.join(path, dataFile), header=True, inferSchema=True)
        datasetClasses[id] = dataset(spark, tempdf)

#Importing and structuring sessions and drivers datasets
with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers to your CPU/io limits
    results = executor.map(process_file, fileList)

for id, key, df in results:
    if id is None:
        continue
    if id not in sessionsAndDrivers:
        sessionsAndDrivers[id] = {}
    sessionsAndDrivers[id][key] = df

print(f"extraction took {time.time()-start:.2f} seconds")

# Implementing data transformation functions, and storing locally the results

start = time.time()
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

# Combining sessions and drivers datasets (before loading them to PostgreSQL)
fullDriverDataset, fullSessionDataset = pd.DataFrame(), pd.DataFrame()
for i in sessionsAndDrivers.keys():
    temp1, temp2 = sessionsAndDrivers[i]['drivers'], sessionsAndDrivers[i]['drivers']
    temp1['raceId'] = i
    temp2['raceId'] = i
    fullDriverDataset = pd.concat([fullDriverDataset,temp1], ignore_index=True)
    fullSessionDataset = pd.concat([fullSessionDataset,temp2], ignore_index=True)

# donloading structured RNN datasets (before renaming and loading them to the cloud)

def upload_dataset(gp_id, ds_obj, folder):
    ds_obj.datasetDev(gp_id, f"{folder}/tempRNN")

# New: Modifying and concatinating telemetry datasets for postgresql
for gp_id, ds_obj in datasetClasses.items():
    ds_obj.addId(gp_id)

schema = random.choice(list(datasetClasses.values())).getSchema()
concatTelemetry = spark.createDataFrame([], schema)

for ds_obj in datasetClasses.values():
    concatTelemetry  = concatTelemetry.union(ds_obj.getDf())

folder_path = '/home/gesser/Desktop/f1_tyre_wear_rate_pred/data'

# Replace it with code to upload it directly to postgresql instead of localy
# concatTelemetry.coalesce(1).write.mode("overwrite").parquet(f"{folder_path}/PBI")

try:
    concatTelemetry.write \
        .format("jdbc") \
        .option("url", db_url) \
        .option("dbtable", db_table) \
        .options(**connection_properties) \
        .mode(write_mode) \
        .save()
    print(f"\nDataFrame successfully written to PostgreSQL table '{db_table}' in database '{db_url}' with mode '{write_mode}'.")

except Exception as e:
    print(f"\nError writing DataFrame to PostgreSQL: {e}")

# Renaming and (afterwards) uploading to the cloud

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