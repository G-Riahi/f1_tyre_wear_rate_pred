import pandas as pd
from datetime import datetime, timezone
import re
import os
import sys
import shutil
import kagglehub
from concurrent.futures import ThreadPoolExecutor

#for testing purposes:
import time

path = kagglehub.dataset_download("coni57/f1-2020-race-data")

sessionsAndDrivers= {}
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

sessionsAndDrivers = {}

# Use ThreadPoolExecutor to parallelize file reading
with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers to your CPU/io limits
    results = executor.map(process_file, fileList)

for id, key, df in results:
    if id is None:
        continue
    if id not in sessionsAndDrivers:
        sessionsAndDrivers[id] = {}
    sessionsAndDrivers[id][key] = df
    

print(f'it took {time.time()-start:.2f} seconds')

fullDriverDataset, fullSessionDataset = pd.DataFrame(), pd.DataFrame()
for i in sessionsAndDrivers.keys():
    temp1, temp2 = sessionsAndDrivers[i]['drivers'], sessionsAndDrivers[i]['drivers']
    temp1['raceId'] = i
    temp2['raceId'] = i
    fullDriverDataset = pd.concat([fullDriverDataset,temp1], ignore_index=True)
    fullSessionDataset = pd.concat([fullSessionDataset,temp2], ignore_index=True)

print(fullDriverDataset)