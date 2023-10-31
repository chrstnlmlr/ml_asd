"""
input: OpenPose output (JSON)
output: dataframe with all JSON data from OpenPose (csv)
"""

import os
import numpy as np
import pandas as pd
import json
from cherrypicker import CherryPicker
import itertools
import time

base_path = "//Users/christian/Library/Mobile Documents/com~apple~CloudDocs/promotion/github_ml_ass/"
json_path = "/Users/christian/promotion/data/JSON_OpenPose_ID_Auto"

body_parts = ["Nose", "Neck", "Shoulder", "Elbow", "RWrist", "LShoulder", "LElbow",
              "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
              "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
              "LHeel", "RBigToe", "RSmallToe", "RHeel"]

value_names = ["x", "y", "acc"]

column_names = [f"{n}_{m}" for n, m in itertools.product(body_parts, value_names)]
column_names.insert(0, "person_id")
column_names.insert(77, "filename")

def load_json(json_files):
    counter = 0
    printcounter = 0
    df_set = pd.DataFrame()
    for file in json_files:
        counter += 1
        printcounter += 1
        if printcounter == 10000:
            print(counter)
            print(file)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            printcounter = 0
        with open(json_path+"/"+file) as json_data:
            data = json.load(json_data)
            picker = CherryPicker(data, n_jobs=-1)
            try:
                flat = picker['people'].flatten().get()
                df = pd.DataFrame(flat)
            except:
                df = pd.DataFrame(np.zeros((1, 76)))
            df['filename'] = file
            df_set = df_set.append(df)
            json_data.close
    df_set.columns = column_names
    return df_set

def json_loader(json_files):
    # json_loader: split JSON_files into 10 sets
    df_full = pd.DataFrame()
    json_files_split = np.array_split(json_files, 10, axis=0)
    for n in range(10):
        print(n)
        json_files_list = json_files_split[n].tolist()
        json_df = load_json(json_files_list)
        df_full = df_full.append(json_df)
    return df_full

# json file list
json_files = [file for file in os.listdir(json_path)]

# load and process json data
json_df = json_loader(json_files)

# save data
data_folder = os.path.join(base_path, "data/csv")
json_df.to_csv(os.path.join(data_folder, "json_df.csv"), sep=';', index=False)
