# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import json
from cherrypicker import CherryPicker
import itertools
import time

########## INFO ##########
"""
input: input: OpenPose JSON filepath
output: dataframe with all JSON data from OpenPose
"""
########## SET PATHS ##########

# change working directory
os.chdir("//Users/christian/Library/Mobile Documents/com~apple~CloudDocs/promotion/github_ml_ass/")
#print(os.getcwd())

# path to Open Pose JSON output
json_path = "/Users/christian/promotion/data/JSON_OpenPose"

########## JSON file variables ##########

# 25 body parts
body_parts = ["Nose", "Neck", "Shoulder", "Elbow", "RWrist", "LShoulder", "LElbow", 
              "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", 
              "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", 
              "LHeel", "RBigToe", "RSmallToe", "RHeel"]

# 3 values
value_names = ["x", "y", "acc"]

# column names
column_names = list(itertools.product(body_parts, value_names))
column_names = [n+"_"+m for n,m in column_names]
column_names.insert(0, "person_id")
column_names.insert(77, "filename")

########## LOAD DATA ##########

json_files = [file for file in os.listdir(json_path)] 

########## FUNCTIONS ##########

def load_json(json_files):
    """
    load JSON-files and append to dataframe
    (every person in a single row)
    """
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
            picker = CherryPicker(data, n_jobs=-1) #n_jobs parameter to specify the number of CPUs you wish to use (a value of -1 will mean all CPUs are used)
            try:
                flat = picker['people'].flatten().get()
                df = pd.DataFrame(flat)
            except:
                df = pd.DataFrame(np.zeros((1, 76)))
            # append data  
            df['filename'] = file # add filename
            df_set = df_set.append(df)
            json_data.close
    df_set.columns = column_names    
    return df_set    

def json_loader(json_files):
    """
    split JSON_files into 10 sets (because load_json slows down with every iteration)
    """
    df_full = pd.DataFrame()
    json_files_split = np.array_split(json_files, 10, axis=0)
    for n in range(10):
        print(n)
        json_files_list = json_files_split[n].tolist()
        json_df = load_json(json_files_list)
        df_full = df_full.append(json_df)
    return df_full
      
########## MAIN ##########

json_df = json_loader(json_files)

########## SAVE DATA ##########

# save JSON dataframe
json_df.to_csv('data/csv/json_df', sep=';', index=False)

