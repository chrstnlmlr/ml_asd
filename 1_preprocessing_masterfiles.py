# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd

########## INFO ##########
"""
input: OpenPose JSON filepath and Excel files: "Klassifizierungen"
output: csv files: masterfiles, masterfile_sequences
"""
########## SET PATHS ##########

# change working directory
os.chdir("//Users/christian/Library/Mobile Documents/com~apple~CloudDocs/promotion/github_ml_ass/")
#print(os.getcwd())
# path to Open Pose JSON output
json_path = "/Users/christian/promotion/data/JSON_OpenPose"
# path to classification excel file
classification_filepath = "data/excel/BOSCC_Klassifizierungen.xlsm"

########## SET PARAMETERS ##########

sequence_length = 15

########## FUNCTIONS ##########

def create_masterfile(json_path, classification_filepath):
    """
    merge JSON-files with classifications (one row per frame)
    """
    # import JSON filenames
    json_files = [file for file in os.listdir(json_path)]
    # shorten filenames
    json_files_short = [x[:-28] for x in json_files]
    # dtype string for merge
    json_df = pd.DataFrame({'filename': json_files, 'filename_short': json_files_short}, dtype= "string")
    # sort df by filename
    json_df.sort_values(by=['filename'], inplace=True)
    # open classification file
    classification_file = pd.read_excel(classification_filepath, sheet_name="class_table")
    # dtype string for merge
    classification_file['Dateiname_kurz']= classification_file.Dateiname_kurz.astype('string')
    # merge JSON filenames with classifications
    masterfile = pd.merge(json_df, classification_file,
                                    left_on='filename_short', right_on='Dateiname_kurz',
                                    how='inner').drop('Dateiname_kurz', axis=1)
    return masterfile

def add_frame(masterfile):
    """
    add frame-number to dataframe
    """
    masterfile_new = masterfile.copy()
    # sort df by filename
    masterfile_new.sort_values(by=['filename'], inplace=True)
    masterfile_new['frame_no'] = masterfile_new.groupby(['Video']).cumcount()
    return masterfile_new

def add_sequence(masterfile_frame, sequence_length=sequence_length):
    """
    generate sequence-number for every sequence_length
    drop leftover frames
    """
    video_id = 0
    counter = 0
    printcounter = 0
    json_df = masterfile_frame.copy()
    # add column video_id
    json_df['video_id'] = ""
    # copy dataframe with needed column names
    json_df_new = json_df[0:0].copy()
    for i in range(0, len(json_df)):
        if printcounter == 10000:
            print(i)
            print (json_df.iloc[i]['filename'])
            print(video_id)
            printcounter = 0
        if '000000000000_keypoints' in json_df.iloc[i]['filename']:
            counter = 0
        elif counter == sequence_length:
            json_df_new = json_df_new.append(json_df.iloc[i-sequence_length:i])
            json_df_new.iloc[-sequence_length:, json_df_new.columns.get_loc('video_id')] = video_id
            counter = 0
            video_id += 1
        counter += 1
        printcounter += 1
        #if video_id == 10:
        #    break
    return json_df_new

########## MAIN ##########

# create masterfile
masterfile = create_masterfile(json_path, classification_filepath)
# add frame_no
masterfile_frame = add_frame(masterfile)
# create masterfile sequences
masterfile_sequences = add_sequence(masterfile_frame, sequence_length)

########## SAVE DATA ##########

# save masterfiles
masterfile.to_csv('data/csv/masterfile', sep=';', index=False)
masterfile_sequences.to_csv('data/csv/masterfile_sequences_'+str(sequence_length), sep=';', index=False)
