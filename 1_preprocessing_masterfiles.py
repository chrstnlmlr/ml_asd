
# info
"""
input: OpenPose output (JSON) and classifications (xlsx)
output: masterfiles (csv) masterfile_sequences (csv)
"""

import os
import pandas as pd
import numpy as np

base_path = "//Users/christian/Library/Mobile Documents/com~apple~CloudDocs/promotion/github_ml_ass/"
json_path = "/Users/christian/promotion/data/JSON_OpenPose_ID_Auto"
classification_filepath = os.path.join(base_path, "data/coding/BOSCC_coding.xlsm")

sequence_length = 15
np.random.seed(42)

def create_masterfile(json_path, classification_filepath):
    json_files = [file for file in os.listdir(json_path)]
    json_files_short = [x[:-28] for x in json_files]
    json_df = pd.DataFrame({'filename': json_files, 'filename_short': json_files_short}, dtype="string")
    json_df.sort_values(by='filename', inplace=True)
    classification_file = pd.read_excel(classification_filepath, sheet_name="class_table")
    classification_file['Dateiname_kurz'] = classification_file['Dateiname_kurz'].astype('string')
    masterfile = pd.merge(json_df, classification_file, left_on='filename_short', right_on='Dateiname_kurz', how='inner').drop('Dateiname_kurz', axis=1)
    return masterfile

def add_frame(masterfile):
    masterfile_new = masterfile.copy()
    masterfile_new.sort_values(by='filename', inplace=True)
    masterfile_new['frame_no'] = masterfile_new.groupby(['Video']).cumcount()
    return masterfile_new

def add_sequence(masterfile_frame, sequence_length):
    video_id = 0
    counter = 0
    json_df = masterfile_frame.copy()
    json_df['video_id'] = ""
    json_df_new = json_df[0:0].copy()
    sequences = []
    for i in range(0, len(json_df)):
        if '000000000000_keypoints' in json_df.iloc[i]['filename']:
            counter = 0
        elif counter == sequence_length:
            sequence = json_df.iloc[i - sequence_length:i].copy()
            sequence.iloc[-sequence_length:, json_df.columns.get_loc('video_id')] = video_id
            sequences.append(sequence)
            counter = 0
            video_id += 1
        counter += 1
    json_df_new = pd.concat(sequences, ignore_index=True)
    return json_df_new

# create masterfile
masterfile = create_masterfile(json_path, classification_filepath)
# add frame numbers
masterfile_frame = add_frame(masterfile)
# create masterfile sequences in parallel
masterfile_sequences = add_sequence(masterfile_frame, sequence_length)
# save data
data_folder = os.path.join(base_path, "data/csv")
masterfile.to_csv(os.path.join(data_folder, "masterfile.csv"), sep=';', index=False)
masterfile_sequences.to_csv(os.path.join(data_folder, f"masterfile_sequences_{sequence_length}.csv"), sep=';', index=False)
