
# info
"""
input:  masterfile_sequences (csv) from 1_preprocessing_masterfiles
        json_df (csv) from 2_preprocessing_json_loader
output: df_cleaned (csv) with cleaned JSON-data (less body parts, high accuracy, centralized) 
"""

import os
import numpy as np
import pandas as pd
import itertools

base_path = "//Users/christian/Library/Mobile Documents/com~apple~CloudDocs/promotion/github_ml_ass/"
os.chdir(base_path)
json_path = "/Users/christian/promotion/data/JSON_OpenPose_ID_Auto"

sequence_length = 15
np.random.seed(42)
accuracy_threshold = 0.6
width = 1920
heigth = 1080
max_width_heigth = max(width, heigth) 
max_z = 0

body_parts_drop = ["Nose", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", 
                   "RBigToe", "RSmallToe", "RHeel", "Neck", "MidHip", "RHip", "RKnee", "RAnkle", 
                   "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", 
                   "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]

body_parts_arms = ["Shoulder", "Elbow", "RWrist", "LShoulder", "LElbow", "LWrist"]

column_names_drop = []
for prefix in ['person_id', 'filename', 'Flattern', 'Klatschen', 'Hüpfen', 'Manierismus', 'Fraglich', 'Hand',
               'Bein', 'Zehenspitzengang', 'Körper', 'Selbstverletzung', 'still', 'schnell', 'Proband', 
               'video_id', 'Dateiname', 'Sequenz_Nr', 'Video', 'clap_jump', 'clap_only', 'filename_short',
               'flap_clap', 'flap_clap_jump', 'flap_jump', 'flap_only', 'frame_no', 'hand_only', 'jump_only', 
               'nothing', 'other']:
    for i in range(5):
        column_name = prefix + str('_') + str(i)
        column_names_drop.append(column_name)

column_names_labels = ['Flattern','Klatschen','Hüpfen','Manierismus']
value_names = ["x", "y"]
column_names_arms = [n+"_"+m for n,m in list(itertools.product(body_parts_arms, value_names))]

df_arms_high_acc = pd.DataFrame()
df_centered_short = pd.DataFrame()
df_persons = []

def drop_body_parts(df, body_parts_drop):
    df = df.loc[:,~df.columns.str.startswith(tuple(body_parts_drop))]
    return df   

def accuracy_cleaning(df, threshold=accuracy_threshold):
    df = df[(df['Shoulder_acc'] >= threshold) & (df['Elbow_acc'] >= threshold)
            & (df['RWrist_acc'] >= threshold) & (df['LShoulder_acc'] >= threshold)
            & (df['LElbow_acc'] >= threshold) & (df['LWrist_acc'] >= threshold)]
    return df   
      
def centralize(z, max_z, min_z=0, max_x_y=max_width_heigth):
    middle = (max_z + min_z) / 2
    scale = max_width_heigth
    z_new = z - middle
    z_new_scaled = z_new / scale
    z_new_translated = z_new_scaled + 0.5
    return z_new_translated

def apply_centralize_x(df_arms_high_acc, max_z):
    df_arms_high_acc_centered = df_arms_high_acc.copy()
    df_arms_high_acc_centered['Shoulder_x'] = df_arms_high_acc_centered['Shoulder_x'].apply(centralize, max_z = width)
    df_arms_high_acc_centered['Elbow_x'] = df_arms_high_acc_centered['Elbow_x'].apply(centralize, max_z = width)
    df_arms_high_acc_centered['RWrist_x'] = df_arms_high_acc_centered['RWrist_x'].apply(centralize, max_z = width)
    df_arms_high_acc_centered['LShoulder_x'] = df_arms_high_acc_centered['LShoulder_x'].apply(centralize, max_z = width)
    df_arms_high_acc_centered['LElbow_x'] = df_arms_high_acc_centered['LElbow_x'].apply(centralize, max_z = width)
    df_arms_high_acc_centered['LWrist_x'] = df_arms_high_acc_centered['LWrist_x'].apply(centralize, max_z = width)
    return df_arms_high_acc_centered

def apply_centralize_y(df_arms_high_acc, max_z):
    df_arms_high_acc_centered = df_arms_high_acc.copy()
    df_arms_high_acc_centered['Shoulder_y'] = df_arms_high_acc_centered['Shoulder_y'].apply(centralize, max_z = heigth)
    df_arms_high_acc_centered['Elbow_y'] = df_arms_high_acc_centered['Elbow_y'].apply(centralize, max_z = heigth)
    df_arms_high_acc_centered['RWrist_y'] = df_arms_high_acc_centered['RWrist_y'].apply(centralize, max_z = heigth)
    df_arms_high_acc_centered['LShoulder_y'] = df_arms_high_acc_centered['LShoulder_y'].apply(centralize, max_z = heigth)
    df_arms_high_acc_centered['LElbow_y'] = df_arms_high_acc_centered['LElbow_y'].apply(centralize, max_z = heigth)
    df_arms_high_acc_centered['LWrist_y'] = df_arms_high_acc_centered['LWrist_y'].apply(centralize, max_z = heigth)
    return df_arms_high_acc_centered

def separate_persons(df_centered_short):
    df_persons = []
    for person_id, df_person in df_centered_short.groupby('person_id'):
        df_persons.append(df_person)
    df_persons = df_persons[:5]
    return df_persons

def keep_sequences(df_persons, sequence_length):
    df_persons_full = []
    for i, df in enumerate(df_persons):
        video_ids = df.groupby(['video_id'], as_index=False).size()
        video_ids_full_sequence = video_ids[video_ids["size"] == sequence_length]
        video_ids_full = video_ids_full_sequence['video_id']
        video_ids_full = video_ids_full.to_frame()
        video_ids_full.columns = ['video_id_full']
        df_person_full = pd.merge(video_ids_full, df, left_on='video_id_full', 
                        right_on='video_id', how='left').drop('video_id_full', axis=1)
        df_person_full['id_filename'] = df_person_full['video_id'].astype(str)+'_'+df_person_full['filename'].astype(str)
        df_person_full = df_person_full.set_index('id_filename')
        df_person_full = df_person_full.add_suffix('_'+str(i))
        df_persons_full.append(df_person_full)
    json_df_new = pd.concat(df_persons_full, join='outer', axis=1).fillna(0)
    json_df_new.sort_values(by=['id_filename'], inplace=True)
    return json_df_new

def clean_data(df_full, column_names_drop):
    columns = ['video_id', 'Proband', 'filename', 'filename_short', 'Dateiname', 'frame_no', 'Sequenz_Nr', 'Video']
    for col in columns:
        df_full[col] = 0    
        for i in range(5):
            condition = df_full[f'{col}_{i}'] == 0
            df_full[col] = np.where(condition, df_full[col], df_full[f'{col}_{i}'])
    for prefix in ['Flattern', 'Klatschen', 'Hüpfen', 'Manierismus', 'Fraglich', 'Hand', 'Bein', 'Zehenspitzengang', 'Körper', 
                   'Selbstverletzung', 'still', 'schnell', 'clap_jump', 'clap_only', 'flap_clap', 'flap_clap_jump', 'flap_jump', 
                   'flap_only', 'hand_only', 'jump_only', 'nothing', 'other']:
        columns = [f'{prefix}_{i}' for i in range(5)]
        if columns[0] in df_full.columns:
            df_full[prefix] = df_full[columns].max(axis=1)
    df_full.drop(column_names_drop, axis=1, inplace=True)
    return df_full

#load data
data_folder = os.path.join(base_path, "data/csv")
masterfile_sequences = pd.read_csv(os.path.join(data_folder, "masterfile_sequences_15.csv"), sep=';')
json_df = pd.read_csv(os.path.join(data_folder, "json_df.csv"), sep=';')
# merge with sequences
df_merged = pd.merge(json_df, masterfile_sequences, left_on='filename', right_on='filename', how='inner')
# drop body parts  
df_arms = drop_body_parts(df_merged, body_parts_drop)
# drop accuracy below threshold
df_arms_high_acc = accuracy_cleaning(df_arms) 
# centralize x and y values
df_arms_centered_x = apply_centralize_x(df_arms_high_acc, width)
df_arms_centered = apply_centralize_y(df_arms_centered_x, heigth)
# drop accuracy columns
df_centered_short = df_arms_centered.loc[:,~df_arms_centered.columns.str.endswith('acc')] # drop accuracy
# create list with dataframes per person
df_persons = separate_persons(df_centered_short)
# keep only full sequences with sequence_length and merge into one dataframe
df_full = keep_sequences(df_persons, sequence_length)
# clean dataframe (remove redundant columns)
df_cleaned = clean_data(df_full, column_names_drop)

# save data
data_folder = os.path.join(base_path, "data/csv")
df_cleaned.to_csv(os.path.join(data_folder, "df_cleaned.csv"), sep=';', index=False)
