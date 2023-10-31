"""
input:  df_cleaned (csv) from 3_preprocessing_normalizing_and_cleaning
        json_df (csv) from 2_preprocessing_json_loader
output: dataframes (train/test split) as input for LSTM models
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

base_path = "//Users/christian/Library/Mobile Documents/com~apple~CloudDocs/promotion/github_ml_ass/"
os.chdir(base_path)

sequence_length = 15
np.random.seed(42)
folds = 3
data_new = pd.DataFrame()

def person_split_column(person_split, data, folds):
    data_new = data
    for fold in range(folds):
        ps_fold = person_split[person_split['fold'] == fold]
        ps_fold_dropped = ps_fold.drop(['fold'], axis = 1)
        ps_fold_dropped.rename(columns={'split':'ps_fold_'+str(fold)}, inplace=True)
        data_new = pd.merge(data_new, ps_fold_dropped, left_on='Proband', right_on='person', how='inner')
    return data_new

def select_subset_man(df, subset):
    if subset == 'flap_jump':
        df_subset = df[(df["Flattern"] == 1) & (df["Fraglich"] == 0) |
                (df["Hüpfen"] == 1) & (df["Fraglich"] == 0)]
    else:
        print('please select subset')
    print('selected subset: '+subset)
    return df_subset

def select_subset_no_man(df):
    df_subset = df[(df["Manierismus"] == 0)]
    return df_subset

def balance_subset_no_man(subset_man, subset_no_man):
    number_man_sequences = subset_man.groupby(["video_id"]).sum().shape[0]
    df_video_id = subset_no_man.groupby(["video_id"], as_index=False).first()
    df_video_id_random = df_video_id.sample(n=number_man_sequences, random_state=42)
    video_id_random = df_video_id_random['video_id']
    df_subset = subset_no_man[(subset_no_man["video_id"].isin(video_id_random))]
    return df_subset

def shuffle_sequences(a, sequence_length):
    b = a.set_index(np.arange(len(a)) // sequence_length, append=True).swaplevel(0, 1)
    return pd.concat([b.xs(i) for i in np.random.permutation(range(len(a) // sequence_length))])

def concat_shuffle(df_1, df_2):
    df = concat_df(df_1, df_2)
    return shuffle_sequences(df,sequence_length)

def concat_df(df_1, df_2):
    return pd.concat([df_1, df_2], axis=0)

def kfold_split(df, n_splits=3):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    trains = []
    tests = []
    df_grouped = df.groupby(['video_id'], as_index=False).max()
    for train, test in kfold.split(df_grouped):
        train_inds = df_grouped.iloc[train]['video_id']
        test_inds = df_grouped.iloc[test]['video_id']
        train = df[(df['video_id'].isin(train_inds))]
        test = df[(df['video_id'].isin(test_inds))]
        trains.append(train)
        tests.append(test)
    return trains, tests

def create_sets_RS_cv(data_new, subset):
    train_sets_RS = []
    test_sets_RS = []
    subset_man = select_subset_man(data_new, subset)
    subset_no_man = select_subset_no_man(data_new)
    subset_no_man_balanced = balance_subset_no_man(subset_man, subset_no_man)
    trains_man, tests_man = kfold_split(subset_man)
    trains_no_man, tests_no_man = kfold_split(subset_no_man_balanced)
    for x,y in zip(trains_man,trains_no_man):
        z = concat_shuffle(x, y)
        train_sets_RS.append(z)
    for x,y in zip(tests_man,tests_no_man):
        z = concat_shuffle(x, y)
        test_sets_RS.append(z)
    for i, (x,y) in enumerate(zip(train_sets_RS,test_sets_RS)):
        print('fold: ' + str(i) + ' train_man_sequences: ' + str(x['Manierismus'].sum()/x['Manierismus'].size*100))
        print('fold: ' + str(i) + ' test_man_sequences: ' + str(y['Manierismus'].sum()/y['Manierismus'].size*100))
    return train_sets_RS, test_sets_RS

def select_subset_person_split(df, column_name, set_no):
    df_subset = df[(df[column_name] == set_no)]
    return df_subset

def create_sets_PS_cv(data_new, subset, folds):
    train_sets_PS = []
    test_sets_PS = []
    for fold in range(folds):
        train_set_PS_unbalanced = select_subset_person_split(data_new, 'ps_fold_'+str(fold), 0)
        test_set_PS = select_subset_person_split(data_new, 'ps_fold_'+str(fold), 1)
        train_set_PS_unbalanced_man = select_subset_man(train_set_PS_unbalanced, subset)
        test_set_PS_man = select_subset_man(test_set_PS, subset)
        train_set_PS_unbalanced_no_man = select_subset_no_man(train_set_PS_unbalanced)
        test_set_PS_no_man = select_subset_no_man(test_set_PS)
        train_set_PS_balanced_no_man = balance_subset_no_man(train_set_PS_unbalanced_man, train_set_PS_unbalanced_no_man)
        test_set_PS_balanced_no_man = balance_subset_no_man(test_set_PS_man, test_set_PS_no_man)
        train_set_PS = concat_shuffle(train_set_PS_unbalanced_man, train_set_PS_balanced_no_man)
        test_set_PS_new = concat_shuffle(test_set_PS_man, test_set_PS_balanced_no_man)
        train_sets_PS.append(train_set_PS)
        test_sets_PS.append(test_set_PS_new)
    for i, (x,y) in enumerate(zip(train_sets_PS,test_sets_PS)):
        print('fold: ' + str(i) + ' train_man_sequences: ' + str(x['Manierismus'].sum()/x['Manierismus'].size*100))
        print('fold: ' + str(i) + ' test_man_sequences: ' + str(y['Manierismus'].sum()/y['Manierismus'].size*100))
    return train_sets_PS, test_sets_PS

#load data
data_folder = os.path.join(base_path, "data/csv")
data = pd.read_csv(os.path.join(data_folder, "df_cleaned.csv"), sep=';')
# train test splits
subsets = ['flap_jump']
for subset in subsets:
    person_split = pd.read_csv(os.path.join(data_folder, "person_split_"+subset+".csv"), sep=';')
    data_new = person_split_column(person_split, data, folds)
    train_sets_RS, test_sets_RS = create_sets_RS_cv(data_new, subset)
    train_sets_PS, test_sets_PS = create_sets_PS_cv(data_new, subset, folds)
    # save data
    for i, (train, test) in enumerate(zip(train_sets_RS,test_sets_RS)):
        train.to_csv(os.path.join(data_folder, subset+'_train_set_RS_cv_'+str(i)), sep=';', index=False)
        test.to_csv(os.path.join(data_folder, subset+'_test_set_RS_cv_'+str(i)), sep=';', index=False)
    for i, (train, test) in enumerate(zip(train_sets_PS,test_sets_PS)):
        train.to_csv(os.path.join(data_folder, subset+'_train_set_PS_cv_'+str(i)), sep=';', index=False)
        test.to_csv(os.path.join(data_folder, subset+'_test_set_PS_cv_'+str(i)), sep=';', index=False)
