import pandas as pd
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedGroupKFold
from keras_tuner import Hyperband
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, multilabel_confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from statistics import mean

# Set random seed for reproducibility
random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)
keras.utils.set_random_seed(random_seed)

# Function to load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
        return df
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None

# File path and data loading
file_name = 'df_cleaned.csv'
data = load_data(file_name)

if data is not None:
    print("Data loaded successfully.")
    print(f"\nData Shape: {data.shape}")
else:
    print("Data loading failed.")

# Define sequence length, labels, features, and group
sequence_length = 15
column_names_labels = ['Flattern', 'Hüpfen']  # Multi-label classification targets
column_names_features = ['Shoulder_x_0', 'Shoulder_y_0', 'Elbow_x_0', 'Elbow_y_0',
                    'RWrist_x_0', 'RWrist_y_0', 'LShoulder_x_0', 'LShoulder_y_0',
                    'LElbow_x_0', 'LElbow_y_0', 'LWrist_x_0', 'LWrist_y_0',
                    'Shoulder_x_1', 'Shoulder_y_1', 'Elbow_x_1', 'Elbow_y_1',
                    'RWrist_x_1', 'RWrist_y_1', 'LShoulder_x_1', 'LShoulder_y_1',
                    'LElbow_x_1', 'LElbow_y_1', 'LWrist_x_1', 'LWrist_y_1',
                    'Shoulder_x_2', 'Shoulder_y_2', 'Elbow_x_2', 'Elbow_y_2',
                    'RWrist_x_2', 'RWrist_y_2', 'LShoulder_x_2', 'LShoulder_y_2',
                    'LElbow_x_2', 'LElbow_y_2', 'LWrist_x_2', 'LWrist_y_2',
                    'Shoulder_x_3', 'Shoulder_y_3', 'Elbow_x_3', 'Elbow_y_3',
                    'RWrist_x_3','RWrist_y_3', 'LShoulder_x_3', 'LShoulder_y_3',
                    'LElbow_x_3', 'LElbow_y_3', 'LWrist_x_3', 'LWrist_y_3',
                    'Shoulder_x_4', 'Shoulder_y_4', 'Elbow_x_4', 'Elbow_y_4',
                    'RWrist_x_4', 'RWrist_y_4', 'LShoulder_x_4', 'LShoulder_y_4',
                    'LElbow_x_4', 'LElbow_y_4', 'LWrist_x_4', 'LWrist_y_4']
group_column = 'Proband' # 'Proband' for subject-wise split and 'video_id' for random split

def print_label_distribution(y_data, groups_data, sequence_length, fold_type):
    # Count occurrences of each label combination
    no_man = np.sum(np.all(y_data == [0, 0], axis=1))
    jump = np.sum(np.all(y_data == [0, 1], axis=1))
    flap = np.sum(np.all(y_data == [1, 0], axis=1))
    flap_jump = np.sum(np.all(y_data == [1, 1], axis=1))
    group_count = len(np.unique(groups_data))
    # Print label distribution
    print(f"\nLabel Distribution in {fold_type} Set:")
    print(f"  Train labels: no_man={no_man/sequence_length}, jump={jump/sequence_length}, flap={flap/sequence_length}, flap_jump={flap_jump/sequence_length}")
    if group_column == 'Proband':
      print(f"  Total samples: {len(y_data)/sequence_length}, children: {group_count}")
    else:
      print(f"  Total samples: {len(y_data)/sequence_length}, videos: {group_count}")

# Separate features and labels in full dataset
X_data = data[column_names_features].values
y_data = data[column_names_labels].values
if group_column == 'Proband':
  groups_data = data['Proband'].values
else:
  groups_data = data['video_id'].values
print(f"\nX_data Shape: {X_data.shape}")
print(f"y_data Shape: {y_data.shape}")
print(f"groups_data Shape: {groups_data.shape}")
print_label_distribution(y_data, groups_data, sequence_length, 'y_data')

# Select subset
def select_subset(df):
    df_subset = df[(df["Flattern"] == 1) & (df["Fraglich"] == 0) |
            (df["Hüpfen"] == 1) & (df["Fraglich"] == 0) |
            (df["Manierismus"] == 0)]
    return df_subset
data_subset = select_subset(data)

# Separate features and labels in subset
X = data_subset[column_names_features].values
y = data_subset[column_names_labels].values
if group_column == 'Proband':
  groups = data_subset['Proband'].values
else:
  groups = data_subset['video_id'].values
print(f"\nX_data Shape: {X.shape}")
print(f"y Shape: {y.shape}")
print(f"groups Shape: {groups.shape}")
print_label_distribution(y, groups, sequence_length, 'y')

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Undersampling
def undersample(X, y, groups, sequence_length=15):
    n_sequences = X.shape[0] // sequence_length
    X_reshaped = X.reshape(n_sequences, sequence_length, -1)
    y_reshaped = y.reshape(n_sequences, sequence_length, -1)
    groups_reshaped = groups[::sequence_length]
    no_man_indices = np.where(np.all(y_reshaped[:, 0] == [0, 0], axis=1))[0]
    jump_indices = np.where(np.all(y_reshaped[:, 0] == [0, 1], axis=1))[0]
    flap_indices = np.where(np.all(y_reshaped[:, 0] == [1, 0], axis=1))[0]
    flap_jump_indices = np.where(np.all(y_reshaped[:, 0] == [1, 1], axis=1))[0]
    max_no_man = len(jump_indices) + len(flap_indices) + len(flap_jump_indices)
    if len(no_man_indices) > max_no_man:
        undersampled_no_man_indices = np.random.choice(no_man_indices, size=max_no_man, replace=False)
    else:
        undersampled_no_man_indices = no_man_indices
    undersampled_indices = np.concatenate([undersampled_no_man_indices, jump_indices, flap_indices, flap_jump_indices])
    X_undersampled = X_reshaped[undersampled_indices]
    y_undersampled = y_reshaped[undersampled_indices]
    groups_undersampled = groups_reshaped[undersampled_indices]
    groups_undersampled_rows = np.repeat(groups_undersampled, sequence_length)
    return X_undersampled.reshape(-1, X.shape[-1]), y_undersampled.reshape(-1, y.shape[-1]), groups_undersampled_rows
X_undersampled, y_undersampled, groups_undersampled = undersample(X, y, groups, sequence_length)
print(f"\nX_undersampled Shape: {X_undersampled.shape}")
print(f"y_undersampled Shape: {y_undersampled.shape}")
print(f"groups_undersampled Shape: {groups_undersampled.shape}")
print_label_distribution(y_undersampled, groups_undersampled, sequence_length, 'y_undersampled')

# Check group overlap
def check_group_overlap(groups_train, groups_test):
    common_groups = np.intersect1d(groups_train, groups_test)
    if len(common_groups) > 0:
        print(f"Overlap found in groups: {common_groups}")
    else:
        print("No overlap in groups between training and test sets.")

# Reshape the data
num_sequences = X_undersampled.shape[0] // sequence_length
X_sequences = X_undersampled.reshape((num_sequences, sequence_length, -1))
y_sequences = y_undersampled.reshape((num_sequences, sequence_length, -1))[:, 0, :]
groups_sequences = groups_undersampled[::sequence_length]

def build_model(hp, input_shape=None):
    if input_shape is None:
        input_shape = (sequence_length, X_train.shape[-1])
    units_layer_1 = hp.Int('units_layer_1', min_value=128, max_value=512, step=128)
    units_layer_2 = hp.Int('units_layer_2', min_value=128, max_value=512, step=128)
    units_layer_3 = hp.Int('units_layer_3', min_value=128, max_value=512, step=128)
    num_layers = hp.Int('num_layers', min_value=3, max_value=3, step=1)
    dropout_rate = hp.Float('dropout_rate', min_value=0.00, max_value=0.15, step=0.05)
    learning_rate = hp.Choice('learning_rate', values=[0.0003])
    batch_size = hp.Int('batch_size', min_value=16, max_value=32, step=16)
    epochs = hp.Int('epochs', min_value=100, max_value=100, step=10)
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i in range(num_layers):
        units = units_layer_1 if i == 0 else units_layer_2 if i == 1 else units_layer_3
        model.add(LSTM(units=units, return_sequences=True if i < num_layers - 1 else False))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=2, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def build_final_model(input_shape, num_layers, units_layer_1, units_layer_2=None, units_layer_3=None,
                      dropout_rate=0.1, learning_rate=0.0003):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i in range(num_layers):
        units = units_layer_1 if i == 0 else units_layer_2 if i == 1 else units_layer_3
        model.add(LSTM(units=units, return_sequences=True if i < num_layers - 1 else False))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=2, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

# Outer loop: Nested cross-validation with StratifiedGroupKFold
outer_msgkf = StratifiedGroupKFold(n_splits=3)
# Combine both labels into a single integer for stratification
combined_y_sequences = y_sequences[:, 0] * 2 + y_sequences[:, 1]  # '00' -> 0, '01' -> 1, '10' -> 2, '11' -> 3
best_hyperparams = {
    'num_layers': [],
    'units_layer_1': [],
    'units_layer_2': [],
    'units_layer_3': [],
    'dropout_rate': [],
    'learning_rate': [],
    'batch_size': [],
    'epochs': []
}
results_cv = pd.DataFrame(columns=['measure', 'value'])
results_list = []
y_preds_all_folds = []
y_tests_all_folds = []
y_trains_all_folds = []

for fold, (train_idx, test_idx) in enumerate(outer_msgkf.split(X_sequences, combined_y_sequences, groups_sequences)):
    X_train, X_test = X_sequences[train_idx], X_sequences[test_idx]
    y_train, y_test = y_sequences[train_idx], y_sequences[test_idx]
    groups_train, groups_test = groups_sequences[train_idx], groups_sequences[test_idx]
    print(f"Fold {fold+1}:")
    check_group_overlap(groups_train, groups_test)
    print_label_distribution(y_train, groups_train, 1, 'y_train')
    print_label_distribution(y_test, groups_test, 1, 'y_test')
    y_trains_all_folds.append(y_train)
    # Inner loop: Nested cross-validation with StratifiedGroupKFold
    inner_msgkf = StratifiedGroupKFold(n_splits=3)
    tuner = Hyperband(
        build_model,
        objective='val_binary_accuracy',
        max_epochs=1000,
        hyperband_iterations=2,
        directory='tuner_data',
        project_name=f'fold_{fold+1}',
        seed=random_seed
    )
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_msgkf.split(X_train, combined_y_sequences[train_idx], groups_train)):
        X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
        y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
        # Start hyperparameter tuning
        tuner.search(X_inner_train, y_inner_train, epochs=200, validation_data=(X_inner_val, y_inner_val))
    best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters for Outer Fold {fold+1}: {best_params.values}")
    for param in best_hyperparams.keys():
        best_hyperparams[param].append(best_params.get(param))
    # Build and train the model with the best hyperparameters
    tuned_model = build_final_model(
        input_shape=(sequence_length, X_train.shape[-1]),
        num_layers=best_params['num_layers'],
        units_layer_1=best_params['units_layer_1'],
        units_layer_2=best_params['units_layer_2'],
        units_layer_3=best_params.get('units_layer_3'),
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    es = EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=0, patience=100)
    model_save_name = f'best_model_fold_{fold+1}.keras'
    mc = ModelCheckpoint(model_save_name, monitor='val_binary_accuracy', mode='max', verbose=0, save_best_only=True)
    history = tuned_model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=best_params['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[es, mc],
        verbose=0
    )
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f'history_fold_{fold+1}.csv', sep=';', index=False)
    # Load best model and make predictions, print and save
    best_model = load_model(model_save_name)
    y_pred = best_model.predict(X_test)
    y_pred_thresholded = (y_pred > 0.5).astype(int)
    y_preds_all_folds.append(y_pred)
    y_tests_all_folds.append(y_test)
    print(f"Classification Report for Fold {fold+1}:")
    print(classification_report(y_test, y_pred_thresholded, target_names=["flap", "jump"], zero_division=0))
    results_new_model = {}
    yhat_flap = y_pred_thresholded[:, 0].ravel()
    yhat_jump = y_pred_thresholded[:, 1].ravel()
    y_test_flap = y_test[:, 0].ravel()
    y_test_jump = y_test[:, 1].ravel()
    accuracy_flap = accuracy_score(y_test_flap, yhat_flap)
    precision_flap = precision_score(y_test_flap, yhat_flap, zero_division=0)
    recall_flap = recall_score(y_test_flap, yhat_flap, zero_division=0)
    f1_flap = f1_score(y_test_flap, yhat_flap, zero_division=0)
    specificity_flap = recall_score(y_test_flap, yhat_flap, pos_label=0)
    accuracy_jump = accuracy_score(y_test_jump, yhat_jump)
    precision_jump = precision_score(y_test_jump, yhat_jump, zero_division=0)
    recall_jump = recall_score(y_test_jump, yhat_jump, zero_division=0)
    f1_jump = f1_score(y_test_jump, yhat_jump, zero_division=0)
    specificity_jump = recall_score(y_test_jump, yhat_jump, pos_label=0)
    results_new_model['accuracy_flap'] = accuracy_flap
    results_new_model['precision_flap'] = precision_flap
    results_new_model['recall_flap'] = recall_flap
    results_new_model['f1_flap'] = f1_flap
    results_new_model['specificity_flap'] = specificity_flap
    results_new_model['accuracy_jump'] = accuracy_jump
    results_new_model['precision_jump'] = precision_jump
    results_new_model['recall_jump'] = recall_jump
    results_new_model['f1_jump'] = f1_jump
    results_new_model['specificity_jump'] = specificity_jump
    results_new_model_df = pd.DataFrame(list(results_new_model.items()), columns=['measure', 'value'])
    results_new_model_df.to_csv(f'results_new_model_fold_{fold+1}.csv', sep=';', index=False)
    results_list.append(results_new_model_df)
    print(f"\nModel Results Fold {fold+1}:")
    for measure, value in results_new_model.items():
        print(f"{measure}: {value}")
    cm = multilabel_confusion_matrix(y_test, y_pred_thresholded)
    cm_flap = cm[0]
    pd.DataFrame(cm_flap).to_csv(f'confusion_matrix_flap_fold_{fold+1}.csv', sep=';', index=False)
    cm_jump = cm[1]
    pd.DataFrame(cm_jump).to_csv(f'confusion_matrix_jump_fold_{fold+1}.csv', sep=';', index=False)
    report = classification_report(y_test, y_pred_thresholded, target_names=['flap', 'jump'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'classification_report_fold_{fold+1}.csv', sep=';', index=False)
    print("="*50)

# Aggregate results
data = pd.concat(results_list, ignore_index=True)
data_grouped = data.groupby('measure')['value'].mean()
print(data)
print(data_grouped)
np.save('y_preds_all_folds.npy', np.array(y_preds_all_folds, dtype=object))
np.save('y_tests_all_folds.npy', np.array(y_tests_all_folds, dtype=object))
np.save('y_trains_all_folds.npy', np.array(y_trains_all_folds, dtype=object))
print("Best Hyperparameters per Fold:")
n_folds = len(next(iter(best_hyperparams.values())))
for fold in range(n_folds):
    fold_params = {param: best_hyperparams[param][fold] for param in best_hyperparams.keys()}
    print(f"Fold {fold + 1}: {fold_params}")
final_hyperparams = {param: mean(values) for param, values in best_hyperparams.items()}
print("Final Hyperparameters (mean across folds):")
print(final_hyperparams)
# Final model: StratifiedGroupKFold
final_msgkf = StratifiedGroupKFold(n_splits=2)
train_idx, val_idx = next(final_msgkf.split(X_sequences, combined_y_sequences, groups_sequences))
X_train_final, X_val_final = X_sequences[train_idx], X_sequences[val_idx]
y_train_final, y_val_final = y_sequences[train_idx], y_sequences[val_idx]
# Build and train the final model with mean hyperparameters
final_model = build_final_model(
    input_shape=(sequence_length, X_sequences.shape[-1]),
    num_layers=int(round(final_hyperparams['num_layers'])),
    units_layer_1=int(round(final_hyperparams['units_layer_1'])),
    units_layer_2=int(round(final_hyperparams['units_layer_2'])),
    units_layer_3=int(round(final_hyperparams.get('units_layer_3', 0))),
    dropout_rate=final_hyperparams['dropout_rate'],
    learning_rate=final_hyperparams['learning_rate']
)
es_final = EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=100)
final_model_save_name = 'final_model.keras'
mc_final = ModelCheckpoint(final_model_save_name, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
history_final = final_model.fit(
    X_train_final, y_train_final,
    epochs=1000,
    batch_size=int(round(final_hyperparams['batch_size'])),
    validation_data=(X_val_final, y_val_final),
    callbacks=[es_final, mc_final],
    verbose=2
)
# Load best model and make predictions, print and save
best_final_model = load_model(final_model_save_name)
y_pred_final = best_final_model.predict(X_val_final)
y_pred_thresholded_final = (y_pred_final > 0.5).astype(int)
np.save('y_preds_final_model.npy', y_pred_final)
np.save('y_tests_final_model.npy', y_val_final)
np.save('y_trains_final_model.npy', y_train_final)
print("Final model predictions, true validation labels, and training labels saved.")
print("Classification Report for Final Model:")
print(classification_report(y_val_final, y_pred_thresholded_final, target_names=["flap", "jump"], zero_division=0))
results_final_model = {}
yhat_flap_final = y_pred_thresholded_final[:, 0].ravel()
yhat_jump_final = y_pred_thresholded_final[:, 1].ravel()
y_val_flap = y_val_final[:, 0].ravel()
y_val_jump = y_val_final[:, 1].ravel()
accuracy_flap_final = accuracy_score(y_val_flap, yhat_flap_final)
precision_flap_final = precision_score(y_val_flap, yhat_flap_final, zero_division=0)
recall_flap_final = recall_score(y_val_flap, yhat_flap_final, zero_division=0)
f1_flap_final = f1_score(y_val_flap, yhat_flap_final, zero_division=0)
specificity_flap_final = recall_score(y_val_flap, yhat_flap_final, pos_label=0)
accuracy_jump_final = accuracy_score(y_val_jump, yhat_jump_final)
precision_jump_final = precision_score(y_val_jump, yhat_jump_final, zero_division=0)
recall_jump_final = recall_score(y_val_jump, yhat_jump_final, zero_division=0)
f1_jump_final = f1_score(y_val_jump, yhat_jump_final, zero_division=0)
specificity_jump_final = recall_score(y_val_jump, yhat_jump_final, pos_label=0)
results_final_model['accuracy_flap'] = accuracy_flap_final
results_final_model['precision_flap'] = precision_flap_final
results_final_model['recall_flap'] = recall_flap_final
results_final_model['f1_flap'] = f1_flap_final
results_final_model['specificity_flap'] = specificity_flap_final
results_final_model['accuracy_jump'] = accuracy_jump_final
results_final_model['precision_jump'] = precision_jump_final
results_final_model['recall_jump'] = recall_jump_final
results_final_model['f1_jump'] = f1_jump_final
results_final_model['specificity_jump'] = specificity_jump_final
results_final_model_df = pd.DataFrame(list(results_final_model.items()), columns=['measure', 'value'])
results_final_model_df.to_csv('results_final_model.csv', sep=';', index=False)
print("\nFinal Model Results:")
for measure, value in results_final_model.items():
    print(f"{measure}: {value}")
cm_final = multilabel_confusion_matrix(y_val_final, y_pred_thresholded_final)
cm_flap_final = cm_final[0]
pd.DataFrame(cm_flap_final).to_csv('confusion_matrix_flap_final.csv', sep=';', index=False)
cm_jump_final = cm_final[1]
pd.DataFrame(cm_jump_final).to_csv('confusion_matrix_jump_final.csv', sep=';', index=False)
report_final = classification_report(y_val_final, y_pred_thresholded_final, target_names=['flap', 'jump'], output_dict=True, zero_division=0)
report_final_df = pd.DataFrame(report_final).transpose()
report_final_df.to_csv('classification_report_final.csv', sep=';', index=False)
print("Final model training, metrics calculation, and saving complete.")