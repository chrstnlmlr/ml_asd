import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)
keras.utils.set_random_seed(random_seed)

# load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
        return df
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None

file_name = 'df_cleaned.csv'
data = load_data(file_name)

if data is not None:
    print("Data loaded successfully.")
    print(f"\nData Shape: {data.shape}")
else:
    print("Data loading failed.")

# prediction

# load final model
model_path = 'final_model.keras'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except OSError:
    print(f"Error: Model file not found at {model_path}")
    exit()

# select sequence
video_ids_to_predict = [76913, 76914, 76915, 76916, 76917, 76918, 76919, 76920, 76921, 76922, 
                        76923, 76924, 76925, 76926, 76927, 76928, 76929, 76930, 76931, 76932, 
                        76933, 76934, 76935, 76936, 76937, 76938, 76939]
features_to_use = ['Shoulder_x_0', 'Shoulder_y_0', 'Elbow_x_0', 'Elbow_y_0',
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

predictions_data = []

for video_id_to_predict in video_ids_to_predict:
    sequence_to_predict = data[data['video_id'] == video_id_to_predict]
    if sequence_to_predict.empty:
        print(f"Error: No sequence found for video_id '{video_id_to_predict}'")
        continue

    X_pred = sequence_to_predict[features_to_use].values

    # scaling
    scaler = StandardScaler()
    X_pred = scaler.fit_transform(X_pred)

    # reshape for LSTM input
    X_pred = X_pred.reshape(1, X_pred.shape[0], X_pred.shape[1])

    # prediction
    predictions = model.predict(X_pred)
    print(f"Predictions for video_id {video_id_to_predict}: {predictions}")


    # convert probabilities to class labels
    prediction = predictions[0]
    threshold = 0.5
    label = ""
    if prediction[0] > threshold:
        label += "likely flapping"
    if prediction[1] > threshold:
        if len(label) > 0:
            label += "+"
        label += "likely jumping"
    if len(label) == 0:
        label = "nothing"

    predictions_data.append({'video_id': video_id_to_predict, 'prediction': label})

# Save predictions
predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv('predictions.csv', index=False, sep = ';')
print("Predictions saved to predictions.csv")