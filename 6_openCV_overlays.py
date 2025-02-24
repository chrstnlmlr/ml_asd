import cv2
import os
import pandas as pd
import json
import numpy as np

# load
data = pd.read_csv('path_to_df_cleaned.csv', sep=';')
masterfile_sequences = pd.read_csv('path_to_masterfile_sequences_15.csv', sep=';')
masterfile = pd.read_csv('path_to_masterfile.csv', sep=';')
predictions = pd.read_csv('path_to_predictions.csv', sep=';')

input_path = 'path_to_videos/'
output_path = input_path
json_path = input_path + 'json/'

selected_sequences = ['FRA_039_T2_BOSCC_SegA_Kam1_4', 'FRA_039_T2_BOSCC_SegA_Kam1_5']

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

POSE_PAIRS = [
    ["Neck", "MidHip"], ["Neck", "RShoulder"], ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"], ["MidHip", "RHip"], ["RHip", "RKnee"],
    ["RKnee", "RAnkle"], ["MidHip", "LHip"], ["LHip", "LKnee"],
    ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],
    ["RAnkle", "RBigToe"], ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"],
    ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"], ["LAnkle", "LHeel"]
]

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0), (255, 0, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)
]

def draw_skeleton(frame, keypoints, confidence_threshold=0.2):
    for pair, color in zip(POSE_PAIRS, COLORS):
        part_a = BODY_PARTS[pair[0]]
        part_b = BODY_PARTS[pair[1]]
        
        if part_a < len(keypoints) and part_b < len(keypoints):
            if keypoints[part_a][2] > confidence_threshold and keypoints[part_b][2] > confidence_threshold:
                pt1 = (int(keypoints[part_a][0]), int(keypoints[part_a][1]))
                pt2 = (int(keypoints[part_b][0]), int(keypoints[part_b][1]))
                cv2.line(frame, pt1, pt2, color, 5)
                cv2.circle(frame, pt1, 8, color, -1)
                cv2.circle(frame, pt2, 8, color, -1)
    return frame

def get_video_ids(masterfile_sequences, filename_short):
    return masterfile_sequences.loc[
        masterfile_sequences['filename_short'] == filename_short, 'video_id'
    ].unique()

def get_video_ids_and_filenames(masterfile_sequences, filename_short):
    return masterfile_sequences.loc[
        masterfile_sequences['filename_short'] == filename_short, 
        ['video_id', 'filename']
    ].drop_duplicates()

def get_filenames(masterfile, filename_short):
    return masterfile.loc[
        masterfile['filename_short'] == filename_short, 'filename'
    ].unique()

def merge_predictions_to_frames(masterfile, masterfile_sequences, predictions, filename_short):
    video_ids_filenames = get_video_ids_and_filenames(masterfile_sequences, filename_short)
    merged_df = video_ids_filenames.merge(predictions, on="video_id", how="left")
    merged_df = masterfile.reset_index().merge(merged_df, on="filename", how="inner")
    frame_labels = dict(zip(merged_df["index"], merged_df["prediction"]))
    return frame_labels
 
def load_openpose_keypoints(json_path):
    """Load OpenPose keypoints from a JSON file and return a list of detected persons."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        people = data.get("people", [])
        #print(f"Loaded {len(people)} people from {json_path}")  # Debugging line
        return people
    except Exception as e:
        print(f"Error loading OpenPose data: {e}")
        return []

def create_silhouette_mask(frame, keypoints_list, frame_number):
    """Create a silhouette mask over detected body keypoints."""
    
    mask = np.zeros_like(frame, dtype=np.uint8)

    for person in keypoints_list:
        keypoints_raw = person.get("pose_keypoints_2d", [])
        keypoints = np.array(keypoints_raw)
        keypoints = keypoints.reshape(-1, 3)
        valid_points = [(int(kp[0]), int(kp[1])) for kp in keypoints if kp[2] > 0.05]
        if len(valid_points) > 3:
            hull = cv2.convexHull(np.array(valid_points, dtype=np.int32))  # Ensure closed shape
            cv2.fillPoly(mask, [hull], (255, 255, 255))  # White mask
            kernel = np.ones((30, 30), np.uint8)  # Adjust expansion size 20, 20
            mask = cv2.dilate(mask, kernel, iterations=3)
        else:
            print("Zu wenige Punkte zum Zeichnen der Maske")
    if mask.sum() == 0:
        print(f"Warning: Leere Maske in Frame {frame_number}. Gültige Punkte: {valid_points}")

    return mask

def process_video(input_path, output_path, frame_labels, placeholder_path, json_folder):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    placeholder_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        json_path = os.path.join(json_folder, f"{sequence}_{frame_number - 1:012d}_keypoints.json")
        print(f"Processing Frame {frame_number}, JSON Path: {json_path}")
        keypoints_list = load_openpose_keypoints(json_path)
        
        mask = create_silhouette_mask(frame, keypoints_list, frame_number)
        frame_with_mask = cv2.addWeighted(frame, 1, mask, 1, 0)  # 0% transparency
        
        # Draw skeleton on the frame
        for person in keypoints_list:
            keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
            frame_with_mask = draw_skeleton(frame_with_mask, keypoints)
        
        # Save first clean frame as placeholder
        if not placeholder_saved:
            cv2.imwrite(placeholder_path, frame_with_mask)
            placeholder_saved = True
        
        # Add labels
        if frame_number in frame_labels and frame_labels[frame_number].lower() != "nothing":
            label = frame_labels[frame_number]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.0
            font_color = (0, 0, 255)
            thickness = 5
            margin = 50
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = width - text_size[0] - margin
            text_y = margin + text_size[1] + 50
            cv2.putText(frame_with_mask, label, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        out.write(frame_with_mask)  # Ensure masked frame with skeleton is saved

    cap.release()
    out.release()
    cv2.destroyAllWindows()  # Close 

# video ids for prediction
video_ids_list = []
for sequence in selected_sequences:
    video_ids = get_video_ids(masterfile_sequences, sequence)
    video_ids_list.extend(video_ids)
print(video_ids_list)

# Process selected videos
for sequence in selected_sequences:
    filenames = get_filenames(masterfile, sequence)
    frame_labels_pred = merge_predictions_to_frames(masterfile, masterfile_sequences, predictions, sequence)
    frame_labels = {i - min(frame_labels_pred.keys()): v for i, v in frame_labels_pred.items()}
    frame_labels = {k: ("nothing" if pd.isna(v) else v) for k, v in frame_labels.items()}
    
    video_input = os.path.join(input_path, sequence + ".mp4")
    video_output = os.path.join(output_path, "output_" + sequence + ".mp4")
    placeholder_output = os.path.join(output_path, "placeholder_" + sequence + ".jpg")
    json_folder = os.path.join(json_path, sequence)  # Folder containing OpenPose JSON files
    
    process_video(video_input, video_output, frame_labels, placeholder_output, json_folder)

print("Videos have been processed and saved.")
