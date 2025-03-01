# multi-label classification of autistic mannerisms by machine learning

The following code allows you to reproduce the training and test of an algorithm for the classification of autistic mannerisms in video data as published in this [article in preparation].

In the absence of video data, you can use the data provided on request to test the algorithm's performance and run [Nested cross-validation](#nested-cross-validation) or the final model prediction.

## Prerequisites

### Video codings

The video files for training and test of the algorithm needs to be labeled according to a given coding scheme as described in the [article in preparation]. The length can be variable. Every single video sequence should contain either no mannerism, one mannerism, or a combination of different mannerisms. We used the software [Interact from Mangold](https://www.mangold-international.com/de/produkte/software/interact-videographie-software.html) for the video editing and saved the codings in an Excel spreadsheet with the name of the video file as the primary key (one row for each video file).

### OpenPose Installation

The OpenPose algorithm is used for feature extraction of skeletal key points from the video data. Installation instructions and prerequisites can be found here: [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

### Python prerequisites

Installation of Python 3.x (3.9 was used). [Hyperparametertuning](#hyperparametertuning) and [classification](#classification) were run on [Google colab notebooks](https://colab.research.google.com).

## Preprocessing

### Feature extraction

The first step is the feature extraction from the video files with OpenPose using the following flags:

  ```
  # Windows
  build\x64\release\OpenPoseDemo.exe --video $file --net_resolution "-1x288"
  --write_json "path_to_JSON_files" --display 0 --render_pose 0
  ```
### Multi-person-detection

Next step is using the script from [Tal Barami](https://github.com/TalBarami) posted [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1448#issuecomment-575936689) to set person_id in JSON files for multi-person-detection:

   [0_set_person_id](https://github.com/chrstnlmlr/ml_ass/blob/main/0_Barami_script_JSON_set_person_id.py)

### Generating sequences

The next script will split all video files into 15-frame sequences. The algorithm then can learn to detect whether there is no mannerism, one mannerism, or multiple mannerisms in parallel based on every 15-frame sequence.   

  [1_generate_sequences](https://github.com/chrstnlmlr/ml_ass/blob/main/1_preprocessing_masterfiles.py)

### JSON loader

This script converts the OpenPose JSON files (one file per frame, in our case nearly 2 million) to a single CSV file.  

  [2_json_loader](https://github.com/chrstnlmlr/ml_ass/blob/main/2_preprocessing_json_loader.py)

### Centralizing

This script centralizes the x- and y-coordinates to values between 0 and 1 and drops values with low detection accuracy.

  [3_centralizing](https://github.com/chrstnlmlr/ml_ass/blob/main/3_preprocessing_normalizing_and_cleaning.py)

## Classification

### Nested cross-validation

Nested cross-validation was performed to test the model's generalization ability. [Keras Hyperband Tuner](https://github.com/keras-team/keras-tuner) was used for Hyperparameter tuning and long short-term memory (LSTM) was utilized for multi-label classification.

  [4_nested_cross_validation](https://github.com/chrstnlmlr/ml_ass/blob/main/4_lstm_cv_stratifiedgroupkfold.py)

### Prediction

Two sequences were used to demonstrate prediction of the algorithm.

  [5_prediction](https://github.com/chrstnlmlr/ml_ass/blob/main/5_prediction.py)

### Open CV video overlays

A OpenCV script was used to create a overlay of the prediction in the two video sequences for demonstration purpose.

  [6_openCV_overlays](https://github.com/chrstnlmlr/ml_ass/blob/main/6_openCV_overlays.py)

## License

This work is licensed under CC BY-NC-SA 4.0 [Creative Commons](http://creativecommons.org/licenses/by-nc-sa/4.0/)
