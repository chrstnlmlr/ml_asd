# Project Title

Semi-automated multi-label classification of autistic mannerisms by machine
learning on post-hoc skeletal tracking

## Getting Started

The following code allows you to reproduce the training and test of a
algorithm for classification of autistic mannerisms in video data.

In the absence of video data, you can use the data provided on this page
to test the algorithms performance and start with [hyperparametertuning](#hyperparametertuning)
or [classification](#classification).

## Prerequisites

### Video codings

The video files must be prepared as followed:
- The length can be variable.
- The individual video files should each contain either no mannerism, one
mannerism, or a combination of different mannerisms.
- For the video editing we used the software [Interact from Mangold](https://www.mangold-international.com/de/produkte/software/interact-videographie-software.html).
- For every video file, labelings according to a given coding scheme are needed
as described in the [paper](Link_to_paper). We saved the codings in an Excel
spreadsheet with the name of the video file as primary key.

### OpenPose Installation

The OpenPose Algorithm ist used for feature extraction of skeletal keypoints
from the video data. Installation instructions and prerequisites can be found
here:
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Python prerequisites

Installation of Python 3.x (3.9 was used)

## Preprocessing

### Feature extraction

First step is feature extraction with OpenPose using following flags:

  ```
  # Windows
  build\x64\release\OpenPoseDemo.exe --video $file --net_resolution "-1x288"
  --write_json "path_to_JSON_files" --display 0 --render_pose 0
  ```
### Multi-person-detection

Next step is using the script from [Tal Barami](https://github.com/TalBarami) posted [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1448#issuecomment-575936689)
to set person_id in JSON files for multi-person-detection:

   [0_set_person_id](https://github.com/chrstnlmlr/ml_ass/blob/main/0_Barami_script_JSON_set_person_id.py)

### Generating sequences

The next script will split all video files into 15 frame-sequences. The algorithm
then can learn to detect whether there is no mannerism, one mannerism, or multiple
mannerisms in parallel based on every 15-frame-sequence.   

  [1_generate_sequences](https://github.com/chrstnlmlr/ml_ass/blob/main/1_preprocessing_masterfiles.py)

### JSON loader

This script converts the OpenPose JSON files (one file per frame, in our case about
1.7 million) to a single csv-file.  

  [2_json_loader](https://github.com/chrstnlmlr/ml_ass/blob/main/2_preprocessing_json_loader.py)

### Centralizing

This scripts centralizes the x- and y-coordinates to values between 0 and 1 and
drops values with low detection-accuracy.

  [3_centralizing](https://github.com/chrstnlmlr/ml_ass/blob/main/3_preprocessing_normalizing_and_cleaning.py)

### Train test split

Train test split with 3-fold cross-validation is performed with two different   
splitting techniques:
- Random split (splitting all sequences randomly)
- Person split (split sequences by person, so that the algorithm can learn to
  identify mannerisms in new, unknown persons)

  [4_train_test_split](https://github.com/chrstnlmlr/ml_ass/blob/main/4_preprocessing_train_test_split_cross_validation.py)

## Classification

### hyperparametertuning

Hyperparametertuning was performed with [keras_tuner](https://github.com/keras-team/keras-tuner):

- [5_random_split_tuner](https://github.com/chrstnlmlr/ml_ass/blob/main/5_lstm_cv_flap_jump_RS_tuner.py)
- [5_person_split_tuner](https://github.com/chrstnlmlr/ml_ass/blob/main/5_lstm_cv_flap_jump_PS_tuner.py)

### cross-validation

3-folds cross-validation was performed to test models generalization ability:

- [5_random_split_cv](https://github.com/chrstnlmlr/ml_ass/blob/main/5_lstm_cv_flap_jump_RS.py)
- [5_person_split_cv](https://github.com/chrstnlmlr/ml_ass/blob/main/5_lstm_cv_flap_jump_PS.py)

## Authors

 - **Christian Lemler** - *Provided code* - [CL](https://github.com/chrstnlmlr)

## License

This work is licensed under CC BY-NC-SA 4.0 [Creative Commons](http://creativecommons.org/licenses/by-nc-sa/4.0/)

## Acknowledgments

 - Thanks to [Tal Barami](https://github.com/TalBarami) for script
