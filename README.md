# Food Vision
Food Vision 101 is a dataset consisting of 101k images belonging to **101 classes** of food.

## Objective
1. Split the 101k image data of 101 classes into train and test splits.
2. Build a CNN model using feature extraction method.
3. Fine tune the model from (1) by unfreezing the layers. Since, we have large dataset it is good to unfreeze all the layers.
4. Build a streamlit application to upload image, and run prediction on uploaded image using model from (2). Display Top-5 classes of predictions.

## Directory structure
.<br>
├── data<br>
├── streamlit<br>
&nbsp;    ├── models<br>
        ├── foodvision101_model_3_fine_tuned.h5<br>
    ├── images<br>
    ├── app.py<br>
    ├── utils.py<br>
    ├── classes.txt<br>
├── data_modification.py<br>
├── food_vision.py<br>
├── model.py<br>
├── preprocess.py<br>
├── fine_tuning.py<br>
├── utils.py<br>
<br>
- Download the data from [kaggle source|https://www.kaggle.com/datasets/dansbecker/food-101], and place the unzipped items in /data directory.
- data_modification.py script is used to prepare the dataset into train and test splits for training and testing the CNN model.
- model.py script contains the architecture of the model.
- food_vision.py is the base script the need to be run to train/test, and save the feature extraction model.
- fine_tuning.py script is used for fine tuning the feature extraction model and improve the accuracy/loss metrics.
- preprocess.py script contains the preprocess function for train and test data.
- utils.py script contains the utility functions.
