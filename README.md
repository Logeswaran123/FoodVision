# Food Vision
Food Vision 101 is a dataset consisting of 101k images belonging to **101 classes** of food.

## Objective
1. Split the 101k image data of 101 classes into train and test splits.
2. Build a CNN model using feature extraction method.
3. Fine tune the model from (1) by unfreezing the layers. Since, we have large dataset it is good to unfreeze all the layers.
4. Build a streamlit application to upload image, and run prediction on uploaded image using model from (2). Display Top-5 classes of predictions.

## Directory structure
<pre>
.
├── data
    ├── FoodVision
├── streamlit
    ├── models
        ├── foodvision101_model_3_fine_tuned.h5
    ├── images
    ├── app.py
    ├── utils.py
    ├── classes.txt
├── data_modification.py
├── food_vision.py
├── model.py
├── preprocess.py
├── fine_tuning.py
├── utils.py
</pre>
<br />

* Download the data from [KAGGLE SOURCE](https://www.kaggle.com/datasets/kmader/food41), and place the unzipped items in /data/FoodVision directory.
* data_modification.py script is used to prepare the dataset into train and test splits for training and testing the CNN model.
* model.py script contains the architecture of the model.
* food_vision.py is the base script the need to be run to train/test, and save the feature extraction model.
* fine_tuning.py script is used for fine tuning the feature extraction model and improve the accuracy/loss metrics.
* preprocess.py script contains the preprocess function for train and test data.
* utils.py script contains the utility functions.
* /streamlit directory contains the streamlit application.

## How to run
### Data Modification
Before starting to work with building the model, it is necessary to split the raw data into training and testing sets. The images are split into train and test sets based on the train.json and test.json files respectively. The entire dataset of 101k images is split into 75750 train images and 25250 test images. The train and test splits are ordered in their respective classes directories. To convert raw data into train and test splits, run the following command,
<br />
<pre><code>
```python
python data_modification.py --classes *<path to classes.txt>* --images *<path to archive>*
```
</code></pre>
<br />
Note:<br />
*<path to classes.txt\>* - Path to "/archive/meta/meta/classes.txt"<br />
*<path to archive\>* - Path to "/archive". archive is the unzipped directory from [KAGGLE SOURCE](https://www.kaggle.com/datasets/kmader/food41).

