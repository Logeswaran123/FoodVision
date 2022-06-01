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
```python
python data_modification.py --classes <path to classes.txt> --images <path to archive>
```
Note:<br />

*<path to classes.txt\>* - Path to "/archive/meta/meta/classes.txt"<br />
*<path to archive\>* - Path to "/archive". archive is the unzipped directory from [KAGGLE SOURCE](https://www.kaggle.com/datasets/kmader/food41).<br />
train_test_split directory is created inside *<path to archive\>*.

### Feature Extraction
Now, there are many SOTA image classification model available that have been trained on large datasets. We can use one such model *EfficientNetB0* that is trained on ImageNet dataset (~14 million images). tf.keras.applications allows to use such model with pre-trained weights. Such models can be fitted to the scope of the task here. In this step, except final few layers that are added for the scope of task, other layers are frozen (i.e.non-trainable). Refer model.py to understand how keras application is used for utilizing EfficientNetB0 with pre-trained weights.

To fit the model to FoodVision 101 dataset, run the following command,
<br />
```python
python food_vision.py --train <path to train set> --test <path to test set> --save <path to save directory> --model_name <name of the model to save> --epoch <number of epochs to train>
```
Note:<br />

*<path to train set\>* - Path to train dataset directory<br />
*<path to test set\>* - Path to test dataset directory<br />
*<path to save directory\>* - Path to save checkpoints, tensorboard event logs, model<br />
*<name of the model to save\>* - Name of the feature extraction model to be saved<br />
*<number of epochs to train\>* - Number of epochs to train. Default: 10

### Fine Tuning
Alright! Let's use the saved model from *Feature Extraction* step, unfreeze all the layers (i.e.all layers as trainable). Now, fitting the model on the FoodVision101 dataset will improve the model's loss/accuracy. In this step, the initial learning rate will always be 10 times less than the initial learning rate used in Feature Extraction step. This step is the final push to improve the model's performance.

To fine tune the model from previous step, run the following command,
<br />
```python
python fine_tuning.py --train <path to train set> --test <path to test set> --saved_model <path to saved model> --save <path to save directory> --model_name <name of the model to save> --epoch <number of epochs to train>
```
Note:<br />

*<path to train set\>* - Path to train dataset directory<br />
*<path to test set\>* - Path to test dataset directory<br />
*<path to saved model\>* - Path to saved feature extraction model<br />
*<path to save directory\>* - Path to save checkpoints, tensorboard event logs, model<br />
*<name of the model to save\>* - Name of the fine tuned model to be saved<br />
*<number of epochs to train\>* - Number of epochs to train. Default: 100

## Streamlit
Streamlit is an open-source framework that helps to build an application with few lines of code. Check it out [Streamlit](https://streamlit.io/).

To run the FoodVision 101 application, clone the repository, and run the following commands,
```python
cd streamlit
streamlit run app.py
```

Happy Learning! :smile:
