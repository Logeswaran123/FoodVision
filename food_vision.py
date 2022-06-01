"""
FoodVision101

Dataset: https://www.kaggle.com/datasets/kmader/food41
References:
(1) https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
(2) https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_data_modification.ipynb
(3) https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/03_convolutional_neural_networks_in_tensorflow.ipynb
(4) https://towardsdatascience.com/demo-your-model-with-streamlit-a76011467dfb
"""

import argparse
import math
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import mixed_precision

from preprocess import preprocess
from model import build_model
from utils import create_tensorboard_callback

SEED=42
INITIAL_LR = 0.001
tf.random.set_seed(SEED)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-tr",  "--train",  required=True, help="Path to train dataset directory", type=str)
ap.add_argument("-te", "--test", required=True, help="Path to test dataset directory", type=str)
ap.add_argument("-s", "--save", required=True,
                help="Path to save checkpoints, tensorboard event logs, model", type=str)
ap.add_argument("-name", "--model_name", required=True,
                help="Name of the feature extraction model to be saved", type=str)
ap.add_argument("-e", "--epoch", required=True,
                help="Number of epochs to train", default=10, type=int)
args = vars(ap.parse_args())

def lr_step_decay(epoch, lr):
    """
    Learning rate decay function
    """
    drop_rate = 0.5
    epochs_drop = 10.0
    return INITIAL_LR * math.pow(drop_rate, math.floor(epoch/epochs_drop))

def main():
    # Turn on mixed precision training
    mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision
    mixed_precision.global_policy() # should output "mixed_float16"

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Preprocess for model
    train_dir = args["train"]
    test_dir = args["test"]
    train_data, test_data = preprocess(train_dir=train_dir,
                                        test_dir=test_dir,
                                        label_mode="categorical",
                                        target_size=(224, 224),
                                        seed=SEED)

    num_classes = 101

    # Build CNN model
    cnn_model = build_model((224, 224, 3), num_classes, False)

    # Compile the model
    cnn_model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
                      metrics=["accuracy"])

    save_dir = args["save"]

    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = save_dir + "/model_checkpoints/FeatureExtraction/model_1.ckpt" # saving weights requires ".ckpt" extension
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      montior="val_accuracy", # save the model weights with best validation accuracy
                                                      save_best_only=True, # only save the best weights
                                                      save_weights_only=True, # only save model weights (not whole model)
                                                      verbose=1) # don't print out whether or not model is being saved

    # Create Learning rate scheduler to modify learning rate in model's training
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)

    # Fit the model
    cnn_model.fit(train_data,
                  epochs=args["epoch"],
                  steps_per_epoch=len(train_data),
                  validation_data=test_data,
                  validation_steps=int(0.15 * len(test_data)),
                  callbacks=[create_tensorboard_callback(save_dir + "/training_logs",
                                                        "feature_extraction"),
                            model_checkpoint,
                            lr_scheduler])

    # Save the model
    cnn_model.save(save_dir + "/" + args["model_name"] + ".h5")


if __name__ == "__main__":
    main()
