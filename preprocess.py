"""
FoodVision101

Prepocess Data to feed into model
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess(train_dir: str,
                test_dir: str,
                label_mode: str,
                target_size: Tuple = (224, 224),
                seed=42):
    """
    Preprocess the data for the model

    Args:
    train_dir    - Path to train data directory
    test_dir     - Path to test data directory
    class_mode   - mode of the data to be preprocessed
                   ("categorical", "binary", "sparse", "input", or None)
    batch_size   - size of single batch. Default: 32
    target_size  - size of single image in tuple. Default: (224, 224)
    seed         - seed. Default: 42
    """

    # Import data and batch it
    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                    label_mode=label_mode,
                                                                    image_size=target_size)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode=label_mode,
                                                                image_size=target_size,
                                                                shuffle=False)

    return train_data, test_data
