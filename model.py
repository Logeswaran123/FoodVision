"""
FoodVision101

Model architecture
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def build_model(input_shape: Tuple,
                num_classes: int,
                include_top: bool = False):
    """
    Build a CNN model
    """
    # Create base model
    base_model = tf.keras.applications.EfficientNetB0(include_top=include_top)
    base_model.trainable = False # freeze base model layers

    # Create Functional model 
    inputs = layers.Input(shape=input_shape, name="input_layer", dtype=tf.float16)
    # Note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
    x = base_model(inputs, training=False) # set base_model to inference mode only
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dense(num_classes)(x) # want one output neuron per class 
    # Separate activation of output layer so we can output float32 activations
    outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_layer")(x) 
    model = tf.keras.Model(inputs, outputs)

    return model
