"""
FoodVision101

Utility function for streamlit application
"""
import io
from typing import Any
from PIL import Image
import tensorflow as tf
import streamlit as st


def get_class_names(classes_path: str):
    """
    Return list of class names
    """
    with open(classes_path) as file:
        lines = file.readlines()
    classes = []
    for line in lines:
        classes.append(line.replace("\n", ""))
    return classes

def load_model(model_path: str):
    """
    Load the model
    """
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess(image: Any, size: list, scale: bool = False):
    """
    Preprocess the loaded image
    """
    image = tf.image.resize(image, size=size)
    if scale:
        image = image / 255.
    return image

def load_image():
    """
    Load the image
    """
    uploaded_file = st.file_uploader(label='Upload an image')
    image = None
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        image = Image.open(io.BytesIO(image_data))
        image = tf.cast(image, tf.float32)
        image = preprocess(image, [224, 224])
    return image

def predict(classes_path: str, model: Any, image: Any):
    """
    Predict the class name from image
    """
    k = 5
    class_names = get_class_names(classes_path)
    pred_prob = model.predict(tf.expand_dims(image, axis=0))
    pred_probs_topk, pred_indices_topk = tf.math.top_k(pred_prob, k=k, sorted=True)
    size = tf.size(pred_indices_topk)
    assert k == size
    pred_probs_topk = tf.reshape(pred_probs_topk, [k])
    pred_indices_topk = tf.reshape(pred_indices_topk, [k])
    for i in range(size):
        pred_class = class_names[pred_indices_topk[i]]
        pred_prob = pred_probs_topk[i].numpy()
        st.write(pred_class, pred_prob)
