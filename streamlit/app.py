"""
FoodVision101

Streamlit Application
"""
import argparse
import os
import streamlit as st
import utils

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c",  "--classes",  required=False, help="Path to classes.txt",
                default=r"classes.txt", type=str)
ap.add_argument("-m", "--model", required=False, help="Path to saved model",
                default=r"models\foodvision101_model_3_fine_tuned.h5", type=str)
try:
    args = vars(ap.parse_args())
except SystemExit as e:
    os._exit(e.code)

def main():
    st.title("FoodVision101 Demo Application")
    model = utils.load_model(args["model"])
    image = utils.load_image()
    button = st.button('Run prediction on uploaded image')
    if button:
        st.write('Predicting...')
        utils.predict(args["classes"], model, image)

if __name__ == "__main__":
    main()
