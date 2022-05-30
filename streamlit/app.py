"""
FoodVision101

Streamlit Application
"""
import streamlit as st
import utils

def main():
    st.title("FoodVision101 Demo Application")
    model = utils.load_model()
    image = utils.load_image()
    button = st.button('Run prediction on uploaded image')
    if button:
        st.write('Predicting...')
        utils.predict(model, image)

if __name__ == "__main__":
    main()
