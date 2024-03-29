# Deployment
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import keras

def run():
    # Load the saved model
    model = tf.keras.models.load_model("model_seq.h5")

    # Define the Streamlit app
    st.title("Traffic Net Detection")
    st.write("This model can detect dense traffic, sparse traffic, an accident, and fire")

    # Allow the user to select an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        img = keras.preprocessing.image.load_img(uploaded_file, target_size=(220, 220))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255. # Rescale to [0,1]
        # Predict the image
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        # Find the predicted label
        predicted_label = np.argmax(score)
        st.success(f"Detection for uploaded image: {predicted_label}")
        # st.image(img, caption=f"{label_names[class_idx]}", use_column_width=True)        


def run():
    st.title("Traffic Net Detection")
    st.write("This model can detect dense traffic, sparse traffic, an accident, and fire")
    model_ann = tf.keras.models.load_model("model_seq.h5")
    uploaded_file = st.file_uploader("Choose an image file")
    if uploaded_file is not None:
    # Image predict
        st.image(uploaded_file, use_column_width=True)
        img = keras.preprocessing.image.load_img(uploaded_file, target_size=(220, 220))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255. # Rescale to [0,1]
        # Predict the image
        predictions = model_ann.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        # Find the predicted label
        predicted_label = np.argmax(score)

        if predicted_label == 0:
            predicted_label = 'The image show there is an accident'
        elif predicted_label == 1:
            predicted_label = 'The image show there is a dense traffic'
        elif predicted_label == 2:
            predicted_label = 'The image show there is a fire'
        else:
            predicted_label = 'The image show there is a sparse traffic'

        # Print the predicted label
        st.success(predicted_label)
if __name__ == "__app__":
    run()