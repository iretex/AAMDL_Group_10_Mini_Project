# adapted from: https://thecleverprogrammer.com/2020/08/02/gender-classification-model/

import sys
import streamlit as st
import cv2
from tensorflow import keras
import tempfile
import numpy as np
import tensorflow as tf
from io import StringIO

img_size = 96

# The model was built using the notebook with a kaggle dataset - https://www.kaggle.com/datasets/ruizgara/socofing
# Load the saved model
model = keras.models.load_model("model/model.h5")

# preprocess the image
def preprocess_image(image):
    # Convert to grayscale as required
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the desired size
    image = cv2.resize(image, (img_size, img_size))

    # Normalize the image
    image = image / 255.0

    # Expand dimensions to match model input shape
    image = image.reshape((1, img_size, img_size, 1))

    return image

# Function to make predictions
def make_prediction(image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Make a prediction using the loaded model
    prediction = model.predict(preprocessed_image)[0]


    return prediction

# Function to capture model summary as string
def get_model_summary(model):
    # Create a StringIO object to capture the printed output
    summary_str = StringIO()
    sys.stdout = summary_str  # Redirect stdout to StringIO

    # Print the model summary
    model.summary()

    # Capture the printed output as a string
    summary_text = summary_str.getvalue()

    # Reset stdout
    sys.stdout = sys.__stdout__

    return summary_text

# Streamlit app
def main():
    # App title
    st.title("Gender Classifier App")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Make prediction on button click
    if st.button("Predict"):
        if uploaded_file is not None:
            # Read the image file
            file_bytes = uploaded_file.read()
            image = cv2.imdecode(np.asarray(bytearray(file_bytes)), cv2.IMREAD_COLOR)

            if image is not None:
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Display the image
                st.image(gray_image, caption="Uploaded Image", use_column_width=True)

                # Make a prediction
                prediction = make_prediction(gray_image)
                st.success(f"Prediction: {prediction}")

                # Display the predicted gender
                # Define the categories
                predicted_label = np.argmax(prediction)
                categories = ["Male", "Female"]

                # predicted_gender = "Male" if prediction == 0 else "Female"
                
                # Get the predicted category
                predicted_category = categories[predicted_label]
                st.success(f"The predicted gender for the uploaded fingerprint is {predicted_category}.")
            else:
                st.warning("Unable to read the image file")
        else:
            st.warning("Please upload an image of a fingerprint")

        # Display model summary
        st.subheader("Model Summary")
        # st.text(model.summary())
        model_summary = get_model_summary(model)
        st.text(model_summary)

        # Display model architecture
        st.subheader("Model Architecture")
        model_plot = tf.keras.utils.plot_model(model, to_file="model_architecture.png", show_shapes=True)
        st.image("model_architecture.png")


# def main():
#     st.title('Pretrained model demo')
#     model = load_model()
#     categories = load_labels()
#     image = load_image()
#     result = st.button('Run on image')
#     if result:
#         st.write('Calculating results...')
#         predict(model, categories, image)


# Run the app
if __name__ == "__main__":
    main()