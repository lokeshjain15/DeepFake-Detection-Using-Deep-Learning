import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("deepfake_model.keras")

st.title("Deepfake Detection System")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write(f"Prediction Score: {prediction:.4f}")

    if prediction < 0.5:
        st.error("FAKE IMAGE DETECTED")
    else:
        st.success("REAL IMAGE DETECTED")