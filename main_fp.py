import streamlit as st
import tensorflow as tf
from tensorflow import keras
import keras.utils as image
from PIL import Image
import requests
import numpy as np

st.title('Mask Correctness Detection')
st.text('Created by Yohanes Setiawan')

# Load model
new_model = tf.keras.models.load_model('my_model', compile=False)

file_upload = st.file_uploader("Upload your image here:")
if file_upload is not None:
    display_image = Image.open(file_upload)
    st.image(display_image)

bt_predict = st.button('Predict!')
try:
    if bt_predict:
        img = image.load_img(file_upload, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = new_model.predict(images, batch_size=10)
        if classes[0,0] != 0:
            st.text("Be careful! You are using incorrect mask.")
        elif classes[0,1] != 0:
            st.text("You are wearing correct mask! Stay healthy :)")
        else:
            st.text("You are not wearing mask!")
except:
    st.text("Insert image first, please!")
        
