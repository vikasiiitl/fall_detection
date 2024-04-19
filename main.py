import os
import io
import cv2
import time
import pickle
import tempfile
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ReLU
from tensorflow.keras.models import load_model

import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os
load_dotenv()

model = load_model('my_model')
receiver = os.getenv("receiver")
text_input = st.text_input("Enter Email Address")
if text_input:
    receiver = text_input

def email_alert(to):
    user = os.getenv("sender_email")
    password = os.getenv("sender_password")
    msg = EmailMessage()
    msg.set_content('I just fell down! I might be unconscious. \nVisit me at my house and please call an ambulance'
)
    msg['subject'] = 'Emergency Alert!'
    msg['to'] = to
    msg['from'] = user

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()
    st.write("Email sent successfully to " , receiver)

# Function to make predictions
def predict_image(img):
    prediction = 5436
    resized_img = cv2.resize(img, (100,100) )
    cur_pred = []
    cur_pred.append(resized_img)
    cur_pred = np.array(cur_pred)
    predicted_labels = ( model.predict(cur_pred) >= 0.5).astype('int64')
    return predicted_labels[0]

# Streamlit UI
st.title("Fall Detection Software")

def get_cap(device_num):
    cap = cv2.VideoCapture(device_num)
    return cap

def status(val):
    if(val):
        return "FALLEN EMERGENCY ALERT !!"
    else:
        return "Everything Normal"



st.title("Upload your image to be checked")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    placeholder_text = st.empty()
    placeholder_text.text("Status Processing...")
    file_bytes = uploaded_file.getvalue()
    bin_stream = io.BytesIO(file_bytes)
    image = cv2.imdecode(np.frombuffer(bin_stream.read(), np.uint8), 1)
    res = predict_image(image)
    print(res)
    placeholder_text.write(status(res))  # Update the placeholder text using write()
    if res:
        email_alert(receiver)


placeholder_text2 = st.empty()
placeholder_text2.text("")