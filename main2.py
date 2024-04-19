import os
import cv2
import time
import pickle
import tempfile
import datetime
import numpy as np
import pandas as pd
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

def save_frame(device_num, cycle):
    cap = get_cap(device_num)

    if not cap.isOpened():
        st.error("Error: Unable to open the webcam.")
        return

    n = 0
    placeholder = st.empty()  # Create an empty placeholder


    st.title('Current Status')
    placeholder_text = st.empty()
    cnt = 0
    prev = 0
    while n <= cycle:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        placeholder.image(rgb_frame, channels="RGB", use_column_width=True, caption=f"Frame {n}")

        updated_text =  predict_image(frame)
        if (prev==1):
            cnt = cnt + 1
            if(cnt>=3):
                placeholder_text.text("Mail Sending...")
                email_alert(receiver)
                placeholder_text.text("Mail Sent")
                break

        elif(updated_text):
            prev=1
            placeholder_text.text("FALLEN EMERGENCY ALERT !!")
        else:
            cnt = 0
            prev=0
            placeholder_text.text("Everything Normal")

        # Update every second
        time.sleep(1)

        n += 1

    # Release the webcam
    cap.release()


st.title("Webcam Stream in Streamlit")
save_frame(0, 60)