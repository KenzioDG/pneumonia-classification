import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sqlite3
import time
import uuid

from utils import (
    init_db, load_model_safe, authenticate_user, register_user,
    update_progress_bar, clear_inputs, generate_patient_id,
    save_patient_data, reset_state
)

K.clear_session()

def main():
    st.sidebar.title(f'Welcome, {st.session_state.username}')
    st.sidebar.markdown('### User Guide')
    st.sidebar.markdown('1. **Upload an Image:** Click "Choose an image..." to upload a chest X-ray.')
    st.sidebar.markdown('2. **Wait for classification model to load**')    
    st.sidebar.markdown('3. **Choose a Method:**')
    st.sidebar.markdown('   - **Multiclass Classification:** Classify into Normal, Bacterial Pneumonia, or Viral Pneumonia.')
    st.sidebar.markdown('   - **Two Stage Binary Classification:** Distinguish Normal vs Pneumonia, if pneumonia is detected you may continue to classify between Bacterial vs Viral.')

    st.sidebar.markdown('4. **View Results:** The app displays predicted class and confidence score.')

    st.sidebar.markdown('5. **Download Sample Data:** [HERE](https://drive.google.com/drive/folders/1V-rkXiJo2H-yFYVortlLsNXDGO4kBKp9?usp=sharing)')

    st.title('Pneumonia Classification Application')
    st.markdown('This application enables users to analyze chest X-ray images for the detection of pneumonia. By uploading an image, users can utilize deep learning models to classify and identify pneumonia conditions based on visual data. The app provides detailed classification results and confidence scores, aiding in medical diagnostics and decision-making processes.')
    st.markdown('**This app is not intended to be a main basis of diagnosis, it is intended to be a second opinion for medical experts**')

    # CSS for fixed position elements
    st.markdown(
        """
        <style>
        .corner-text {
            position: fixed;
            top: 35px;
            right: 10px;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="corner-text">Logged in as {st.session_state.username}</div>', unsafe_allow_html=True)

    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

    tab1, tab2 = st.tabs(['Multiclass', 'Two Stage Binary'])

    with tab1:
        st.header('Multiclass Classification')
        st.markdown('Use this method to classify images into three categories: Normal, Bacterial Pneumonia, and Viral Pneumonia.')

        model_path = r"mobilenetv2_ME8.h5"
        model = load_model_safe(model_path)
        class_labels = ['Bacterial', 'Normal', 'Viral']

        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'], key="multiclass")
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("")
                
                if st.button("Reset Input", key="multiclass_omit", use_container_width=True):
                    reset_state()
                    st.experimental_rerun()
                
                patient_name = st.text_input('Patient Name', key="multiclass_name")
                notes = st.text_area('Notes (optional)', key="multiclass_notes")

                if st.button('Classify', use_container_width=True, key="multiclass_classify"):
                    if not patient_name:
                        st.error('Patient name is required')
                    else:
                        progress_bar = st.progress(0)
                        total_steps = 5
                        progress = 0

                        with st.spinner('Classifying...'):
                            progress = update_progress_bar(progress_bar, progress, 1, total_steps)
                            img = img.convert('RGB')
                            img = img.resize((224, 224))
                            img_array = image.img_to_array(img)
                            img_array = np.expand_dims(img_array, axis=0)
                            img_array = img_array / 255.0

                            progress = update_progress_bar(progress_bar, progress, 2, total_steps)
                            predictions = model.predict(img_array)
                            score = np.max(predictions)
                            predicted_class = class_labels[np.argmax(predictions)]

                            progress = update_progress_bar(progress_bar, progress, 2, total_steps)

                        # st.success(f"Predicted Class: {predicted_class} (confidence: {score:.2f})")
                        st.metric(label="Predicted Class", value=predicted_class)
                        st.metric(label="Confidence Score", value=f"{score:.2f}")

                        image_data = uploaded_file.getvalue()
                        patient_id = save_patient_data(st.session_state.username, patient_name, notes, image_data, predicted_class, score)
                        st.info(f"Patient ID: {patient_id}")
                        if st.button("Omit Result", key="multiclass_reset", use_container_width=True):
                            reset_state()
                            st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing image: {e}")

    with tab2:
        st.header('Two Stage Binary Classification')
        st.markdown('Use this method to classify images in two stages: first distinguishing Normal from Pneumonia, then identifying Bacterial or Viral Pneumonia.')

        model_path = r"mobilenetv2_ME8_BinaryNormal.h5"
        model = load_model_safe(model_path)
        class_labels = ['Normal', 'Pneumonia']

        uploaded_binarynorm = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'], key="binary")
        if uploaded_binarynorm is not None:
            try:
                img = Image.open(uploaded_binarynorm)
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("")

                patient_name = st.text_input('Patient Name', key="binary_name")
                notes = st.text_area('Notes (optional)', key="binary_notes")

                if st.button('Classify', key="binary_classify", use_container_width=True):
                    if not patient_name:
                        st.error('Patient name is required!')
                    else:
                        progress_bar = st.progress(0)
                        total_steps = 5
                        progress = 0

                        with st.spinner('Classifying...'):
                            progress = update_progress_bar(progress_bar, progress, 1, total_steps)
                            img = img.convert('RGB')
                            img = img.resize((224, 224))
                            img_array = image.img_to_array(img)
                            img_array = np.expand_dims(img_array, axis=0)
                            img_array = img_array / 255.0

                            progress = update_progress_bar(progress_bar, progress, 2, total_steps)
                            predictions = model.predict(img_array)
                            score = np.max(predictions)
                            predicted_class = class_labels[np.argmax(predictions)]

                            progress = update_progress_bar(progress_bar, progress, 2, total_steps)

                        st.metric(label="Predicted Class", value=predicted_class)
                        st.metric(label="Confidence Score", value=f"{score:.2f}")

                        image_data = uploaded_binarynorm.getvalue()
                        patient_id = save_patient_data(st.session_state.username, patient_name, notes, image_data, predicted_class, score)
                        st.info(f"Patient ID: {patient_id}")

                        # Store predicted_class and other relevant data in session state
                        st.session_state.predicted_class = predicted_class
                        st.session_state.first_classification_done = True
                        st.session_state.img_array = img_array
                        st.session_state.confirm_clicked = False  # Initialize confirm button state

                if st.session_state.get('first_classification_done'):
                    if st.session_state.predicted_class == "Pneumonia":
                        pneumonia_class = ['Bacterial', 'Viral']
                        model_path = r"mobilenetv2_ME9_BinaryPneumonia.h5"

                        if not st.session_state.get('confirm_clicked'):
                            st.warning('Pneumonia detected. Proceeding with second stage classification (Bacterial/Viral).')
                            if st.button('Confirm', key="confirm", use_container_width=True):
                                st.session_state.confirm_clicked = True  # Set confirm button state to True
                                st.experimental_rerun()  # Rerun the app to immediately reflect the state change

                        if st.session_state.get('confirm_clicked'):
                            model = load_model_safe(model_path)
                            progress_bar = st.progress(0)
                            total_steps = 5
                            progress = 0

                            with st.spinner('Classifying...'):
                                progress = update_progress_bar(progress_bar, progress, 2, total_steps)
                                predictions = model.predict(st.session_state.img_array)
                                score = np.max(predictions)
                                predicted_class = pneumonia_class[np.argmax(predictions)]

                                progress = update_progress_bar(progress_bar, progress, 3, total_steps)

                            st.metric(label="Predicted Pneumonia Type", value=predicted_class)
                            st.metric(label="Confidence Score", value=f"{score:.2f}")

                            # Display info message after classification results
                            st.info("Second stage classification completed.")
                            if st.button("Reset Input", key="multiclass_reset", use_container_width=True):
                                reset_state()
                                st.experimental_rerun()
                    else:
                        st.info("No pneumonia detected. Second stage classification not needed.")
                        if st.button("Reset Input", key="multiclass_reset", use_container_width=True):
                            reset_state()
                            st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing image: {e}")


# Main app logic
if st.session_state.logged_in:
    main()
else:
    if st.session_state.page == 'login':
        login()
    elif st.session_state.page == 'register':
        register()
