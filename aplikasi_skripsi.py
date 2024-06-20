import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sqlite3
import time

# Clear previous session
K.clear_session()

# Database initialization function
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Function to load models with error handling
def load_model_safe(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Function to register user
def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Function to update progress bar
def update_progress_bar(progress_bar, progress, step, total_steps):
    progress += step
    progress_bar.progress(progress / total_steps)
    return progress

# Main application
def main():
    st.title('Pneumonia Classification')
    st.write("""
    This application helps in classifying pneumonia from chest X-ray images. 
    You can either perform a multiclass classification to identify if the pneumonia 
    is Bacterial, Normal, or Viral, or use the two-stage binary classification to first 
    detect pneumonia and then identify its type.
    """)

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

    # Logout button
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

    # Sidebar content
    st.sidebar.title("Welcome!")
    st.sidebar.write(f"Hello, {st.session_state.username}!")
    st.sidebar.header("How to Use This Application")
    st.sidebar.write("""
    - **Upload Image**: Choose an image file (JPG, JPEG, PNG) for classification.
    - **Multiclass Tab**: Predicts whether the image is Bacterial, Normal, or Viral.
    - **Two Stage Binary Tab**: First checks if the image is Normal or Pneumonia, and if Pneumonia, further classifies into Bacterial or Viral.
    - **View Results**: See the predicted class and confidence score.
    - **Logout**: Click the logout button to end your session.
    """)
    st.sidebar.header("Download Data")
    st.sidebar.write("You can download the sample data to test the application [here](https://drive.google.com/drive/folders/1V-rkXiJo2H-yFYVortlLsNXDGO4kBKp9?usp=sharing).")

    tab1, tab2 = st.tabs(['Multiclass', 'Two Stage Binary'])

    with tab1:
        st.subheader("Multiclass Classification")
        st.write("""
        In this tab, you can upload a chest X-ray image to classify it into one of the following categories:
        - **Bacterial Pneumonia**
        - **Normal**
        - **Viral Pneumonia**
        """)
        
        model_path = r"mobilenetv2_ME8.h5"
        model = load_model_safe(model_path)
        class_labels = ['Bacterial', 'Normal', 'Viral']

        uploaded_file = st.file_uploader('Choose an image... (you can download test images in the sidebar)', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("")

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

                st.success(f"Predicted Class: {predicted_class} (confidence: {score:.2f})")
                st.metric(label="Predicted Class", value=predicted_class)
                st.metric(label="Confidence Score", value=f"{score:.2f}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    with tab2:
        st.subheader("Two Stage Binary Classification")
        st.write("""
        In this tab, you can upload a chest X-ray image to first classify it as Normal or Pneumonia. 
        If the image is classified as Pneumonia, it will further be classified as either Bacterial or Viral Pneumonia by a separate classification model.
        """)
        
        model_path = r"mobilenetv2_ME8_BinaryNormal.h5"
        model = load_model_safe(model_path)
        class_labels = ['Normal', 'Pneumonia']

        uploaded_binarynorm = st.file_uploader('Choose an image... (you can download test image in the sidebar)', type=['jpg', 'jpeg', 'png'], key=2)

        if uploaded_binarynorm is not None:
            try:
                img = Image.open(uploaded_binarynorm)
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("")

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

                st.success(f"Predicted Class: {predicted_class} (confidence: {score:.2f})")
                st.metric(label="Predicted Class", value=predicted_class)
                st.metric(label="Confidence Score", value=f"{score:.2f}")
                
                if predicted_class == "Pneumonia":
                    if 'confirmed' not in st.session_state:
                        st.session_state.confirmed = False

                    if not st.session_state.confirmed:
                        st.warning('Pneumonia detected, would you like to try and identify the cause? (Bacterial/Viral)')
                        if st.button('Confirm'):
                            st.session_state.confirmed = True
                            st.experimental_rerun()

                    if st.session_state.confirmed:
                        pneumonia_class = ['Bacterial', 'Viral']
                        try:
                            model_path = r"mobilenetv2_ME9_BinaryPneumonia.h5"
                            model = load_model_safe(model_path)
                            class_labels = ['Bacterial', 'Viral']                       
                            
                            progress_bar = st.progress(0)
                            total_steps = 5
                            progress = 0

                            with st.spinner('Classifying...'):
                                progress = update_progress_bar(progress_bar, progress, 2, total_steps)
                                predictions = model.predict(img_array)
                                score = np.max(predictions)
                                predicted_class = pneumonia_class[np.argmax(predictions)]

                                progress = update_progress_bar(progress_bar, progress, 3, total_steps)

                            st.success(f"Predicted Class: {predicted_class} (confidence: {score:.2f})")
                            st.metric(label="Predicted Class", value=predicted_class)
                            st.metric(label="Confidence Score", value=f"{score:.2f}")
                        except Exception as e:
                            st.error(f"Error processing image with secondary model: {e}")

            except Exception as e:
                st.error(f"Error processing image: {e}")


# Login function
def login():
    st.title('Login')
    st.write("Please enter your username and password to log in.")

    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login', use_container_width=True):
        with st.spinner('Logging in...'):
            time.sleep(1)  # Simulate a delay for logging in
            user = authenticate_user(username, password)
            if user:
                st.success('Logged in successfully!')
                time.sleep(1)  # Briefly show the success message
                st.session_state.logged_in = True
                st.session_state.username = username  # Store the username in the session state
                st.experimental_rerun()
            else:
                st.error('Invalid username or password')

    st.write("Don't have an account?")
    if st.button('Register', use_container_width=True):
        st.session_state.page = 'register'
        st.experimental_rerun()

# Registration function
def register():
    st.title('Register')
    st.write("Create a new account by choosing a unique username and password.")

    username = st.text_input('New Username')
    password = st.text_input('New Password', type='password')

    if st.button('Register', use_container_width=True):
        with st.spinner('Creating account...'):
            time.sleep(1)  # Simulate a delay for registration
            if register_user(username, password):
                st.success('Account created successfully! You can now log in.')
                time.sleep(1)
                st.session_state.page = 'login'
                st.experimental_rerun()
            else:
                st.error('Username already taken. Please choose a different username.')

    st.write("Already have an account?")
    if st.button('Go to Login', use_container_width=True):
        st.session_state.page = 'login'
        st.experimental_rerun()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'

init_db()

if st.session_state.logged_in:
    main()
else:
    if st.session_state.page == 'login':
        login()
    elif st.session_state.page == 'register':
        register()
