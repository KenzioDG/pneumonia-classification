import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sqlite3
import time
import uuid

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
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            notes TEXT,
            image BLOB,
            classification TEXT,
            confidence REAL,
            timestamp TEXT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS id_counter (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    ''')
    conn.commit()
    conn.close()

# Function to load models with error handling
def load_model_safe(model_path):
    try:
        model = load_model(model_path)
        st.success("Classification model loaded successfully")
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

def clear_inputs():
    st.session_state['patient_name'] = ''
    st.session_state['notes'] = ''
    st.session_state['uploaded_file'] = None

def generate_patient_id():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO id_counter (id) VALUES (NULL)')
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return f"PT{new_id:03d}"

def save_patient_data(username, patient_name, notes, image_data, classification, confidence):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    patient_id = generate_patient_id()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO patients (patient_id, username, patient_name, notes, image, classification, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, username, patient_name, notes, image_data, classification, confidence, timestamp))
    conn.commit()
    conn.close()
    return patient_id

def reset_state():
    keys_to_reset = [
        'uploaded_file', 'patient_name', 'notes', 
        'first_classification_done', 'img_array', 'predicted_class',
        'multiclass', 'multiclass_name', 'multiclass_notes',
        'binary', 'binary_name', 'binary_notes'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state['uploaded_file'] = None  # Reset uploaded file

# Login function
def login():
    st.title('Login')
    st.markdown('Please enter your username and password to log in.')

    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login', use_container_width=True):
        if not username or not password:
            st.error('Username and password cannot be empty!')
        else:     
            with st.spinner('Logging in...'):
                time.sleep(1)  # Simulate a delay for logging in
                user = authenticate_user(username, password)
                if user:
                    st.success('Account found')
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
    st.markdown('Please fill in the form below to create a new account.')

    username = st.text_input('New Username')
    password = st.text_input('New Password', type='password')

    if st.button('Register', use_container_width=True):
        if not username or not password:
            st.error('Username and password cannot be empty')
        else:
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

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'first_classification_done' not in st.session_state:
    st.session_state.first_classification_done = False

# Initialize the database
init_db()
