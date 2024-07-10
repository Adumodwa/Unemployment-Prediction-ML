import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the model
def load_model():
    with open('model.pkl','rb') as file:
        model = pickle.load(file)
    return model

# Function to preprocess input data
def preprocess_input(degree_level, gender):
    # Define mapping for degree levels
    degree_level_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}

    # Define mapping for gender
    gender_mapping = {'Male': 0, 'Female': 1}

    # Encode degree level and gender
    degree_level_encoded = degree_level_mapping.get(degree_level, -1)
    gender_encoded = gender_mapping.get(gender, -1)

    # Check if any category is missing
    if degree_level_encoded == -1 or gender_encoded == -1:
        st.error("Invalid input. Please select valid degree level and gender.")
        return None

    # Create DataFrame from preprocessed data
    data = pd.DataFrame({'Degree Level': [degree_level_encoded], 'Gender': [gender_encoded]})
    
    return data


# Streamlit app
def main():
    st.title("Unemployment Rate Prediction")

    # Input widgets
    degree_level = st.selectbox("Select Degree Level", ['High School', 'Bachelor', 'Master', 'PhD'])
    gender = st.selectbox("Select Gender", ['Male', 'Female'])

    # Load the model
    model = load_model()

    # Predict button
    if st.button("Predict"):
        # Preprocess input data
        input_data = preprocess_input(degree_level, gender)
        
        if input_data is not None:
            try:
                # Predict unemployment rate
                prediction = model.predict(input_data)
                # Display prediction
                st.success(f"Predicted unemployment rate: {prediction[0]:.2f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
 