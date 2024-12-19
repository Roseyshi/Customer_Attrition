import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load


@st.cache_resource
def load_and_train_model():
    try:
        # Load and process data
        data = pd.read_csv("/content/clean_attrition_train (3).csv")
    except FileNotFoundError:
        st.error("Error: The file 'clean_attrition_train.csv' was not found. Please upload the file.")
        st.stop()

    # Splitting data into features and target
    X = data.drop('Attrition_Flag', axis=1)
    y = data['Attrition_Flag']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Save the trained model to a file
    dump(rf, 'attrition_classifier.joblib')
    return rf

@st.cache_resource
def load_model():
    import os
    if os.path.exists('attrition_classifier.joblib'):
        try:
            return load('attrition_classifier.joblib')
        except Exception as e:
            st.warning(f"Error loading model: {e}. Retraining the model...")
            return load_and_train_model()
    else:
        st.warning("Model file not found. Training a new model...")
        return load_and_train_model()

# Load the trained model
loaded_model = load_model()

# Streamlit App
st.title("Customer Attrition Prediction App")
st.markdown("Predict existing or attrited customer based on user inputs.")

# Input fields
st.sidebar.header("Enter Feature Values:")
feature1 = st.sidebar.number_input("Total_Trans_Ct", min_value=0.0, step=1.0)
feature2 = st.sidebar.number_input("Total_Trans_Amt", min_value =0.0, step=100.0)
feature3 = st.sidebar.number_input("Total_Revolving_Bal", min_value=0.0, step=100.0)
feature4 = st.sidebar.number_input("Total_Ct_Chng_Q4_Q1", min_value=0.0, step=100.0)
feature5 = st.sidebar.number_input("Avg_Utilization_Ratio", min_value=0.0, step=1.0)
feature6 = st.sidebar.number_input("Total_Relationship_Count", min_value=0.0, step=1.0)
feature7 = st.sidebar.number_input("Months_Inactive_12_mon", min_value=1.0, step=1.0)
feature8 = st.sidebar.number_input("Customer_Age", min_value=18.0, step=1.0)
feature9 = st.sidebar.number_input("Avg_Open_To_Buy", min_value=10.0, step=100.0)
feature10 = st.sidebar.number_input("Credit_Limit", min_value=10.0, step=1.0)
feature11 = st.sidebar.number_input("Months_on_book", min_value=1.0, step=1.0)
feature12 = st.sidebar.number_input("Dependent_count", min_value=1.0, step=1.0)
feature13 = st.sidebar.selectbox("Education_Level(0='Uneducated', 1='High School', 2='College', 3='Graduate', 4='Post-Graduate', 5='Doctorate')", [0,1,2,3,4,5])
feature14 = st.sidebar.selectbox("Income_Category (0='Less than $40K', 1='$40K - $60K', 2='$60K - $80K', 3='$80K - $120K', 4='$120K +')", [0,1,2,3,4])
feature15 = st.sidebar.selectbox("Marital_Status (0=Married, 1=Single, 2=Divorced)", [0,1,2])
feature16 = st.sidebar.selectbox("Gender (0=Male, 1=Female)", [0,1])
feature17 = st.sidebar.selectbox("Card_Category(0=Blue, 1=Silver, 2=Gold, 3=Platinum)", [0,1,2,3])
feature18 = st.sidebar.number_input("Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
min_value = 0.0, step = 0.01)
feature19 = st.sidebar.number_input("Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
min_value = 0.0, step = 0.01)




# Define target names
target_names = {0: "Existing Customer", 1: "Attrited Customer",}

# Predict Button
if st.button("Predict"):
    try:
        # Prepare the input as a 2D array
        inputs = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8,
        feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,feature17,feature18,feature19]])

        # Make prediction
        prediction = loaded_model.predict(inputs)

        # Display result
        st.success(f"Predicted Customer Classification: {target_names[prediction[0]]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")