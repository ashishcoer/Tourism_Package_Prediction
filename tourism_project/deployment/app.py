import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Garg06/Tourism-Package-Model", filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the customer details and interaction data below to get a prediction.
""")

# User input fields
st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=90, value=30)
typeofcontact = st.selectbox("Type of Contact", options=['Company Invited', 'Self Inquiry'])
citytier = st.number_input("City Tier (1, 2, or 3)", min_value=1, max_value=3, value=1)
occupation = st.selectbox("Occupation", options=['Freelancer', 'Large Business', 'Salaried', 'Small Business', 'Unemployed'])
gender = st.selectbox("Gender", options=['Female', 'Male'])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferredpropertystar = st.number_input("Preferred Property Star (e.g., 3, 4, 5)", min_value=1, max_value=5, value=3)
maritalstatus = st.selectbox("Marital Status", options=['Divorced', 'Married', 'Single'])
numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
owncar = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
numberofchildrenvisiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", options=['Director', 'Executive', 'Manager', 'Senior Executive', 'VP'])
monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0, step=100.0)

st.header("Customer Interaction Data")
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
productpitched = st.selectbox("Product Pitched", options=['Basic', 'Deluxe', 'King', 'Standard', 'Super Deluxe'])
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)


# Assemble input into DataFrame, ensuring column order matches training data
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome,
    'NumberOfFollowups': numberoffollowups,
    'ProductPitched': productpitched,
}])


if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    # Using the classification_threshold defined in train.py
    classification_threshold = 0.45
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    result = "Customer WILL purchase the Wellness Tourism Package" if prediction == 1 else "Customer will NOT purchase the Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.info(f"Probability of purchase: {prediction_proba[0]:.2f}")
