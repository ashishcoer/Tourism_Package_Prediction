import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub.
# The model will be used for making predictions in the Streamlit app.
model_path = hf_hub_download(repo_id="Garg06/Tourism-Package-Model", filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Set the title and description for the Streamlit web application.
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the customer details and interaction data below to get a prediction.
""")

# User input fields for customer details, organized under a header.
st.header("Customer Details")
# Numerical input for Age, with defined min/max values and a default.
age = st.number_input("Age", min_value=18, max_value=90, value=30)
# Dropdown for Type of Contact, with string options.
typeofcontact = st.selectbox("Type of Contact", options=['Company Invited', 'Self Inquiry'])
# Numerical input for City Tier.
citytier = st.number_input("City Tier (1, 2, or 3)", min_value=1, max_value=3, value=1)
# Dropdown for Occupation.
occupation = st.selectbox("Occupation", options=['Freelancer', 'Large Business', 'Salaried', 'Small Business', 'Unemployed'])
# Dropdown for Gender.
gender = st.selectbox("Gender", options=['Female', 'Male'])
# Numerical input for Number of Persons Visiting.
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
# Numerical input for Preferred Property Star rating.
preferredpropertystar = st.number_input("Preferred Property Star (e.g., 3, 4, 5)", min_value=1, max_value=5, value=3)
# Dropdown for Marital Status.
maritalstatus = st.selectbox("Marital Status", options=['Divorced', 'Married', 'Single'])
# Numerical input for Number of Trips Annually.
numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
# Dropdown for Passport, with custom display for 0/1.
passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# Dropdown for Own Car, with custom display for 0/1.
owncar = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# Numerical input for Number of Children Visiting.
numberofchildrenvisiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
# Dropdown for Designation.
designation = st.selectbox("Designation", options=['Director', 'Executive', 'Manager', 'Senior Executive', 'VP'])
# Numerical input for Monthly Income.
monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0, step=100.0)

# User input fields for customer interaction data, organized under a header.
st.header("Customer Interaction Data")
# Numerical input for Pitch Satisfaction Score.
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
# Dropdown for Product Pitched.
productpitched = st.selectbox("Product Pitched", options=['Basic', 'Deluxe', 'King', 'Standard', 'Super Deluxe'])
# Numerical input for Number of Follow-ups.
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
# Numerical input for Duration of Pitch.
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)


# Assemble the user input into a Pandas DataFrame.
# The column names must exactly match those expected by the trained model.
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


# When the "Predict Purchase" button is clicked:
if st.button("Predict Purchase"):
    # Get prediction probabilities from the model.
    prediction_proba = model.predict_proba(input_data)[:, 1]
    # Define the classification threshold (as used during model evaluation).
    classification_threshold = 0.45
    # Convert probabilities to binary predictions based on the threshold.
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    # Display the prediction result to the user.
    result = "Customer WILL purchase the Wellness Tourism Package" if prediction == 1 else "Customer will NOT purchase the Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.info(f"Probability of purchase: {prediction_proba[0]:.2f}")
