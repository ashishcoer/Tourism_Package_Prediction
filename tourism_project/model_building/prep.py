# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset path on Hugging Face Hub.
# The API token is used for authentication to access and upload files.
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Garg06/Tourism-Package-Prediction/Tourism-Package-Prediction.csv"
# Load the dataset from the specified Hugging Face path into a Pandas DataFrame.
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop columns that are unique identifiers and not relevant for model training.
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# List of categorical columns to be label encoded.
label_encoded_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

# Apply Label Encoding to convert categorical text data into numerical representations.
# This is necessary for many machine learning algorithms.
for col in label_encoded_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define the target variable for the prediction task.
target_col = 'ProdTaken'

# Split the DataFrame into features (X) and target (y).
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform a train-test split on the dataset.
# This divides the data into training and testing sets to evaluate model performance.
# `test_size=0.2` allocates 20% of data for testing, `random_state=42` ensures reproducibility.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the training and testing sets to CSV files.
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# List of files to be uploaded to Hugging Face.
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload the generated CSV files (train/test splits) to the Hugging Face dataset repository.
# This makes the split datasets available for future steps or collaboration.
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # Extracts just the filename.
        repo_id="Garg06/Tourism-Package-Prediction",
        repo_type="dataset",
    )
