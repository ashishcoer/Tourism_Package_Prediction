# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set MLflow tracking URI to a local server.
# This is typically used in production environments or CI/CD pipelines.
mlflow.set_tracking_uri("http://localhost:5000")
# Set the name of the MLflow experiment.
mlflow.set_experiment("mlops-training-experiment")

# Initialize Hugging Face API client.
api = HfApi()

# Define paths to the preprocessed training and testing data on Hugging Face Hub.
Xtrain_path = "hf://datasets/Garg06/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Garg06/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/Garg06/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/Garg06/Tourism-Package-Prediction/ytest.csv"

# Load the training and testing datasets using pandas.
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Define numeric and categorical features for preprocessing.
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]


# Calculate class weight to handle potential class imbalance in the target variable.
# This assigns a higher weight to the minority class during training.
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps using `make_column_transformer`.
# Numeric features are scaled using `StandardScaler`.
# Categorical features are one-hot encoded using `OneHotEncoder`.
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define the base XGBoost classifier model.
# `scale_pos_weight` is set to the calculated `class_weight` to address class imbalance.
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define the hyperparameter grid for `GridSearchCV`.
# This specifies the range of hyperparameter values to search for the best model.
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Create a machine learning pipeline that first preprocesses the data and then applies the XGBoost model.
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start an MLflow run to track experiment details.
with mlflow.start_run():
    # Perform hyperparameter tuning using GridSearchCV.
    # This searches for the best combination of hyperparameters.
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores to MLflow.
    # Each combination is logged as a nested MLflow run.
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run.
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log the best parameters found by GridSearchCV to the main MLflow run.
    mlflow.log_params(grid_search.best_params_)

    # Store the best model found during hyperparameter tuning.
    best_model = grid_search.best_estimator_

    # Define a classification threshold for converting probabilities to binary predictions.
    classification_threshold = 0.45

    # Make predictions on the training data using the best model.
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # Make predictions on the test data using the best model.
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Generate classification reports for both training and testing sets.
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log key evaluation metrics for the best model to MLflow.
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the best model locally.
    model_path = "best_machine_failure_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the saved model as an artifact in MLflow.
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Define the repository ID and type for Hugging Face model upload.
    repo_id = "Garg06/Tourism-Package-Model"
    repo_type = "model"

    # Check if the Hugging Face model repository exists; if not, create it.
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Upload the trained model to the Hugging Face model repository.
    api.upload_file(
        path_or_fileobj="best_machine_failure_model_v1.joblib",
        path_in_repo="best_machine_failure_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
