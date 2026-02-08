from huggingface_hub import HfApi
import os

# Initialize Hugging Face API client using an authentication token.
# The token is retrieved from environment variables for security.
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the contents of the 'tourism_project/deployment' folder to a Hugging Face Space.
# This space will host the Streamlit application.
api.upload_folder(
    folder_path="tourism_project/deployment",     # The local folder containing your Streamlit app and Dockerfile.
    repo_id="Garg06/Tourism-Package-Prediction",  # The target repository ID on Hugging Face (this should be a Space).
    repo_type="space",                      # Specifies that the repository is a Hugging Face Space.
    path_in_repo="",                          # Optional: subfolder path inside the repo (empty means root).
)
