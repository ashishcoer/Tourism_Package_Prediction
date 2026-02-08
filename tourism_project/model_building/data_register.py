from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "Garg06/Tourism-Package-Prediction"
repo_type = "dataset"

# Initialize Hugging Face API client using an authentication token.
# The token is retrieved from environment variables for security.
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the Hugging Face dataset repository exists.
# If it doesn't exist, a new repository is created with the specified repo_id and type.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False) # 'private=False' makes the repo public.
    print(f"Space '{repo_id}' created.")

# Upload the contents of the local 'tourism_project/data' folder to the Hugging Face repository.
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
