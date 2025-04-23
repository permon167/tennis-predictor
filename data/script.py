import kagglehub
import os

# Define the destination folder
destination_folder = os.path.join("data", "raw")

# Download the latest version to the specified folder
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull", path=destination_folder)

print("Path to dataset files:", path)