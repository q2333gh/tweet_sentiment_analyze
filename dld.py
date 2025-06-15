# pip install kagglehub
import kagglehub

# Download latest version
path = kagglehub.dataset_download("pavellexyr/the-reddit-climate-change-dataset")

print("Path to dataset files:", path)