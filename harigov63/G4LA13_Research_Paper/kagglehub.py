import os
import kagglehub

# 1. Define your specific Kaggle handles and directory paths
# Replace <KAGGLE_USERNAME> if different from your official handle
KAGGLE_USERNAME = "harigov63"  
DATASET_SLUG = "g4la13-agent-hyperparameters"
handle = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

local_dataset_dir = "./g4la13_dataset"
os.makedirs(local_dataset_dir, exist_ok=True)

# 2. Write the hyperparameter configurations into the directory
sampling_yaml_content = """
manifest_version: "1.0"
framework: "vllm"
model:
  name: "gemma-4-31b-it-qat-a4b16-ct"
  max_sequence_length: 32768
sampling:
  temperature: 0.2
  top_p: 0.95
  thinking_budget: 4096
"""

config_path = os.path.join(local_dataset_dir, "sampling.yaml")
with open(config_path, "w") as f:
    f.write(sampling_yaml_content.strip())

print(f"Created configuration file at: {config_path}")

# 3. Create and upload the initial dataset version via kagglehub
print(f"Uploading dataset to Kaggle as: {handle}...")
kagglehub.dataset_upload(handle, local_dataset_dir)

# 4. Optional: Code block to push updates later with version notes
# kagglehub.dataset_upload(handle, local_dataset_dir, version_notes='Updated thinking budget layout')
