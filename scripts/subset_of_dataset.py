from datasets import load_dataset, DatasetDict
import random

# Set your Hugging Face API token
HF_TOKEN = "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"

# Set the dataset name and your username
ORIGINAL_DATASET_NAME = "PHBJT/mls_french_tokenized_french"
YOUR_USERNAME = "PHBJT"
NEW_DATASET_NAME = "mls_french_tokenized_reduced_400_french"

# Load the dataset
dataset = load_dataset(ORIGINAL_DATASET_NAME)

# Function to get a random subset
def get_subset(dataset, percentage=0.01, max_samples=None):
    if max_samples is not None:
        subset_size = min(max_samples, len(dataset))
    else:
        subset_size = max(1, int(len(dataset) * percentage))
    
    indices = random.sample(range(len(dataset)), subset_size)
    return dataset.select(indices)

# Create subsets for each split
new_dataset = DatasetDict()
for split in dataset.keys():
    if split == "test":
        new_dataset[split] = get_subset(dataset[split], max_samples=1000)
    else:
        new_dataset[split] = get_subset(dataset[split], percentage=0.4)

# Push the new dataset to the Hugging Face Hub
new_dataset.push_to_hub(f"{YOUR_USERNAME}/{NEW_DATASET_NAME}", token=HF_TOKEN)

print(f"Subset dataset uploaded to: https://huggingface.co/datasets/{YOUR_USERNAME}/{NEW_DATASET_NAME}")
print("\nDataset statistics:")
for split in new_dataset.keys():
    print(f"{split}: {len(new_dataset[split])} samples")