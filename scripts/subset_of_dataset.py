from datasets import load_dataset
import random
from huggingface_hub import notebook_login

def create_dataset_subset(dataset_name, subset_percentage, config=None):
    # Load the dataset
    dataset = load_dataset(dataset_name, config)
    
    # Function to create a subset of a given split
    def subset_split(split):
        split_size = len(dataset[split])
        subset_size = int(split_size * subset_percentage / 100)
        indices = random.sample(range(split_size), subset_size)
        return dataset[split].select(indices)
    
    # Create subsets for each split
    subset = {}
    for split in dataset.keys():
        subset[split] = subset_split(split)
    
    return subset

# Example usage
dataset_name = "ylacombe/cml-tts"
config = "french"  # Specify the config/subset if needed
subset_percentage = 20  # Percentage of the original dataset size

# Create the subset
subset = create_dataset_subset(dataset_name, subset_percentage, config)


# Login to Hugging Face (you'll need to enter your auth token)
notebook_login()

# Push the subset to the Hugging Face Hub
new_dataset_name = f"cml-tts-{subset_percentage}percent-subset"
for split in subset.keys():
    subset[split].push_to_hub(new_dataset_name, split=split)

print(f"Dataset pushed to Hugging Face Hub: {new_dataset_name}")