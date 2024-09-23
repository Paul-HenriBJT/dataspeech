from datasets import load_dataset
import random
from huggingface_hub import notebook_login

def create_dataset_subset(dataset_name, subset_percentage, config=None, min_duration=1.0, buffer_percentage=2.0):
    # Load the dataset
    dataset = load_dataset(dataset_name, config)
    
    # Function to create a subset of a given split
    def subset_split(split):
        split_size = len(dataset[split])
        
        # Calculate the subset size with buffer
        buffered_percentage = subset_percentage + buffer_percentage
        buffered_subset_size = int(split_size * buffered_percentage / 100)
        
        # Select a larger subset
        indices = random.sample(range(split_size), buffered_subset_size)
        buffered_subset = dataset[split].select(indices)
        
        # Filter out audio files with duration less than min_duration
        filtered_subset = buffered_subset.filter(lambda x: x['audio']['array'].shape[0] / x['audio']['sampling_rate'] >= min_duration)
        
        # If we have more samples than needed after filtering, trim the excess
        if len(filtered_subset) > int(split_size * subset_percentage / 100):
            return filtered_subset.select(range(int(split_size * subset_percentage / 100)))
        
        return filtered_subset
    
    # Create subsets for each split
    subset = {}
    for split in dataset.keys():
        subset[split] = subset_split(split)
    
    return subset

# Example usage
dataset_name = "parler-tts/mls_eng_10k"
config = None  # Specify the config/subset if needed
subset_percentage = 10  # Percentage of the original dataset size
min_duration = 1.0  # Minimum duration in seconds
buffer_percentage = 2.0  # Additional percentage to select initially

# Create the subset
subset = create_dataset_subset(dataset_name, subset_percentage, config, min_duration, buffer_percentage)

# Push the subset to the Hugging Face Hub
new_dataset_name = f"mls-eng-{subset_percentage}percent-subset-min{min_duration}s"
for split in subset.keys():
    subset[split].push_to_hub(new_dataset_name, split=split)

print(f"Dataset pushed to Hugging Face Hub: {new_dataset_name}")