from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import numpy as np
import os

def clean_and_update_dataset(dataset_name, local_path, subset=None, min_duration=1, text_column_name='text'):
    # Load the dataset
    if subset:
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)

    def clean_and_filter_dataset(ds):
        def safe_process(example):
            try:
                # Check if the text column exists and is a non-empty string
                if text_column_name not in example or not isinstance(example[text_column_name], str) or len(example[text_column_name].strip()) == 0:
                    print(f"Skipping entry with invalid or empty text: {example.get(text_column_name, 'N/A')}")
                    return None

                # If we get here, it means the entry is valid
                return example
            except Exception as e:
                print(f"Error processing entry: {e}")
                return None

        # Apply the safe_process function to each example
        cleaned_ds = ds.map(safe_process, remove_columns=ds.column_names)

        # Remove None entries (which are the problematic ones)
        cleaned_ds = cleaned_ds.filter(lambda x: x is not None)

        # Sort the dataset by duration if it exists
        if 'duration' in cleaned_ds.features:
            sorted_indices = np.argsort(cleaned_ds['duration'])
            cleaned_ds = cleaned_ds.select(sorted_indices)

            # Find the index where duration becomes >= min_duration
            cut_off_index = np.searchsorted(cleaned_ds['duration'], min_duration)

            # Select only the part of the dataset where duration >= min_duration
            return cleaned_ds.select(range(cut_off_index, len(cleaned_ds)))
        else:
            print("Warning: 'duration' column not found. Skipping duration-based filtering.")
            return cleaned_ds

    # Clean and filter the dataset
    if isinstance(dataset, DatasetDict):
        cleaned_dataset = DatasetDict({k: clean_and_filter_dataset(v) for k, v in dataset.items()})
    else:
        cleaned_dataset = clean_and_filter_dataset(dataset)

    # Save the cleaned dataset locally
    cleaned_dataset.save_to_disk(local_path)
    print(f"Cleaned dataset saved locally at {local_path}")

    # Push the updated dataset back to the Hugging Face Hub
    api = HfApi()
    api.upload_folder(
        folder_path=local_path,
        repo_id=dataset_name,
        repo_type="dataset",
    )
    print(f"Cleaned dataset uploaded to Hugging Face Hub, updating {dataset_name}")

    # Clean up local files
    for root, dirs, files in os.walk(local_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(local_path)
    print(f"Cleaned up local files at {local_path}")

if __name__ == "__main__":
    # Replace these with your actual values
    dataset_name = "PHBJT/cml-tts"  # This is now both the source and destination
    local_save_path = "./temp_cleaned_dataset"
    dataset_subset = "portuguese"  # Set to None if there's no specific subset
    min_duration = 1  # Minimum duration in seconds
    text_column = "text"  # Replace with the actual name of your text column

    clean_and_update_dataset(dataset_name, local_save_path, subset=dataset_subset, min_duration=min_duration, text_column_name=text_column)