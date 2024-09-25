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
        print(f"Original dataset size: {len(ds)}")
        
        def safe_process(example):
            try:
                # Check if the text column exists and is a non-empty string
                if text_column_name not in example:
                    print(f"Text column '{text_column_name}' not found in example.")
                    return example  # Keep the example, don't filter it out
                if not isinstance(example[text_column_name], str):
                    print(f"Text is not a string: {type(example[text_column_name])}")
                    return example  # Keep the example, don't filter it out
                if len(example[text_column_name].strip()) == 0:
                    print("Text is empty after stripping whitespace.")
                    return example  # Keep the example, don't filter it out

                # If we get here, it means the entry is valid
                return example
            except Exception as e:
                print(f"Error processing entry: {e}")
                return example  # Keep the example, don't filter it out

        # Apply the safe_process function to each example
        cleaned_ds = ds.map(safe_process, desc="Processing examples")
        print(f"Dataset size after processing: {len(cleaned_ds)}")

        # Sort the dataset by duration if it exists
        if 'duration' in cleaned_ds.features:
            sorted_indices = np.argsort(cleaned_ds['duration'])
            cleaned_ds = cleaned_ds.select(sorted_indices)

            # Find the index where duration becomes >= min_duration
            cut_off_index = np.searchsorted(cleaned_ds['duration'], min_duration)

            # Select only the part of the dataset where duration >= min_duration
            cleaned_ds = cleaned_ds.select(range(cut_off_index, len(cleaned_ds)))
            print(f"Dataset size after duration filtering: {len(cleaned_ds)}")
        else:
            print("Warning: 'duration' column not found. Skipping duration-based filtering.")

        return cleaned_ds

    # Clean and filter the dataset
    if isinstance(dataset, DatasetDict):
        cleaned_dataset = DatasetDict()
        for k, v in dataset.items():
            print(f"Processing split: {k}")
            cleaned_split = clean_and_filter_dataset(v)
            cleaned_dataset[k] = cleaned_split
            print(f"Final size of split '{k}': {len(cleaned_split)}")
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
    dataset_subset = "polish"  # Set to None if there's no specific subset
    min_duration = 1  # Minimum duration in seconds
    text_column = "text"  # Replace with the actual name of your text column

    clean_and_update_dataset(dataset_name, local_save_path, subset=dataset_subset, min_duration=min_duration, text_column_name=text_column)