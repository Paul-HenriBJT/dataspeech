from datasets import load_dataset, Dataset, DatasetDict
import numpy as np

def filter_and_upload_dataset(dataset_name, new_dataset_name, local_path, subset=None, min_duration=1):
    # Load the dataset
    if subset:
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)

    def optimize_and_filter_dataset(ds):
        # Filter out rows where:
        # 1. text is just "." or "—"
        # 2. transcript_wav2vec is null
        ds = ds.filter(lambda x: isinstance(x['text'], str) and x['text'].strip() not in [".", "—"] and x['transcript_wav2vec'] is not None)

        # Sort the dataset by duration
        sorted_indices = np.argsort(ds['duration'])
        ds = ds.select(sorted_indices)

        # Find the index where duration becomes >= min_duration
        cut_off_index = np.searchsorted(ds['duration'], min_duration)

        # Select only the part of the dataset where duration >= min_duration
        return ds.select(range(cut_off_index, len(ds)))

    # Filter the dataset
    if isinstance(dataset, DatasetDict):
        filtered_dataset = DatasetDict({k: optimize_and_filter_dataset(v) for k, v in dataset.items()})
    else:
        filtered_dataset = optimize_and_filter_dataset(dataset)

    # Save the filtered dataset locally
    filtered_dataset.save_to_disk(local_path)
    print(f"Filtered dataset saved locally at {local_path}")

    # Push the dataset to the Hugging Face Hub
    if subset:
        filtered_dataset.push_to_hub(new_dataset_name, subset)
    else:
        filtered_dataset.push_to_hub(new_dataset_name)
    print(f"Filtered dataset uploaded to Hugging Face Hub as {new_dataset_name}")

if __name__ == "__main__":
    # Replace these with your actual values
    original_dataset = "ylacombe/cml-tts"
    new_dataset = "PHBJT/cml-tts-cleaned"
    local_save_path = "./filtered_dataset"
    dataset_subset = "german"  # Set to None if there's no specific subset
    min_duration = 1.5  # Minimum duration in seconds

    filter_and_upload_dataset(original_dataset, new_dataset, local_save_path, subset=dataset_subset, min_duration=min_duration)