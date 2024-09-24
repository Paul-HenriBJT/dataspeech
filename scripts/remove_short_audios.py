from datasets import load_dataset, Dataset, DatasetDict

def filter_and_upload_dataset(dataset_name, new_dataset_name, local_path, subset=None):
    # Load the dataset
    if subset:
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)

    # Function to filter a single dataset or subset
    def filter_dataset(ds):
        return ds.filter(lambda example: example['duration'] >= 1)

    # Filter the dataset
    if isinstance(dataset, DatasetDict):
        filtered_dataset = DatasetDict({k: filter_dataset(v) for k, v in dataset.items()})
    else:
        filtered_dataset = filter_dataset(dataset)

    # Save the filtered dataset locally
    if subset:
        filtered_dataset.save_to_disk(local_path, subset)
    else:
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
    new_dataset = "PHBJT/cml-tts"
    local_save_path = "./filtered_dataset"
    dataset_subset = "polish"  # Set to None if there's no specific subset

    filter_and_upload_dataset(original_dataset, new_dataset, local_save_path, subset=dataset_subset)