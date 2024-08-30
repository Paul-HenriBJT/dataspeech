import argparse
from datasets import load_dataset
import numpy as np

def estimate_gender_from_pitch(pitch_values, low_threshold=120, high_threshold=180):
    """
    Estimate gender based on average pitch.
    """
    avg_pitch = np.mean(pitch_values)
    if avg_pitch < low_threshold:
        return 'male'
    elif avg_pitch > high_threshold:
        return 'female'
    else:
        return 'male'

def add_gender_column(example, pitch_column):
    """
    Add a gender column to the dataset based on pitch.
    """
    example['gender'] = estimate_gender_from_pitch(example[pitch_column])
    return example

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.configuration)

    # Add the gender column
    for split in dataset:
        dataset[split] = dataset[split].map(
            lambda x: add_gender_column(x, args.pitch_column),
            num_proc=args.num_workers
        )

    # Save the updated dataset
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
        print(f"Dataset saved to {args.output_dir}")

    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub)
        print(f"Dataset pushed to hub: {args.push_to_hub}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add gender column to dataset based on pitch.")
    parser.add_argument("dataset_name", type=str, help="Name or path of the dataset.")
    parser.add_argument("--configuration", type=str, default=None, help="Dataset configuration to use.")
    parser.add_argument("--pitch_column", type=str, default="utterance_pitch_mean", help="Name of the column containing pitch values.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the updated dataset.")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repository name to push the dataset to the Hugging Face Hub.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for dataset processing.")

    args = parser.parse_args()
    main(args)