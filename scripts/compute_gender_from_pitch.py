import argparse
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import numpy as np

def classify_gender(audio, processor, model):
    """
    Classify gender based on audio using a pre-trained model.
    """
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

def add_gender_column(example, audio_column, processor, model):
    """
    Add a gender column to the dataset based on audio classification.
    """
    example['gender'] = classify_gender(example[audio_column], processor, model)
    return example

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.configuration)

    # Initialize the gender classification model and processor
    processor = AutoProcessor.from_pretrained("alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
    model = AutoModelForAudioClassification.from_pretrained("alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")

    # Add the gender column
    for split in dataset:
        dataset[split] = dataset[split].map(
            lambda x: add_gender_column(x, args.audio_column, processor, model),
            num_proc=args.num_workers
        )

    # Save the updated dataset
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
        print(f"Dataset saved to {args.output_dir}")

    if args.repo_id:
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration, token=args.hub_token)
        else:
            dataset.push_to_hub(args.repo_id, token=args.hub_token)
        print(f"Dataset pushed to hub: {args.repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add gender column to dataset based on audio classification.")
    parser.add_argument("dataset_name", type=str, help="Name or path of the dataset.")
    parser.add_argument("--configuration", type=str, default=None, help="Dataset configuration to use.")
    parser.add_argument("--audio_column", type=str, default="audio", help="Name of the column containing audio data for classification.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the updated dataset.")
    parser.add_argument("--repo_id", type=str, default=None, help="Repository name to push the dataset to the Hugging Face Hub.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for dataset processing.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face API token.")

    args = parser.parse_args()
    main(args)