import os
import argparse
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np

def process_audio(audio, feature_extractor, sampling_rate=16000, max_length=5):
    if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
        speech_array = torch.tensor(audio['array'])
        sr = audio['sampling_rate']
    elif isinstance(audio, np.ndarray):
        speech_array = torch.tensor(audio)
        sr = sampling_rate
    else:
        raise ValueError("Unsupported audio format")

    if speech_array.dim() > 1:
        speech_array = speech_array.mean(dim=0)

    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        speech_array = resampler(speech_array)

    # Pad or truncate
    target_length = max_length * sampling_rate
    if speech_array.shape[0] < target_length:
        speech_array = torch.nn.functional.pad(speech_array, (0, target_length - speech_array.shape[0]))
    else:
        speech_array = speech_array[:target_length]

    inputs = feature_extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
    return inputs

def predict_gender(audio, model, feature_extractor, device):
    try:
        inputs = process_audio(audio, feature_extractor)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = probabilities.argmax().item()
        return model.config.id2label[predicted_class_id]
    except Exception as e:
        print(f"Error processing audio: {e}")
        return "unknown"

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.configuration)
    
    # Load the model and feature extractor
    model_name = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def process_and_add_gender(example):
        gender = predict_gender(example['audio'], model, feature_extractor, device)
        example['gender'] = gender
        return example

    # Process the dataset
    dataset = dataset.map(process_and_add_gender, num_proc=args.cpu_num_workers)

    # Save and push to hub if specified
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
    
    if args.repo_id:
        dataset.push_to_hub(args.repo_id, token=args.hub_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Repo id or local path of the dataset.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the model to the hub.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for data processing.")
    parser.add_argument("--hub_token", default=None, type=str, help="Hugging Face API token.")

    args = parser.parse_args()
    main(args)