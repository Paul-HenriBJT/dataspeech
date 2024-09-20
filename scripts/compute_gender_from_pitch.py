import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from tqdm import tqdm

def process_audio(audio, feature_extractor, sampling_rate=16000, max_audio_len=5):
    if isinstance(audio, dict):
        speech_array = torch.tensor(audio['array'])
        sr = audio['sampling_rate']
    else:
        speech_array, sr = torchaudio.load(audio)

    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(sr, sampling_rate)
        speech_array = transform(speech_array)

    len_audio = speech_array.shape[1]

    if len_audio < max_audio_len * sampling_rate:
        padding = torch.zeros(1, max_audio_len * sampling_rate - len_audio)
        speech_array = torch.cat([speech_array, padding], dim=1)
    else:
        speech_array = speech_array[:, :max_audio_len * sampling_rate]

    speech_array = speech_array.squeeze().numpy()

    inputs = feature_extractor(
        speech_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt", 
        padding=True
    )

    return inputs

def predict_gender(audio, model, feature_extractor, device):
    inputs = process_audio(audio, feature_extractor)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

    return predicted_class_id

def main(args):
    # Load the dataset
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration)
    else:
        dataset = load_dataset(args.dataset_name)

    # Model configuration
    model_name_or_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    label2id = {"female": 0, "male": 1}
    id2label = {0: "F", 1: "M"}
    num_labels = 2

    # Initialize the feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Process the dataset
    speaker_data = []
    for split in dataset:
        # Group by speaker ID and get the first audio sample for each speaker
        speakers = dataset[split].unique(args.speaker_column)
        for speaker in tqdm(speakers, desc=f"Processing {split} split"):
            speaker_samples = dataset[split].filter(lambda x: x[args.speaker_column] == speaker)
            first_sample = speaker_samples[0]
            
            audio = first_sample[args.audio_column]
            prediction = predict_gender(audio, model, feature_extractor, device)
            speaker_data.append({"speaker_id": speaker, "sex": id2label[prediction]})

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(speaker_data)
    csv_path = args.output_file or "speaker_gender.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict gender for each speaker and output to CSV.")
    parser.add_argument("dataset_name", type=str, help="Name or path of the dataset.")
    parser.add_argument("--configuration", type=str, default=None, help="Dataset configuration to use.")
    parser.add_argument("--audio_column", type=str, default="audio", help="Name of the column containing audio data for classification.")
    parser.add_argument("--speaker_column", type=str, default="speaker_id", help="Name of the column containing speaker IDs.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the output CSV file.")

    args = parser.parse_args()
    main(args)