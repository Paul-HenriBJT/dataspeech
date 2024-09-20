import argparse
import torch
import torchaudio
import pandas as pd
import os
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from tqdm import tqdm
from collections import defaultdict

def process_audio(audio, feature_extractor, sampling_rate=16000, max_audio_len=5):
    if isinstance(audio, dict):
        speech_array = torch.tensor(audio['array'])
        sr = audio['sampling_rate']
    else:
        speech_array, sr = torchaudio.load(audio)

    # Ensure speech_array is 2D: (channels, time)
    if speech_array.dim() == 1:
        speech_array = speech_array.unsqueeze(0)

    # Convert to mono if it's multi-channel
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(sr, sampling_rate)
        speech_array = transform(speech_array)

    # Get the length of the audio
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
    dataset = load_dataset(args.dataset_name, args.configuration)

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
        # Create a dictionary to store the first sample for each speaker
        speaker_samples = defaultdict(list)
        
        # Iterate through the dataset once, collecting the first sample for each speaker
        for sample in tqdm(dataset[split], desc=f"Collecting samples from {split} split"):
            speaker_id = sample[args.speaker_column]
            if not speaker_samples[speaker_id]:
                speaker_samples[speaker_id] = sample[args.audio_column]
        
        # Process the collected samples
        for speaker, audio in tqdm(speaker_samples.items(), desc=f"Processing {split} split"):
            prediction = predict_gender(audio, model, feature_extractor, device)
            speaker_data.append({"speaker_id": speaker, "sex": id2label[prediction]})

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(speaker_data)
    csv_path = args.output_file or "speaker_gender.csv"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

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