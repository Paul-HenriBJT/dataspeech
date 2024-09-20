import argparse
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor
from tqdm import tqdm
from typing import List, Optional, Union, Dict

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, audio_column: str, sampling_rate: int = 16000, max_audio_len: int = 5):
        self.dataset = dataset
        self.audio_column = audio_column
        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        audio = self.dataset[index][self.audio_column]
        
        if isinstance(audio, dict):
            speech_array = torch.tensor(audio['array'])
            sr = audio['sampling_rate']
        else:
            speech_array, sr = torchaudio.load(audio)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)

        len_audio = speech_array.shape[1]

        if len_audio < self.max_audio_len * self.sampling_rate:
            padding = torch.zeros(1, self.max_audio_len * self.sampling_rate - len_audio)
            speech_array = torch.cat([speech_array, padding], dim=1)
        else:
            speech_array = speech_array[:, :self.max_audio_len * self.sampling_rate]

        speech_array = speech_array.squeeze().numpy()

        return {"input_values": speech_array, "attention_mask": None}

class CollateFunc:
    def __init__(self, processor: Wav2Vec2Processor, padding: Union[bool, str] = True,
                 pad_to_multiple_of: Optional[int] = None, return_attention_mask: bool = True,
                 sampling_rate: int = 16000, max_length: Optional[int] = None):
        self.processor = processor
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.sampling_rate = sampling_rate
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        input_values = [item["input_values"] for item in batch]

        batch = self.processor(
            input_values,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask
        )

        return {
            "input_values": batch.input_values,
            "attention_mask": batch.attention_mask if self.return_attention_mask else None
        }

def predict(dataloader, model, device: torch.device):
    model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)

            logits = model(input_values, attention_mask=attention_mask).logits
            scores = F.softmax(logits, dim=-1)

            pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

            preds.extend(pred)

    return preds

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.configuration)

    # Model configuration
    model_name_or_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    label2id = {"female": 0, "male": 1}
    id2label = {0: "female", 1: "male"}
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

    # Process each split in the dataset
    for split in dataset:
        custom_dataset = CustomDataset(dataset[split], args.audio_column, max_audio_len=5)

        data_collator = CollateFunc(
            processor=feature_extractor,
            padding=True,
            sampling_rate=16000,
        )

        dataloader = DataLoader(
            dataset=custom_dataset,
            batch_size=16,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=2
        )

        preds = predict(dataloader=dataloader, model=model, device=device)

        # Add predictions to the dataset
        dataset[split] = dataset[split].add_column("gender_prediction", [id2label[pred] for pred in preds])

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
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face API token.")

    args = parser.parse_args()
    main(args)