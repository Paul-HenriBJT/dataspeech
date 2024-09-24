import torch
import penn
import numpy as np

# Here we'll use a 10 millisecond hopsize
hopsize = 0.01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = 0.065

# Default values to return if pitch estimation fails
DEFAULT_PITCH_MEAN = 100.0  # Example default value, adjust as needed
DEFAULT_PITCH_STD = 10.0   # Example default value, adjust as needed

def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    def process_single_audio(audio_tensor, sample_rate):
        try:
            pitch, periodicity = penn.from_audio(
                audio_tensor,
                sample_rate,
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0) % torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
            )
            return pitch.mean().cpu().item(), pitch.std().cpu().item()
        except Exception as e:
            print(f"Error processing audio of length {audio_tensor.shape[1]}: {e}")
            return DEFAULT_PITCH_MEAN, DEFAULT_PITCH_STD

    if isinstance(batch[audio_column_name], list):
        utterance_pitch_mean = []
        utterance_pitch_std = []
        for sample in batch[audio_column_name]:
            mean, std = process_single_audio(torch.tensor(sample["array"][None, :]).float(), sample["sampling_rate"])
            utterance_pitch_mean.append(mean)
            utterance_pitch_std.append(std)
        
        batch[f"{output_column_name}_mean"] = utterance_pitch_mean 
        batch[f"{output_column_name}_std"] = utterance_pitch_std
    else:
        sample = batch[audio_column_name]
        mean, std = process_single_audio(torch.tensor(sample["array"][None, :]).float(), sample["sampling_rate"])
        batch[f"{output_column_name}_mean"] = mean
        batch[f"{output_column_name}_std"] = std

    return batch
