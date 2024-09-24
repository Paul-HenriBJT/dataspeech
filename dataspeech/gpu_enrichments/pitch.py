import torch
import penn
import numpy as np

# Here we'll use a 10 millisecond hopsize
DEFAULT_HOPSIZE = 0.01

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

# Minimum number of samples required for pitch estimation
MIN_SAMPLES = 320  # This is equivalent to 20ms at 16kHz

def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    def process_single_audio(audio_tensor, sample_rate):
        original_length = audio_tensor.shape[1]
        
        # If audio is too short, pad it
        if original_length < MIN_SAMPLES:
            pad_size = MIN_SAMPLES - original_length
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))
        
        # Calculate adaptive hopsize
        adaptive_hopsize = max(DEFAULT_HOPSIZE, audio_tensor.shape[1] / 1000)
        
        try:
            pitch, periodicity = penn.from_audio(
                audio_tensor,
                sample_rate,
                hopsize=adaptive_hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0) % torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
            )
            
            # If we padded the audio, trim the pitch to match original length
            if original_length < MIN_SAMPLES:
                pitch = pitch[:, :int(original_length / (sample_rate * adaptive_hopsize))]
            
            return pitch.mean().cpu().item(), pitch.std().cpu().item()
        except Exception as e:
            print(f"Error processing audio: {e}")
            return np.nan, np.nan

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
