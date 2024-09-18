from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

backend = EspeakBackend('es', language_switch='remove-flags')

def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):  
        speaking_rates = []
        phonemes_list = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                phonemes = backend.phonemize([text])[0]
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes.replace(' ', '')) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                phonemes = backend.phonemize([text])[0]
                
                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                
                speaking_rate = len(phonemes.replace(' ', '')) / audio_length

                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = backend.phonemize([batch[text_column_name]])[0]
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

        speaking_rate = len(phonemes.replace(' ', '')) / audio_length
        
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch

# Dummy example to test the function
import numpy as np

# Example batch with a list of texts and audios
dummy_batch = {
    "text": ["Hola, ¿cómo estás?", "Buenos días, amigo"],
    "audio": [
        {"sampling_rate": 16000, "array": np.random.rand(16000).reshape(1, -1)},  # 1 second audio
        {"sampling_rate": 16000, "array": np.random.rand(24000).reshape(1, -1)}   # 1.5 seconds audio
    ]
}

# Apply the rate_apply function
result = rate_apply(dummy_batch)

# Print the results
for text, phonemes, rate in zip(result["text"], result["phonemes"], result["speaking_rate"]):
    print(f"Text: {text}")
    print(f"Phonemes: {phonemes}")
    print(f"Speaking rate: {rate:.2f} phonemes/second")
    print()

# Example with speech_duration
dummy_batch_with_duration = {
    "text": ["Hola, ¿cómo estás?", "Buenos días, amigo"],
    "speech_duration": [1.0, 1.5]
}

result_with_duration = rate_apply(dummy_batch_with_duration)

print("Results with speech_duration:")
for text, phonemes, rate in zip(result_with_duration["text"], result_with_duration["phonemes"], result_with_duration["speaking_rate"]):
    print(f"Text: {text}")
    print(f"Phonemes: {phonemes}")
    print(f"Speaking rate: {rate:.2f} phonemes/second")
    print()