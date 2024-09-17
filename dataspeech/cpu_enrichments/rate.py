from phonemizer import phonemize

# Initialize the phonemizer backend for German
backend = 'espeak'
language = 'de'

def rate_apply(batch, rank=None, text_column_name="text", duration_column_name="speech_duration"):
    if isinstance(batch[text_column_name], list):  
        speaking_rates = []
        phonemes_list = []
        for text, duration in zip(batch[text_column_name], batch[duration_column_name]):
            phonemes = phonemize(text, language=language, backend=backend, strip=True)
            duration = max(duration, 0.01)  # Avoid division by zero
            speaking_rate = len(phonemes) / duration
            speaking_rates.append(speaking_rate)
            phonemes_list.append(phonemes)
        
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = phonemize(batch[text_column_name], language=language, backend=backend, strip=True)
        duration = max(batch[duration_column_name], 0.01)  # Avoid division by zero
        speaking_rate = len(phonemes) / duration
        
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch

