from datasets import load_dataset
import os
import re

# Hugging Face credentials
HF_TOKEN = "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"
os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# Load the dataset
dataset_name = "PHBJT/mls_french_tagged_generated_sorted"
dataset = load_dataset(dataset_name)

def capitalize_sentences(text):
    # Split the text into sentences
    sentences = re.split('(?<=[.!?]) +', text)
    # Capitalize the first letter of each sentence
    capitalized_sentences = [s.capitalize() for s in sentences]
    # Join the sentences back together
    return ' '.join(capitalized_sentences)

# Function to modify the text and text_description columns
def modify_example(example):
    # Modify 'text' field
    example['text_description'] = "Language: french. " + example['text_description']
    
    # Modify 'text_description' field
    if 'text_description' in example:
        # Remove all single quotes
        example['text_description'] = example['text_description'].replace("'", "")
        # Capitalize the first letter of each sentence
        example['text_description'] = capitalize_sentences(example['text_description'])
    
    return example

# Apply the modification to all splits
modified_dataset = dataset.map(modify_example)

# Create a new dataset name
new_dataset_name = f"{dataset_name}_french"

# Upload the modified dataset
modified_dataset.push_to_hub(new_dataset_name, token=HF_TOKEN)

print(f"Modified dataset uploaded as '{new_dataset_name}'")