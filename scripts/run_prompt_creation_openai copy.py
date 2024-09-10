import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from datasets import DatasetDict, load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import HfArgumentParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompt templates
NEW_PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', the corresponding description is:
"""

@dataclass
class BatchAPIArguments:
    """Arguments related to the OpenAI Batch API configuration."""
    api_key: str = field(metadata={"help": "OpenAI API key"})
    model: str = field(default="gpt-3.5-turbo-0125", metadata={"help": "Model to use for text generation"})
    max_tokens: int = field(default=1000, metadata={"help": "Maximum number of tokens to generate"})

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "The name of the dataset to use"})
    output_dir: str = field(metadata={"help": "The output directory where the processed dataset will be saved"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use"})
    dataset_split_name: Optional[str] = field(default=None, metadata={"help": "The split name of the dataset to use"})
    dataset_cache_dir: Optional[str] = field(default=None, metadata={"help": "Path to cache directory for saving and loading datasets"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples"})
    push_to_hub: bool = field(default=False, metadata={"help": "Whether or not to push the processed dataset to the Hub"})
    hub_dataset_id: Optional[str] = field(default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`"})

def prepare_batch_input(dataset, batch_api_args: BatchAPIArguments) -> str:
    """Prepare the batch input file for the OpenAI Batch API."""
    batch_input = []
    for idx, sample in enumerate(dataset):
        prompt = NEW_PROMPT
        for key in ["gender", "reverberation", "sdr_noise", "speech_monotony", "speaking_rate", "pitch"]:
            if key in sample:
                prompt = prompt.replace(f"[{key}]", str(sample[key]))
            else:
                logger.warning(f"Key '{key}' not found in sample {idx}")
        
        batch_request = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": batch_api_args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": batch_api_args.max_tokens
            }
        }
        batch_input.append(json.dumps(batch_request))
    
    return "\n".join(batch_input)

def wait_for_batch_completion(client: OpenAI, batch_id: str, check_interval: int = 60) -> Dict[str, Any]:
    """Wait for the batch to complete, with a progress bar."""
    batch = client.batches.retrieve(batch_id)
    pbar = tqdm(total=100, desc="Batch Progress")
    last_completed = 0

    while batch.status not in ["completed", "failed", "expired"]:
        time.sleep(check_interval)
        batch = client.batches.retrieve(batch_id)
        completed_percentage = (batch.request_counts.completed / batch.request_counts.total) * 100
        pbar.update(completed_percentage - last_completed)
        last_completed = completed_percentage

    pbar.close()
    return batch

def main():
    parser = HfArgumentParser((BatchAPIArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        batch_api_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        batch_api_args, data_args = parser.parse_args_into_dataclasses()

    client = OpenAI(api_key=batch_api_args.api_key)

    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.dataset_split_name,
        cache_dir=data_args.dataset_cache_dir,
    )

    if data_args.max_eval_samples:
        logger.info(f"Limiting dataset to {data_args.max_eval_samples} samples")
        raw_datasets = raw_datasets.select(range(data_args.max_eval_samples))

    # Prepare batch input
    logger.info("Preparing batch input")
    batch_input_content = prepare_batch_input(raw_datasets, batch_api_args)

    # Upload batch input file
    logger.info("Uploading batch input file")
    with open("batch_input.jsonl", "w") as f:
        f.write(batch_input_content)

    batch_input_file = client.files.create(
        file=open("batch_input.jsonl", "rb"),
        purpose="batch"
    )

    # Create batch
    logger.info("Creating batch")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # Wait for batch completion
    logger.info(f"Waiting for batch {batch.id} to complete")
    batch = wait_for_batch_completion(client, batch.id)

    if batch.status != "completed":
        logger.error(f"Batch failed or expired. Status: {batch.status}")
        return

    # Retrieve results
    logger.info("Retrieving batch results")
    output_file_content = client.files.content(batch.output_file_id)
    output_lines = output_file_content.text.strip().split("\n")

    # Process results
    logger.info("Processing batch results")
    text_descriptions = []
    for line in output_lines:
        result = json.loads(line)
        text_descriptions.append(result["response"]["body"]["choices"][0]["message"]["content"])

    # Add results to dataset
    logger.info("Adding results to dataset")
    processed_dataset = raw_datasets.add_column("text_description", text_descriptions)

    # Save processed dataset
    logger.info(f"Saving processed dataset to {data_args.output_dir}")
    processed_dataset.save_to_disk(data_args.output_dir)

    if data_args.push_to_hub:
        logger.info(f"Pushing dataset to the Hub: {data_args.hub_dataset_id}")
        processed_dataset.push_to_hub(data_args.hub_dataset_id)

    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()