from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tqdm import tqdm
import asyncio
import aiohttp
from typing import List, Dict
import re

PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (e.g., male, female)
2. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
3. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
4. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
5. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)
6. The pitch of the speaker's voice (e.g., very low pitch, quite low pitch, slightly low pitch, moderate pitch, slightly high pitch, quite high pitch, very high pitch)
Your task is to create a text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.
For example, given the following keywords: 'female', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'slightly low pitch', 'very slowly', a valid description would be: 'a woman with a deep voice speaks slowly but has an animated delivery in an echoey room with some background noise'.
For the keywords: '[gender]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]', the corresponding description is:"""

@dataclass
class DatasetArguments:
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    openai_api_key: str = field(
        default=None,
        metadata={"help": "OpenAI API key"},
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "Output directory for Arrow files"},
    )
    batch_size: int = field(
        default=100,
        metadata={"help": "Number of concurrent requests to send to OpenAI API"},
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "Whether to run in test mode (process only first 200 lines)"},
    )
    repo_id: str = field(
        default=None,
        metadata={"help": "Repository ID to push the dataset to the Hugging Face Hub"},
    )
    language: str = field(
        default=None,
        metadata={"help": "Language of the speaker to be included in the description"},
    )

def create_prompt(example: Dict[str, str], expected_columns: List[str]) -> str:
    prompt = PROMPT
    for column in expected_columns:
        if column in example:
            prompt = prompt.replace(f"[{column}]", example[column])
    return prompt

def replace_gender_with_language(text: str, language: str) -> str:
    def replace_word(match):
        word = match.group(0)
        if word.lower().startswith(language.lower()):
            return word
        return f'{language} {word.lstrip("a ").lstrip("an ")}'

    pattern = r'\b(a |an )?(man|woman|male|female)\b'
    return re.sub(pattern, replace_word, text, flags=re.IGNORECASE)

async def process_batch(session: aiohttp.ClientSession, batch: List[str], api_key: str, language: str) -> List[str]:
    async def process_prompt(prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an AI assistant that analyzes audio characteristics."},
                {"role": "user", "content": prompt}
            ]
        }
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            description = result['choices'][0]['message']['content']
            return replace_gender_with_language(description, language)

    tasks = [process_prompt(prompt) for prompt in batch]
    return await asyncio.gather(*tasks)

async def process_split(split_name: str, split_dataset: Dataset, data_args: DatasetArguments) -> Dataset:
    # Create the prompts
    EXPECTED_COLUMNS = {"gender", "pitch", "noise", "reverberation", "speech_monotony", "speaking_rate"}
    prompts = [create_prompt(example, EXPECTED_COLUMNS) for example in split_dataset]

    # Send the prompts to the OpenAI API in batches
    results = []
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(prompts), data_args.batch_size), desc=f"Processing {split_name}"):
            batch = prompts[i:i+data_args.batch_size]
            batch_results = await process_batch(session, batch, data_args.openai_api_key, data_args.language)
            results.extend(batch_results)

    # Add the new column to the dataset
    return split_dataset.add_column("text_description", results)

async def main():
    parser = HfArgumentParser((DatasetArguments))
    data_args = parser.parse_args_into_dataclasses()[0]

    if data_args.language is None:
        raise ValueError("The 'language' argument must be provided.")

    # 1. Load the dataset
    dataset = load_dataset(data_args.dataset_name)

    # 2. Limit to first 200 lines if in test mode
    if data_args.test_mode:
        dataset = dataset.select(range(min(200, len(dataset))))
        print("Running in test mode: processing only the first 200 lines (or less if the dataset is smaller)")

    # 3. Process each split
    new_dataset = DatasetDict()
    for split_name, split_dataset in dataset.items():
        new_split = await process_split(split_name, split_dataset, data_args)
        new_dataset[split_name] = new_split

    # 4. Save the results as Arrow files
    new_dataset.save_to_disk(data_args.output_dir)
    print(f"Results saved to {data_args.output_dir}")

    # Option to send it to the hub
    if data_args.repo_id:
        new_dataset.push_to_hub(data_args.repo_id)
        print(f"Results pushed to the Hugging Face Hub: {data_args.repo_id}")

if __name__ == "__main__":
    asyncio.run(main())