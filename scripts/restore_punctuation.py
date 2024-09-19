from datasets import load_dataset
from deepmultilingualpunctuation import PunctuationModel
import argparse
import torch
from multiprocessing import set_start_method

def apply_punctuation(examples, model, text_column):
    restored_texts = [model.restore_punctuation(text) for text in examples[text_column]]
    return {
        f"original_{text_column}": examples[text_column],
        text_column: restored_texts
    }

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset. See: https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/loading_methods#datasets.load_dataset.path")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--hub_token", default=None, type=str, help="Hugging Face API token.")
    parser.add_argument("--text_column", default="text", type=str, help="Name of the column containing the text to be punctuated.")
    parser.add_argument("--language", default=None, type=str, help="Language of the dataset. If not specified, uses the default multilingual model.")
    parser.add_argument("--batch_size", default=32, type=int, help="This parameter specifies how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--num_workers_per_gpu", default=1, type=int, help="Number of workers per GPU if GPUs are available. Defaults to 1 if some are available. Useful if you want multiple processes per GPUs to maximise GPU usage.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    # Load the dataset
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.cpu_num_workers)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers)

    # Initialize the punctuation model
    if args.language and args.language.lower() not in {"english", "italian", "french", "german", "dutch", "catalan"}:
        raise ValueError(f"Language {args.language} is not supported. Please choose from: english, italian, french, german, dutch, catalan")
    elif args.language and args.language.lower() == "catalan":    
        model = PunctuationModel(model=f"softcatala/fullstop-catalan-punctuation-prediction")
    elif args.language and args.language.lower() == "dutch":
        model = PunctuationModel(model="oliverguhr/fullstop-dutch-punctuation-prediction")
    else:
        model = PunctuationModel(model=f"oliverguhr/fullstop-punctuation-multilang-large")

    # Determine the number of GPUs and workers
    num_gpus = torch.cuda.device_count() if args.use_gpu else 0
    num_workers = num_gpus * args.num_workers_per_gpu if num_gpus > 0 else args.cpu_num_workers

    print(f"Using {'GPU' if num_gpus > 0 else 'CPU'} with {num_workers} workers")

    # Apply punctuation restoration
    punctuated_dataset = dataset.map(
        apply_punctuation,
        batched=True,
        batch_size=args.batch_size,
        num_proc=num_workers,
        with_rank=True if num_gpus > 0 else False,
        fn_kwargs={"model": model, "text_column": args.text_column},
        desc="Restoring punctuation"
    )

    # Save or push the dataset
    if args.output_dir:
        print("Saving to disk...")
        punctuated_dataset.save_to_disk(args.output_dir)
    
    if args.repo_id:
        print("Pushing to the hub...")
        if args.configuration:
            punctuated_dataset.push_to_hub(args.repo_id, args.configuration, token=args.hub_token)
        else:
            punctuated_dataset.push_to_hub(args.repo_id, token=args.hub_token)

    print("Punctuation restoration completed.")