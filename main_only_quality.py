from datasets import load_dataset, Audio, concatenate_datasets
from multiprocess import set_start_method
from dataspeech import squim_apply
import torch
import argparse

def process_audio_dataset(dataset, args):
    print("Compute SI-SDR, PESQ, STOI")
    return dataset.map(
        squim_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count() > 0 else False,
        num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_squim if torch.cuda.device_count() > 0 else args.cpu_num_workers,
        remove_columns=[args.audio_column_name],  # tricks to avoid rewriting audio
        fn_kwargs={"audio_column_name": args.audio_column_name},
    )

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("audio_dataset_name", type=str, help="Path or name of the dataset with audio column.")
    parser.add_argument("target_dataset_name", type=str, help="Path or name of the dataset to add computed metrics to.")
    parser.add_argument("--audio_configuration", default=None, type=str, help="Configuration for the audio dataset, if necessary.")
    parser.add_argument("--target_configuration", default=None, type=str, help="Configuration for the target dataset, if necessary.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Column name of the audio column to be processed.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--batch_size", default=2, type=int, help="This parameters specify how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int, help="Number of workers per GPU for the SI-SNR, STOI and PESQ estimation if GPUs are available.")
    parser.add_argument("--hub_token", default=None, type=str, help="Hugging Face API token.")

    args = parser.parse_args()
    
    # Load the audio dataset
    if args.audio_configuration:
        audio_dataset = load_dataset(args.audio_dataset_name, args.audio_configuration, num_proc=args.cpu_num_workers)
    else:
        audio_dataset = load_dataset(args.audio_dataset_name, num_proc=args.cpu_num_workers)

    # Load the target dataset
    if args.target_configuration:
        target_dataset = load_dataset(args.target_dataset_name, args.target_configuration, num_proc=args.cpu_num_workers)
    else:
        target_dataset = load_dataset(args.target_dataset_name, num_proc=args.cpu_num_workers)

    # Process the audio dataset
    processed_audio_dataset = process_audio_dataset(audio_dataset, args)

    # Merge the computed metrics into the target dataset
    for split in target_dataset.keys():
        if split in processed_audio_dataset:
            new_columns = {
                "stoi": processed_audio_dataset[split]["stoi"],
                "si-sdr": processed_audio_dataset[split]["sdr"],
                "pesq": processed_audio_dataset[split]["pesq"]
            }
            target_dataset[split] = target_dataset[split].add_columns(new_columns)

    if args.output_dir:
        print("Saving to disk...")
        target_dataset.save_to_disk(args.output_dir)
    if args.repo_id:
        print("Pushing to the hub...")
        if args.target_configuration:
            target_dataset.push_to_hub(args.repo_id, args.target_configuration, token=args.hub_token)
        else:
            target_dataset.push_to_hub(args.repo_id, token=args.hub_token)