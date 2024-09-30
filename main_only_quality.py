from datasets import load_dataset, Audio
from multiprocess import set_start_method
from dataspeech import squim_apply
import torch
import argparse

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset.")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Column name of the audio column to be enriched.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--batch_size", default=2, type=int, help="This parameters specify how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int, help="Number of workers per GPU for the SI-SNR, STOI and PESQ estimation if GPUs are available.")

    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.cpu_num_workers)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers)

    print("Compute SI-SDR, PESQ, STOI")
    squim_dataset = dataset.map(
        squim_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count() > 0 else False,
        num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_squim if torch.cuda.device_count() > 0 else args.cpu_num_workers,
        remove_columns=[args.audio_column_name],  # tricks to avoid rewriting audio
        fn_kwargs={"audio_column_name": args.audio_column_name},
    )

    for split in dataset.keys():
        dataset[split] = dataset[split].add_column("stoi", squim_dataset[split]["stoi"])
        dataset[split] = dataset[split].add_column("si-sdr", squim_dataset[split]["sdr"])
        dataset[split] = dataset[split].add_column("pesq", squim_dataset[split]["pesq"])
    
    if args.output_dir:
        print("Saving to disk...")
        dataset.save_to_disk(args.output_dir)
    if args.repo_id:
        print("Pushing to the hub...")
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)
