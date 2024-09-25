import logging
from datasets import load_dataset, get_dataset_config_names
from accelerate import Accelerator
import argparse
import os
from tqdm import tqdm

def download_and_save_dataset(dataset_name, save_directory, configs=None, split=None, streaming=False):
    logging.info(f"Downloading dataset: {dataset_name}")
    
    accelerator = Accelerator()
    
    if configs is None or configs == ["all"]:
        configs = get_dataset_config_names(dataset_name)
        logging.info(f"Downloading all configurations: {configs}")
    else:
        logging.info(f"Downloading specified configurations: {configs}")
    
    for config in tqdm(configs, desc="Configurations"):
        config_save_dir = os.path.join(save_directory, config)
        logging.info(f"Processing configuration: {config}")
        
        with accelerator.local_main_process_first():
            try:
                dataset = load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    streaming=streaming,
                )
                
                if streaming:
                    # For streaming datasets, we need to materialize it first
                    dataset = dataset.take(len(dataset))
                
                logging.info(f"Saving dataset configuration {config} to: {config_save_dir}")
                dataset.save_to_disk(config_save_dir)
                logging.info(f"Dataset configuration {config} saved successfully")
            except Exception as e:
                logging.error(f"Error processing configuration {config}: {str(e)}")
    
    logging.info(f"All specified configurations for {dataset_name} have been processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to download")
    parser.add_argument("save_directory", type=str, help="Directory to save the dataset")
    parser.add_argument("--configs", nargs='+', default=["all"], help="Configuration names for the dataset (use 'all' for all configs)")
    parser.add_argument("--split", type=str, default=None, help="Split of the dataset to download")
    parser.add_argument("--streaming", action="store_true", help="Whether to use streaming mode")
    
    args = parser.parse_args()

    # Set up logging
    log_file = f"download_log_{args.dataset_name.replace('/', '_')}_{os.path.basename(args.save_directory)}.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    download_and_save_dataset(args.dataset_name, args.save_directory, args.configs, args.split, args.streaming)
    