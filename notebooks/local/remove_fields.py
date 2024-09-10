from datasets import load_dataset, Dataset

def remove_columns(input_path, output_path, columns_to_remove):
    # Load the dataset
    dataset = load_dataset('arrow', data_files=input_path, split='train')
    
    # Remove the specified columns
    new_dataset = dataset.remove_columns(columns_to_remove)
    
    # Save the modified dataset
    new_dataset.save_to_disk(output_path)

    print(f"Columns {columns_to_remove} have been removed. New dataset saved to {output_path}")

# Usage
input_path = '/Users/paul-henri/Documents/QS/dataspeech/datasets/temp/data/data-00000-of-00011.arrow'
output_path = '/Users/paul-henri/Documents/QS/dataspeech/datasets/mls_french_opus_generated_modified'
columns_to_remove = ['stoi', 'si-sdr', 'pesq']

remove_columns(input_path, output_path, columns_to_remove)