#!/bin/bash

# Create a file with paths to remove
cat << EOF > files_to_remove.txt
notebooks/jenny-tts-6h/train/data-00001-of-00003.arrow
notebooks/jenny-tts-6h/train/data-00000-of-00003.arrow
notebooks/jenny-tts-6h/train/data-00002-of-00003.arrow
notebooks/local/output/train/data-00000-of-00001.arrow
notebooks/flexai/anotate_dataset.ipynb
notebooks/anotate_dataset.ipynb
notebooks/local/output.arrow/data-00000-of-00001.arrow
EOF

# Use git filter-repo to remove files
git filter-repo --force --invert-paths --paths-from-file files_to_remove.txt

# Clean up
rm files_to_remove.txt

# Force push changes
git push origin --force --all

echo "Large files have been removed from the entire Git history."
echo "Please make sure all team members clone a fresh copy of the repository."