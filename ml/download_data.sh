#!/bin/bash
# Create data directory if it doesn't exist
mkdir -p data

# Check if dataset is already downloaded
if [ -f "data/data.zip" ]; then
    echo "data.zip already exists. Skipping download."
else
    echo "Downloading dataset..."
    curl -L -o data/data.zip \
      https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data
fi

# Check for repair mode
if [ "$1" == "--repair" ]; then
    echo "Repair mode: Overwriting and repairing any corrupted or partially unzipped files..."
    unzip -q -o data/data.zip -d data/
    echo "Repair and extraction complete."
    exit 0
fi

# Skip extraction if it already appears unzipped
if [ -f "data/Data_Entry_2017.csv" ]; then
    echo "Dataset appears to already be extracted. Skipping unzipping."
    echo "Note: If you have corrupted image files, run this script with: ./download_data.sh --repair"
else
    echo "Unzipping dataset..."
    unzip -q -n data/data.zip -d data/
    echo "Download and extraction complete."
fi
