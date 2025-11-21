#!/bin/bash
# Create data directory if it doesn't exist
mkdir -p data

# Download the dataset
echo "Downloading dataset..."
curl -L -o data/data.zip \
  https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data

# Unzip the dataset
echo "Unzipping dataset..."
unzip -q data/data.zip -d data/

echo "Download and extraction complete."
