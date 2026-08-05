#!/bin/bash
set -e

DATA_DIR="data/cubicasa5k"
ZIP_FILE="data/cubicasa5k.zip"
URL="https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1"

echo "Downloading CubiCasa5k (~4GB)..."
if command -v wget &> /dev/null; then
    wget -c -O "$ZIP_FILE" "$URL"
else
    curl -L -C - -o "$ZIP_FILE" "$URL"
fi

echo "Extracting..."
mkdir -p "$DATA_DIR"
unzip -q "$ZIP_FILE" -d "$DATA_DIR"
rm "$ZIP_FILE"

echo "Done! Dataset at: $DATA_DIR"
echo "Samples:"
find "$DATA_DIR" -maxdepth 2 -type d | head -10