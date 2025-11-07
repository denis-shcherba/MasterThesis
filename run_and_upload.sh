#!/bin/bash
set -e

# === Configuration ===
WORKDIR="/home/denis/git/MasterThesis"      
SCRIPT="scripts/collect_data.py"
OUTPUT_FILE="table_demo.h5"
REMOTE_DEST="hal:/home/denis/git/MasterThesis/demos"

# === Go to working directory ===
cd "$WORKDIR"

# === Function to upload after exit ===
upload_file() {
    echo "Uploading $OUTPUT_FILE to $REMOTE_DEST..."
    if scp "$OUTPUT_FILE" "$REMOTE_DEST"; then
        echo "✅ Upload successful."
    else
        echo "❌ Upload failed."
    fi
}

# === Ensure upload happens even if you stop the script manually ===
trap upload_file EXIT

# === Run the Python script ===
echo "🚀 Running data collection script..."
python3 "$SCRIPT"

echo "🎉 Python script completed successfully."
