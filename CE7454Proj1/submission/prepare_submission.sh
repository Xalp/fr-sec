#!/bin/bash

# Usage: ./prepare_submission.sh <mode> <output_dir> [model_path]
# mode: dev or test
# output_dir: where to save the submission
# model_path: path to best model checkpoint (optional, defaults to ckpt.pth)

MODE=$1
OUTPUT_DIR=$2
MODEL_PATH=${3:-ckpt.pth}

if [ -z "$MODE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: ./prepare_submission.sh <mode> <output_dir> [model_path]"
    echo "  mode: dev or test"
    echo "  output_dir: where to save the submission"
    echo "  model_path: path to model checkpoint (optional, defaults to ckpt.pth)"
    exit 1
fi

if [ "$MODE" != "dev" ] && [ "$MODE" != "test" ]; then
    echo "Error: mode must be 'dev' or 'test'"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p "$OUTPUT_DIR/masks"

if [ "$MODE" == "test" ]; then
    mkdir -p "$OUTPUT_DIR/solution"
fi

# Get the absolute path to the submission directory
SUBMISSION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Process images
echo "Processing images..."
if [ "$MODE" == "dev" ]; then
    INPUT_DIR="../dev-public/test/images"
else
    INPUT_DIR="../test/images"
fi

# Count total images
TOTAL_IMAGES=$(ls $INPUT_DIR/*.jpg 2>/dev/null | wc -l)
if [ $TOTAL_IMAGES -eq 0 ]; then
    echo "Error: No images found in $INPUT_DIR"
    exit 1
fi

echo "Found $TOTAL_IMAGES images to process"

# Process each image
COUNTER=0
for img in $INPUT_DIR/*.jpg; do
    COUNTER=$((COUNTER + 1))
    BASENAME=$(basename "$img" .jpg)
    echo "[$COUNTER/$TOTAL_IMAGES] Processing $BASENAME..."
    
    python "$SUBMISSION_DIR/run.py" \
        --input "$img" \
        --output "$OUTPUT_DIR/masks/${BASENAME}.png" \
        --weights "$MODEL_PATH"
    
    if [ $? -ne 0 ]; then
        echo "Error processing $img"
        exit 1
    fi
done

# Copy solution files for test phase
if [ "$MODE" == "test" ]; then
    echo "Preparing solution files..."
    
    # Copy model checkpoint
    if [ -f "$MODEL_PATH" ]; then
        cp "$MODEL_PATH" "$OUTPUT_DIR/solution/ckpt.pth"
    else
        echo "Error: Model checkpoint not found at $MODEL_PATH"
        exit 1
    fi
    
    # Copy required files
    cp "$SUBMISSION_DIR/requirements.txt" "$OUTPUT_DIR/solution/"
    cp "$SUBMISSION_DIR/run.py" "$OUTPUT_DIR/solution/"
    cp "$SUBMISSION_DIR/model.py" "$OUTPUT_DIR/solution/"
    cp "$SUBMISSION_DIR/model_utils.py" "$OUTPUT_DIR/solution/"
    
    echo "Solution files copied to $OUTPUT_DIR/solution/"
fi

# Create zip file
echo "Creating submission zip..."
cd "$OUTPUT_DIR"
if [ "$MODE" == "dev" ]; then
    zip -r submission_dev.zip masks/
    echo "Created submission_dev.zip"
else
    zip -r submission_test.zip masks/ solution/
    echo "Created submission_test.zip"
fi

echo "Submission preparation complete!"
echo "Output saved to: $OUTPUT_DIR"