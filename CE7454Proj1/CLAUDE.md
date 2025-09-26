# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a face parsing challenge project based on the CelebAMask-HQ dataset. The goal is to design and train a face parsing network that assigns pixel-wise labels for semantic face components (eyes, nose, mouth, etc.).

## Project Structure

- `train/` - Training dataset with 1000 image pairs
- `test/` - Test dataset images  
- `submission/` - Directory for submission files
- `readme.md` - Challenge description and requirements

## Key Requirements and Constraints

1. **Model Constraints**: Model must have fewer than 1,821,085 trainable parameters
2. **Data Constraints**: Only use the provided 1000 training image pairs - no external data or pretrained models allowed
3. **No Ensemble**: Single model only, no ensemble of models

## Submission Requirements

### Directory Structure
```
submission.zip
├── masks/
│   ├── [filename].png  # Single-channel mask images matching input filenames
│   └── ...
└── solution/  # INCLUDE DURING TEST PHASE ONLY
    ├── ckpt.pth         # Model checkpoint (REQUIRED)
    ├── requirements.txt  # Python dependencies (REQUIRED)
    └── run.py           # Inference script (REQUIRED)
```

### Inference Command
The solution must be invokable with:
```bash
pip install -r requirements.txt
python3 run.py --input /path/to/input-image.jpg --output /path/to/output-mask.png --weights ckpt.pth
```

## Important Notes

- Output masks must be SINGLE-CHANNEL images, not RGB
- Maximum 10 submissions allowed in test phase
- Use matric number as display name
- The code is meant to be run on a remote server with 8 A100-80G GPUs (per user's global instructions)

## Evaluation

F-Score metric is used to measure similarity between predicted and ground truth masks using the provided `compute_multiclass_fscore` function.