import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import zipfile
import shutil
import subprocess

def process_images(input_dir, output_dir, model_path):
    """Process all images in input directory and save masks to output directory"""
    
    # Create masks directory
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get all jpg files
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        mask_name = img_file.replace('.jpg', '.png')
        mask_path = os.path.join(masks_dir, mask_name)
        
        # Run inference
        cmd = [
            'python', 'run.py',
            '--input', img_path,
            '--output', mask_path,
            '--weights', model_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error processing {img_file}: {result.stderr}")
            raise RuntimeError(f"Failed to process {img_file}")

def prepare_solution_folder(output_dir, model_path):
    """Prepare solution folder for test phase"""
    
    solution_dir = os.path.join(output_dir, 'solution')
    os.makedirs(solution_dir, exist_ok=True)
    
    # Copy model checkpoint
    shutil.copy(model_path, os.path.join(solution_dir, 'ckpt.pth'))
    
    # Copy required files
    files_to_copy = ['requirements.txt', 'run.py', 'model.py', 'model_utils.py']
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, solution_dir)
        else:
            raise FileNotFoundError(f"Required file {file} not found")
    
    print(f"Solution files copied to {solution_dir}")

def create_submission_zip(output_dir, mode):
    """Create final submission zip file"""
    
    zip_name = f"submission_{mode}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add masks
        masks_dir = os.path.join(output_dir, 'masks')
        for root, dirs, files in os.walk(masks_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
        
        # Add solution folder for test mode
        if mode == 'test':
            solution_dir = os.path.join(output_dir, 'solution')
            for root, dirs, files in os.walk(solution_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
    
    print(f"Created {zip_path}")
    return zip_path

def main():
    parser = argparse.ArgumentParser(description='Prepare submission for face parsing challenge')
    parser.add_argument('mode', choices=['dev', 'test'], help='Submission mode')
    parser.add_argument('output_dir', help='Output directory for submission')
    parser.add_argument('--model_path', default='ckpt.pth', help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Determine input directory
    if args.mode == 'dev':
        input_dir = '../dev-public/test/images'
    else:
        input_dir = '../test/images'
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    process_images(input_dir, args.output_dir, args.model_path)
    
    # Prepare solution folder for test mode
    if args.mode == 'test':
        prepare_solution_folder(args.output_dir, args.model_path)
    
    # Create submission zip
    zip_path = create_submission_zip(args.output_dir, args.mode)
    
    print(f"\nSubmission preparation complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"Submission zip: {zip_path}")

if __name__ == "__main__":
    main()