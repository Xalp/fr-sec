import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import AttentionUNet


def load_model(weights_path, device):
    model = AttentionUNet(feature_scale=8, n_classes=19)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to 512x512 if needed
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.BILINEAR)
    
    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def postprocess_output(output):
    # Convert model output to single-channel mask
    # output shape: [1, 19, 512, 512]
    pred = torch.argmax(output, dim=1)[0]  # [512, 512]
    
    # Convert to numpy and ensure uint8 type
    mask = pred.cpu().numpy().astype(np.uint8)
    
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output mask')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.weights, device)
    
    # Process image
    image_tensor = preprocess_image(args.input).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Post-process and save
    mask = postprocess_output(output)
    
    # Save as single-channel PNG
    mask_image = Image.fromarray(mask, mode='P')
    mask_image.save(args.output)
    
    print(f"Mask saved to {args.output}")


if __name__ == "__main__":
    main()