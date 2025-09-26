import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_hrnet_lite import HRNetLiteFaceParser


def load_model(weights_path, device):
    model = HRNetLiteFaceParser(n_classes=19)
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Training checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict format
        model.load_state_dict(checkpoint)
    
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


def get_segmentation(frame, mask, normalization_params=None, ignore_idx=255, alpha=0.4):
    PALETTE = np.array([[i, i, i] for i in range(256)])
    PALETTE[:16] = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [191, 0, 0],
        [64, 128, 0],
        [191, 128, 0],
        [64, 0, 128],
        [191, 0, 128],
        [64, 128, 128],
        [191, 128, 128],
    ])

    mask = mask.cpu().numpy()
    if frame is None:
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.putpalette(PALETTE.reshape(-1).tolist())
        return mask


def postprocess_output(output):
    # Convert model output to single-channel mask
    # output shape: [1, 19, 512, 512]
    pred = torch.argmax(output, dim=1)[0]  # [512, 512]
    
    # Use get_segmentation to create the mask with palette
    mask = get_segmentation(None, pred, None)
    
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output mask')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--tta', action='store_true', help='Use test time augmentation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.weights, device)
    
    # Process image
    image_tensor = preprocess_image(args.input).to(device)
    
    # Run inference
    with torch.no_grad():
        if args.tta:
            # Test time augmentation - horizontal flip
            output1, _, _ = model(image_tensor)
            output2, _, _ = model(torch.flip(image_tensor, dims=[3]))
            output2 = torch.flip(output2, dims=[3])
            output = (output1 + output2) / 2
        else:
            output, _, _ = model(image_tensor)
    
    # Post-process and save
    mask = postprocess_output(output)
    
    # Save the mask (already has palette set by get_segmentation)
    mask.save(args.output)
    
    print(f"Mask saved to {args.output}")


if __name__ == "__main__":
    main()