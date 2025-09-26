import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class CelebAMaskDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment and (split == 'train')
        
        # Get image paths
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        
        # List all images
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        
        # Data augmentation
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('P')
            mask = np.array(mask)
        else:
            # Create dummy mask for test set
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Apply augmentations
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = np.fliplr(mask).copy()
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, fill=0)
                mask = Image.fromarray(mask)
                mask = TF.rotate(mask, angle, fill=0)
                mask = np.array(mask)
            
            # Color jittering
            image = self.color_jitter(image)
            
            # Random crop and resize
            if random.random() > 0.3:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.8, 1.0), ratio=(0.95, 1.05))
                image = TF.crop(image, i, j, h, w)
                mask = Image.fromarray(mask)
                mask = TF.crop(mask, i, j, h, w)
                mask = np.array(mask)
                
                image = TF.resize(image, (512, 512), transforms.InterpolationMode.BILINEAR)
                mask = Image.fromarray(mask)
                mask = TF.resize(mask, (512, 512), transforms.InterpolationMode.NEAREST)
                mask = np.array(mask)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def get_dataloader(root_dir, batch_size, num_workers=4):
    # Training dataset with augmentation
    train_dataset = CelebAMaskDataset(root_dir, split='train', augment=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset without augmentation
    val_dataset = CelebAMaskDataset(root_dir, split='val', augment=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = CelebAMaskDataset('../', split='train', augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Unique mask values: {torch.unique(mask)}")