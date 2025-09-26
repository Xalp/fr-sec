import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from model import AttentionUNet, count_parameters
from dataset import get_dataloader
import argparse


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        dice = 0.0
        
        for i in range(num_classes):
            pred_i = pred[:, i, :, :]
            target_i = (target == i).float()
            
            intersection = (pred_i * target_i).sum()
            dice += (2. * intersection + self.smooth) / (pred_i.sum() + target_i.sum() + self.smooth)
        
        return 1 - dice / num_classes


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, deep_supervision=False):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.deep_supervision = deep_supervision
        
    def forward(self, pred, target):
        if self.deep_supervision and isinstance(pred, tuple):
            # Main output
            main_pred = pred[0]
            ce = self.ce_loss(main_pred, target)
            dice = self.dice_loss(main_pred, target)
            loss = self.ce_weight * ce + self.dice_weight * dice
            
            # Deep supervision losses with decreasing weights
            weights = [0.4, 0.3, 0.2]  # for dsv4, dsv3, dsv2
            for i, (aux_pred, w) in enumerate(zip(pred[1:], weights)):
                aux_ce = self.ce_loss(aux_pred, target)
                aux_dice = self.dice_loss(aux_pred, target)
                loss += w * (self.ce_weight * aux_ce + self.dice_weight * aux_dice)
            
            return loss
        else:
            # Single output
            ce = self.ce_loss(pred, target)
            dice = self.dice_loss(pred, target)
            return self.ce_weight * ce + self.dice_weight * dice


def train_epoch(model, dataloader, optimizer, criterion, scaler, scheduler, device):
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler per batch for OneCycleLR
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with reduced feature_scale to increase parameters
    model = AttentionUNet(feature_scale=args.feature_scale, n_classes=19, use_deep_supervision=args.deep_supervision).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params > 1821085:
        print(f"WARNING: Model exceeds parameter limit ({num_params} > 1,821,085)")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloader(args.data_dir, args.batch_size, args.num_workers)
    
    # Loss and optimizer
    criterion = CombinedLoss(ce_weight=0.9, dice_weight=0.1, deep_supervision=args.deep_supervision)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    
    # OneCycleLR scheduler with 1 epoch warmup
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=1.0/args.epochs,  # 1 epoch warmup
        div_factor=25,
        final_div_factor=1000,
    )
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print("Saved best model!")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'latest_model.pth')
    
    # Save final model as ckpt.pth for submission
    torch.save(model.state_dict(), 'ckpt.pth')
    print("\nTraining complete! Model saved as ckpt.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--feature_scale', type=float, default=4.15, help='Feature scale for model size')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision')
    
    args = parser.parse_args()
    main(args)