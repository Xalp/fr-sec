import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from model_hrnet_lite import HRNetLiteFaceParser, count_parameters
from dataset import get_dataloader
import argparse
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage


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


class TVLoss(nn.Module):
    """Total Variation Loss for smoothness"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred):
        # pred shape: [B, C, H, W]
        pred_prob = F.softmax(pred, dim=1)
        
        # Compute differences
        diff_h = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :])
        diff_w = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1])
        
        return diff_h.mean() + diff_w.mean()


def get_edge_from_mask(mask):
    """Extract edges from segmentation mask using Sobel filter"""
    edges = []
    for m in mask:
        # Apply Sobel filter
        sx = ndimage.sobel(m.cpu().numpy(), axis=0)
        sy = ndimage.sobel(m.cpu().numpy(), axis=1)
        edge = np.hypot(sx, sy)
        edge = (edge > 0).astype(np.float32)
        edges.append(edge)
    return torch.from_numpy(np.array(edges)).to(mask.device)


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, aux_weight=0.4, edge_weight=0.2, tv_weight=2e-4):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.edge_loss = nn.BCELoss()
        self.tv_loss = TVLoss()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.tv_weight = tv_weight
        
    def forward(self, pred, target, edge_pred=None, aux_pred=None):
        # Main segmentation loss
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        main_loss = self.ce_weight * ce + self.dice_weight * dice
        
        # TV loss for smoothness
        tv = self.tv_loss(pred)
        main_loss += self.tv_weight * tv
        
        total_loss = main_loss
        
        # Edge loss
        if edge_pred is not None:
            edge_gt = get_edge_from_mask(target).unsqueeze(1)
            edge_loss = self.edge_loss(edge_pred, edge_gt)
            total_loss += self.edge_weight * edge_loss
        
        # Auxiliary loss
        if aux_pred is not None:
            aux_ce = self.ce_loss(aux_pred, target)
            total_loss += self.aux_weight * aux_ce
        
        return total_loss


def train_epoch(model, dataloader, optimizer, criterion, scaler, scheduler, device, resolution=512):
    model.train()
    total_loss = 0
    
    # Resolution-aware transform
    if resolution != 512:
        resize = transforms.Resize((resolution, resolution), transforms.InterpolationMode.BILINEAR)
        resize_mask = transforms.Resize((resolution, resolution), transforms.InterpolationMode.NEAREST)
    else:
        resize = resize_mask = None
    
    for images, masks in tqdm(dataloader, desc=f'Training {resolution}x{resolution}'):
        images = images.to(device)
        masks = masks.to(device)
        
        # Apply resolution curriculum
        if resize is not None:
            images = resize(images)
            masks = resize_mask(masks.unsqueeze(1)).squeeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs, edge, aux = model(images)
            loss = criterion(outputs, masks, edge, aux)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler per batch
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
            
            outputs, edge, aux = model(images)
            loss = criterion(outputs, masks, edge, aux)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = HRNetLiteFaceParser(n_classes=19).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    if num_params > 1821085:
        print(f"WARNING: Model exceeds parameter limit ({num_params} > 1,821,085)")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloader(args.data_dir, args.batch_size, args.num_workers)
    
    # Loss and optimizer
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0, aux_weight=0.4, edge_weight=0.2, tv_weight=2e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine scheduler with warmup
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * 5  # 5 epoch warmup
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
    )
    
    scaler = GradScaler()
    
    # EMA
    if args.ema:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    
    # Resolution curriculum
    resolution_schedule = [
        (0, 30, 256),    # epochs 0-30: 256x256
        (30, 70, 384),   # epochs 30-70: 384x384
        (70, args.epochs, 512),  # epochs 70+: 512x512
    ]
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Determine current resolution
        current_res = 512
        for start_ep, end_ep, res in resolution_schedule:
            if start_ep <= epoch < end_ep:
                current_res = res
                break
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, current_res)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Update EMA
        if args.ema:
            ema.update()
        
        # Validate
        if args.ema:
            with ema.average_parameters():
                val_loss = validate(model, val_loader, criterion, device)
        else:
            val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            state_dict = model.state_dict()
            if args.ema:
                with ema.average_parameters():
                    state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model_hrnet.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print("Early stopping triggered!")
                break
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'latest_model_hrnet.pth')
    
    # Save final model as ckpt.pth for submission
    final_state = model.state_dict()
    if args.ema:
        with ema.average_parameters():
            final_state = model.state_dict()
    torch.save(final_state, 'ckpt_hrnet.pth')
    print("\nTraining complete! Model saved as ckpt_hrnet.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=7e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--ema', action='store_true', default=True, help='Use EMA')
    
    args = parser.parse_args()
    main(args)