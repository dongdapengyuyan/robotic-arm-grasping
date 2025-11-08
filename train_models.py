import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models import GraspingCNN
from dataset_cornell_fixed import CornellGraspDataset


class Config:
    batch_size = 8
    num_epochs = 30
    learning_rate = 0.0001
    patience = 10

    weight_pos = 1.0
    weight_angle = 3.0
    gradient_clip = 1.0

    dataset_path = "cornell_dataset/datasets/oneoneliu/cornell-grasp/versions/1/01"
    checkpoint_dir = "checkpoints"
    best_model_name = "best_cnn_model.pth"


def train_one_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    total_pos_error = 0
    total_angle_error = 0
    total_success = 0
    total_samples = 0

    all_pred_angles = []
    all_true_angles = []

    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        center_x = batch['center_x'].to(device)
        center_y = batch['center_y'].to(device)
        angle = batch['angle'].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        pred_x = outputs[:, 0]
        pred_y = outputs[:, 1]
        pred_angle = outputs[:, 2]

        loss_x = criterion(pred_x, center_x)
        loss_y = criterion(pred_y, center_y)
        loss_angle = criterion(pred_angle, angle)

        loss = config.weight_pos * (loss_x + loss_y) + config.weight_angle * loss_angle

        if torch.isnan(loss):
            print("\n⚠️  Warning: NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
        optimizer.step()

        pos_error = torch.sqrt((pred_x - center_x) ** 2 + (pred_y - center_y) ** 2)
        angle_diff = torch.abs(pred_angle - angle) * 180
        angle_error = torch.min(angle_diff, 180 - angle_diff)

        success = (angle_error < 30).float()

        total_loss += loss.item() * images.size(0)
        total_pos_error += pos_error.sum().item()
        total_angle_error += angle_error.sum().item()
        total_success += success.sum().item()
        total_samples += images.size(0)

        all_pred_angles.extend((pred_angle * 180).detach().cpu().numpy())
        all_true_angles.extend((angle * 180).detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = {
        'loss': total_loss / total_samples,
        'pos_error': (total_pos_error / total_samples) * 224,
        'angle_error': total_angle_error / total_samples,
        'success_rate': (total_success / total_samples) * 100,
        'pred_angles': all_pred_angles,
        'true_angles': all_true_angles
    }

    return metrics


def validate(model, val_loader, criterion, device, config):
    model.eval()
    total_loss = 0
    total_pos_error = 0
    total_angle_error = 0
    total_success = 0
    total_samples = 0

    all_pred_angles = []
    all_true_angles = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="  Validating", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            center_x = batch['center_x'].to(device)
            center_y = batch['center_y'].to(device)
            angle = batch['angle'].to(device)

            outputs = model(images)

            pred_x = outputs[:, 0]
            pred_y = outputs[:, 1]
            pred_angle = outputs[:, 2]

            loss_x = criterion(pred_x, center_x)
            loss_y = criterion(pred_y, center_y)
            loss_angle = criterion(pred_angle, angle)

            loss = config.weight_pos * (loss_x + loss_y) + config.weight_angle * loss_angle

            pos_error = torch.sqrt((pred_x - center_x) ** 2 + (pred_y - center_y) ** 2)
            angle_diff = torch.abs(pred_angle - angle) * 180
            angle_error = torch.min(angle_diff, 180 - angle_diff)

            success = (angle_error < 30).float()

            total_loss += loss.item() * images.size(0)
            total_pos_error += pos_error.sum().item()
            total_angle_error += angle_error.sum().item()
            total_success += success.sum().item()
            total_samples += images.size(0)

            all_pred_angles.extend((pred_angle * 180).detach().cpu().numpy())
            all_true_angles.extend((angle * 180).detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = {
        'loss': total_loss / total_samples,
        'pos_error': (total_pos_error / total_samples) * 224,
        'angle_error': total_angle_error / total_samples,
        'success_rate': (total_success / total_samples) * 100,
        'pred_angles': all_pred_angles,
        'true_angles': all_true_angles
    }

    return metrics


def print_header():
    print("=" * 70)
    print(" " * 15 + "🤖 ROBOTIC ARM GRASPING TRAINING 🤖")
    print("=" * 70)
    print()


def print_config(config, device):
    print(f"📁 Project directory: {Path.cwd()}")
    print(f"🖥️  Computing device: {device}")
    print(f"⚡ Batch size: {config.batch_size}")
    print(f"🔢 Epochs: {config.num_epochs}")
    print(f"📚 Learning rate: {config.learning_rate}")
    print(f"🎯 Position weight: {config.weight_pos}")
    print(f"🎯 Angle weight: {config.weight_angle}")
    print(f"✂️  Gradient clip: {config.gradient_clip}")
    print()


def main():
    config = Config()

    print_header()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_config(config, device)

    print("=" * 70)
    print("📥 STEP 1: Checking Cornell Grasping Dataset")
    print("=" * 70)

    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        sys.exit(1)

    print("✅ Found dataset")
    print()

    print("=" * 70)
    print("📊 STEP 2: Loading Cornell Dataset")
    print("=" * 70)

    train_dataset = CornellGraspDataset(
        root_dir=config.dataset_path,
        split='train'
    )

    val_dataset = CornellGraspDataset(
        root_dir=config.dataset_path,
        split='val'
    )

    test_dataset = CornellGraspDataset(
        root_dir=config.dataset_path,
        split='test'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"\n✅ Dataset: Cornell Real Dataset")
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"   Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    print()

    print("=" * 70)
    print("🧠 STEP 3: Training CNN Model")
    print("=" * 70)

    model = GraspingCNN().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CNN Parameters: {total_params / 1e6:.2f}M")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    print("✅ Model weights initialized")
    print()

    print("=" * 50)
    print("Training CNN Model")
    print("=" * 50)
    print()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config)
        val_metrics = validate(model, val_loader, criterion, device, config)

        print(f"\n[DEBUG] 角度预测范围: [{min(val_metrics['pred_angles']):.2f}, {max(val_metrics['pred_angles']):.2f}]")
        print(f"[DEBUG] 角度真值范围: [{min(val_metrics['true_angles']):.2f}, {max(val_metrics['true_angles']):.2f}]")
        print()

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"  Loss      - Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f} | LR: {current_lr:.7f}")
        print(f"  Pos Error - Train: {train_metrics['pos_error']:.2f}px | Val: {val_metrics['pos_error']:.2f}px")
        print(f"  Ang Error - Train: {train_metrics['angle_error']:6.2f}°  | Val: {val_metrics['angle_error']:6.2f}°")
        print(f"  Success   - Train: {train_metrics['success_rate']:6.1f}%  | Val: {val_metrics['success_rate']:6.1f}%")

        scheduler.step(val_metrics['loss'])

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'config': {
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'weight_pos': config.weight_pos,
                'weight_angle': config.weight_angle
            }
        }

        epoch_path = os.path.join(config.checkpoint_dir, f'cnn_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0

            best_path = os.path.join(config.checkpoint_dir, config.best_model_name)
            torch.save(checkpoint, best_path)
            print(f"  ✅ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        print("-" * 50)

        if epochs_no_improve >= config.patience:
            print(f"\n⚠️  Early stopping! No improvement for {config.patience} epochs.")
            break

    training_time = (time.time() - start_time) / 60
    print(f"\n✅ Training completed in {training_time:.1f} minutes")
    print()

    print("=" * 70)
    print("✨ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"📦 Best model saved to: {os.path.join(config.checkpoint_dir, config.best_model_name)}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print()
    print("🎨 Next step: Run visualization")
    print(
        f"   python visualize_grasp_predictions.py --model {os.path.join(config.checkpoint_dir, config.best_model_name)}")
    print()


if __name__ == '__main__':
    main()