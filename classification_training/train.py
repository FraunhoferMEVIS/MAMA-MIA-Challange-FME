import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataloader import NiftiImageDataset
from augmentations import random_mirroring, batch_generators_intensity_augmentations

def parse_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def validate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def train_model(config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True if torch.cuda.is_available() else False

    data_augmentation_transforms = [random_mirroring, batch_generators_intensity_augmentations]


    train_dataset = NiftiImageDataset(
        data_dir=os.path.join(config['data_path']),
        data_split_file=config['data_split_file'],
        group='training',
        target_size=config['target_size'],
        transforms=data_augmentation_transforms
    )

    val_dataset = NiftiImageDataset(
        data_dir=os.path.join(config['data_path']),
        data_split_file=config['data_split_file'],
        group='validation',
        target_size=config['target_size']
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              num_workers=6)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            pin_memory=True,
                            num_workers=6)

    # Model
    weights = Swin3D_T_Weights.KINETICS400_V1
    model = swin3d_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, 2)  # 2 labels
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float('inf')

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'loss_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, config['epochs']+1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = validate(model, val_loader, device, criterion)
        scheduler.step()

        # Log
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

def main():
    parser = argparse.ArgumentParser(description="Train Swin3D on NIfTI dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save logs and models")
    args = parser.parse_args()

    config = parse_config(args.config)
    train_model(config, args.output_dir)

if __name__ == '__main__':
    main()
