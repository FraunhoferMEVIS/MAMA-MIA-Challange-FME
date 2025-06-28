import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np

from classification_training.augmentations import random_mirroring, random_mirroring_2d, \
    batch_generators_intensity_augmentations, batch_generators_spatial_augmentations
from classification_training.dataloader import NiftiImageDataset, NiftiImageDataset2DCenterSlice, NiftiImageDataset2DAttention
import classification_training.models as models
from classification_training.validate import validate


def parse_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def train_model(config: dict, output_dir: str) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True if torch.cuda.is_available() else False

    # Fairness evaluation settings
    selected_fairness_variables = config.get('fairness_variables', ['age', 'breast_density', 'menopausal_status'])
    print(f"Fairness variables: {selected_fairness_variables}")

    model_key = config['model_key']
    if model_key == "2.5d_attention_model":
        data_augmentation_transforms = [random_mirroring,
                                        batch_generators_intensity_augmentations,
                                        batch_generators_spatial_augmentations] 
        DatasetClass = NiftiImageDataset2DAttention
        num_slices = config['target_size'][0]
        internal_target_size_2d = (config['target_size'][1], config['target_size'][2])
        print(f"Training 2.5D model with {num_slices} slices and internal 2D size: {internal_target_size_2d}")
    elif len(config['target_size']) == 3: 
        data_augmentation_transforms = [random_mirroring,
                                        batch_generators_intensity_augmentations,
                                        batch_generators_spatial_augmentations]
        DatasetClass = NiftiImageDataset
    elif len(config['target_size']) == 2: 
        data_augmentation_transforms = [random_mirroring_2d,
                                        batch_generators_intensity_augmentations,
                                        batch_generators_spatial_augmentations]
        DatasetClass = NiftiImageDataset2DCenterSlice
    else:
        raise ValueError("Invalid 'target_size' in config. Must be 2 or 3 dimensions for the dataset, or specifically configured for 2.5D.")

    dataset_common_args = {
        'data_dir': os.path.join(config['data_path']),
        'data_split_file': config['data_split_file'],
        'target_size': config['target_size'],
        'normalization': config['normalization']
    }

    if model_key == "2.5d_attention_model":
        train_dataset = DatasetClass(
            group='training',
            transforms=data_augmentation_transforms,
            num_slices=num_slices,
            **dataset_common_args
        )
        val_dataset = DatasetClass(
            group='validation',
            num_slices=num_slices,
            **dataset_common_args
        )
    else:
        train_dataset = DatasetClass(
            group='training',
            transforms=data_augmentation_transforms,
            **dataset_common_args
        )
        val_dataset = DatasetClass(
            group='validation',
            **dataset_common_args
        )

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config['num_workers'],
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config['num_workers'],
                            persistent_workers=True)

    if model_key == "2.5d_attention_model":
        model = models.get_model(
            model_key,
            pretrained=config['pretrained'],
            encoder_key=config['encoder_key'], 
            attention_type=config['attention_type']
        )
    else:
        model = models.get_model(model_key, pretrained=config['pretrained'])
    
    model = model.to(device)

    # Loss, optimizer, scheduler
    class_weights = torch.Tensor(config['class_weights']).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    
    momentum = config.get('momentum', 0.9)
    optimizer_name = config.get('optimizer_name', 'adamw')
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    match optimizer_name:
        case 'adamw':
            optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(momentum, 0.999), weight_decay=weight_decay)
        case 'sgd':
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config.get('final_learning_rate', 0))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_ranking_score = 0.0
    best_balanced_accuracy = 0.0

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'loss_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
    log_config_path = os.path.join(output_dir, 'config.json')
    with open(log_config_path, 'w') as file:
        json.dump(config, file, indent=4)

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
        
        # Extended validation with challenge metrics
        val_loss, ranking_score, fairness_metrics = validate(
            model, val_loader, device, criterion, 
            selected_fairness_variables, epoch=epoch, log_path=log_path,
            verbose=False
        )
        balanced_accuracy = fairness_metrics['balanced_accuracy']
        
        scheduler.step()

        # Log basic metrics
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Ranking Score: {ranking_score:.4f} | Bal. Accuracy: {balanced_accuracy:.4f}")

        # Only do best model updates if the balanced accuracy is > 0.5
        if balanced_accuracy > 0.5:
            if ranking_score > best_ranking_score:
                best_ranking_score = ranking_score
                best_balanced_accuracy = balanced_accuracy
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"New best ranking score: {best_ranking_score:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

    return best_ranking_score, best_balanced_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train Swin3D on NIfTI dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save logs and models")
    args = parser.parse_args()

    config = parse_config(args.config)
    train_model(config, args.output_dir)

if __name__ == '__main__':
    main()