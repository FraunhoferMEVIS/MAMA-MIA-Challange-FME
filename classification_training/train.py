import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np

from classification_training.augmentations import random_mirroring, random_mirroring_2d, \
    batch_generators_intensity_augmentations, batch_generators_spatial_augmentations
from classification_training.dataloader import NiftiImageDataset, NiftiImageDataset2DCenterSlice
from classification_training.validate import validate
import classification_training.models as models


def parse_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def train_model(config: dict, output_dir: str) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True if torch.cuda.is_available() else False

    # Fairness evaluation settings
    selected_fairness_variables = config.get('fairness_variables', ['age', 'breast_density', 'menopausal_status'])
    print(f"Fairness variables: {selected_fairness_variables}")

    spatial_dimensions = config.get('spatial_dimensions', 3)

    if spatial_dimensions == 3:
        data_augmentation_transforms = [random_mirroring,
                                        batch_generators_intensity_augmentations,
                                        batch_generators_spatial_augmentations]
        DatasetClass = NiftiImageDataset
    elif spatial_dimensions == 2:
        data_augmentation_transforms = [random_mirroring_2d,
                                        batch_generators_intensity_augmentations,
                                        batch_generators_spatial_augmentations]
        DatasetClass = NiftiImageDataset2DCenterSlice


    train_dataset = DatasetClass(
        data_dir=os.path.join(config['data_path']),
        data_split_file=config['data_split_file'],
        group='training',
        target_size=config['target_size'],
        transforms=data_augmentation_transforms,
        normalization=config['normalization']
    )

    val_dataset = DatasetClass(
        data_dir=os.path.join(config['data_path']),
        data_split_file=config['data_split_file'],
        group='validation',
        target_size=config['target_size'],
        normalization=config['normalization']
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config['num_workers'])

    # Model
    model_key = config['model_key']
    pretrained = config['pretrained']
    model = models.get_model(model_key, pretrained=pretrained)
    model = model.to(device)

    # Loss, optimizer, scheduler
    class_weights = torch.Tensor(config['class_weights']).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_ranking_score = 0.0
    ranking_scores = []

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'loss_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
    log_config_path = os.path.join(output_dir, 'config.json')
    with open(log_config_path, 'w') as file:
        json.dump(config, file)

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
        
        scheduler.step()

        # Log basic metrics
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Ranking Score: {ranking_score:.4f}")

        balanced_accuracy = fairness_metrics['balanced_accuracy']
        # Only do best model updates if the balanced accuracy is > 0.5
        if balanced_accuracy > 0.5:
            ranking_scores.append(ranking_score)
            if ranking_score > best_ranking_score:
                best_ranking_score = ranking_score
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"New best ranking score: {best_ranking_score:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

    if len(ranking_scores) > 0:
        ranking_scores.sort(reverse=True)
        top_4_ranking_scores = ranking_scores[:4]
        top_4_ranking_average = np.mean(top_4_ranking_scores)
    else:
        top_4_ranking_average = 0.5
    return top_4_ranking_average

def main():
    parser = argparse.ArgumentParser(description="Train Swin3D on NIfTI dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save logs and models")
    args = parser.parse_args()

    config = parse_config(args.config)
    train_model(config, args.output_dir)

if __name__ == '__main__':
    main()