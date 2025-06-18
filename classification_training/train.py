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
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score

from dataloader import NiftiImageDataset
from augmentations import random_mirroring, batch_generators_intensity_augmentations, batch_generators_spatial_augmentations

def parse_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def compute_fairness_metrics(predictions, labels, metadata_dict_list, selected_fairness_variables):
    """
    Compute fairness metrics across demographic subgroups.
    
    Args:
        predictions: torch.Tensor of predicted probabilities
        labels: torch.Tensor of true labels
        metadata_list: dictionary of lists with demographic information for each sample
        selected_fairness_variables: list of demographic variables to evaluate
    
    Returns:
        dict with fairness metrics
    """
    # Convert to numpy
    pred_probs = predictions.cpu().numpy()
    pred_labels = (pred_probs > 0.5).astype(int)
    true_labels = labels.cpu().numpy().astype(int)
    
    # Create results DataFrame from metadata list
    results_data = {
        'pcr': true_labels,
        'pcr_pred_prob': pred_probs,
        'pcr_pred': pred_labels
    }
    
    # Add demographic variables from metadata
    for var in selected_fairness_variables:
        results_data[var] = metadata_dict_list.get(var, None)
    
    results_df = pd.DataFrame(results_data)
    
    # Process demographic variables to match challenge format
    if 'age' in results_df.columns:
        # Convert age to age groups
        results_df['age'] = pd.cut(results_df['age'].astype(float), 
                                  bins=[0, 40, 50, 60, 70, 100], 
                                  labels=['0-40', '41-50', '51-60', '61-70', '71+'])
    
    if 'menopausal_status' in results_df.columns:
        # Process menopausal status
        results_df['menopausal_status'] = (
            results_df['menopausal_status']
            .fillna('unknown')
            .astype(str)
            .str.lower()
            .apply(lambda x: 'premenopause' if 'peri' in x or 'pre' in x 
                   else ('postmenopause' if 'post' in x else x))
        )
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    
    # Compute fairness disparities
    equalized_odds_disparities = {}
    
    for var in selected_fairness_variables:
        if var not in results_df.columns:
            print(f"Warning: {var} not found in metadata, skipping...")
            continue
        
        groups = results_df.groupby(var, observed=True) # Deactivate warning with observed=True
        tpr, fpr = {}, {}
        
        for i, (group_name, group) in enumerate(groups):
            if len(group) == 0:
                continue
                
            yt = group['pcr'].astype(int)
            yp = group['pcr_pred'].astype(int)
            
            # Skip if not enough samples or no variation
            if len(np.unique(yt)) < 2 or len(group) < 5:
                tpr[i], fpr[i] = 0, 0
                continue
                
            try:
                cm = confusion_matrix(yt, yp, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp_, fn, tp = cm.ravel()
                    tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr[i] = fp_ / (fp_ + tn) if (fp_ + tn) > 0 else 0
                else:
                    tpr[i], fpr[i] = 0, 0
            except ValueError:
                tpr[i], fpr[i] = 0, 0
        
        if len(tpr) > 1 and len(fpr) > 1:
            disparity = (max(tpr.values()) - min(tpr.values()) + 
                        max(fpr.values()) - min(fpr.values()))
            equalized_odds_disparities[var] = disparity
        else:
            equalized_odds_disparities[var] = 0.0
    
    # Compute fairness score
    if equalized_odds_disparities:
        fairness_score = np.mean(list(equalized_odds_disparities.values()))
        fairness_score = np.clip(fairness_score, 0, 1)
    else:
        fairness_score = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': balanced_accuracy,
        'fairness_score': fairness_score,
        'fairness_score_normalized': 1 - fairness_score,
        'equalized_odds_disparities': equalized_odds_disparities,
        'results_df': results_df
    }

def validate(model, dataloader, device, criterion, selected_fairness_variables=None, 
             alpha=0.5, epoch=None, log_path=None):
    """
    Extended validation function that computes challenge metrics.
    
    Args:
        model: PyTorch model
        dataloader: validation DataLoader that returns (images, labels, metadata_dict)
        device: torch device
        criterion: loss function
        selected_fairness_variables: list of demographic variables to evaluate
        alpha: weight for balancing performance and fairness in ranking score
        epoch: current epoch number for logging
        log_path: path to log file for detailed metrics
    
    Returns:
        tuple: (validation_loss, ranking_score, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_metadata_age = []
    all_metadata_breast_density = []
    all_metadata_menopausal_status = []
    
    with torch.no_grad():
        for images, labels, metadata_batch in dataloader:
            all_metadata_age.extend(metadata_batch['age'])
            all_metadata_breast_density.extend(metadata_batch['breast_density'])
            all_metadata_menopausal_status.extend(metadata_batch['menopausal_status'])
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Store predictions and labels
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            all_predictions.append(probabilities)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    validation_loss = running_loss / len(dataloader.dataset)

    all_metadata = {
        'age': all_metadata_age,
        'breast_density': all_metadata_breast_density,
        'menopausal_status': all_metadata_menopausal_status,
    }
    
    # Compute challenge metrics if metadata is available
    if all_metadata and selected_fairness_variables is not None:
        fairness_metrics = compute_fairness_metrics(
            all_predictions, all_labels, all_metadata, selected_fairness_variables
        )
        
        precision = fairness_metrics['precision']
        recall = fairness_metrics['recall']

        # Compute ranking score
        performance_score = fairness_metrics['balanced_accuracy']
        fairness_score = fairness_metrics['fairness_score']
        ranking_score = (1 - alpha) * performance_score + alpha * (1 - fairness_score)
        
        # Print metrics
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Balanced Accuracy: {performance_score:.4f}")
        print(f"Fairness Score (1-disparity): {fairness_metrics['fairness_score_normalized']:.4f}")
        print(f"Ranking Score: {ranking_score:.4f}")
        
        # Print disparity details
        print("Equalized Odds Disparities by subgroup:")
        for var, disparity in fairness_metrics['equalized_odds_disparities'].items():
            print(f"  {var}: {disparity:.4f}")
        
        # Log detailed metrics if log path is provided
        if log_path is not None and epoch is not None:
            # Create detailed log file if it doesn't exist
            detailed_log_path = log_path.replace('.csv', '_detailed.csv')
            if not os.path.exists(detailed_log_path):
                with open(detailed_log_path, 'w') as f:
                    header = "epoch,val_loss,precision,recall,balanced_accuracy,fairness_score,ranking_score"
                    for var in selected_fairness_variables:
                        header += f",{var}_disparity"
                    f.write(header + "\n")
            
            # Append metrics
            with open(detailed_log_path, 'a') as f:
                line = f"{epoch},{validation_loss:.4f},{precision:.4f},{recall:.4f},{performance_score:.4f}," \
                       f"{fairness_metrics['fairness_score_normalized']:.4f},{ranking_score:.4f}"
                for var in selected_fairness_variables:
                    disparity = fairness_metrics['equalized_odds_disparities'].get(var, 0.0)
                    line += f",{disparity:.4f}"
                f.write(line + "\n")
        
        return validation_loss, ranking_score, fairness_metrics
    
    else:
        # Fallback to basic validation loss
        print(f"Validation Loss: {validation_loss:.4f}")
        return validation_loss, validation_loss, None

def train_model(config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True if torch.cuda.is_available() else False

    data_augmentation_transforms = [random_mirroring,
                                    batch_generators_intensity_augmentations,
                                    batch_generators_spatial_augmentations]

    # Fairness evaluation settings
    selected_fairness_variables = config.get('fairness_variables', ['age', 'breast_density', 'menopausal_status'])
    print(f"Fairness variables: {selected_fairness_variables}")

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
    class_weights = torch.Tensor(config['class_weights']).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float('inf')
    best_ranking_score = 0.0

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
        
        # Extended validation with challenge metrics
        val_loss, ranking_score, fairness_metrics = validate(
            model, val_loader, device, criterion, 
            selected_fairness_variables, epoch, log_path
        )
        
        scheduler.step()

        # Log basic metrics
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model based on ranking score if available, otherwise validation loss
        if fairness_metrics is not None:
            if ranking_score > best_ranking_score:
                best_ranking_score = ranking_score
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"New best ranking score: {best_ranking_score:.4f}")
        else:
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