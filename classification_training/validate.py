
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score

def compute_fairness_metrics(predictions: torch.Tensor,
                             labels: torch.Tensor,
                             metadata_dict_list: dict,
                             selected_fairness_variables: list[str]) -> dict:
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
        mean_equalized_odds_disparities = np.mean(list(equalized_odds_disparities.values()))
        mean_equalized_odds_disparities = np.clip(mean_equalized_odds_disparities, 0, 1)
    else:
        mean_equalized_odds_disparities = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': balanced_accuracy,
        'mean_equalized_odds_disparities': mean_equalized_odds_disparities,
        'fairness_score': 1 - mean_equalized_odds_disparities,
        'equalized_odds_disparities': equalized_odds_disparities,
        'results_df': results_df
    }

def validate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             device,
             criterion: torch.nn.Module,
             selected_fairness_variables: list[str] = None, 
             alpha: float = 0.5,
             epoch: int | None = None,
             log_path: str = None,
             verbose: bool = False) -> tuple[float, float, dict]:
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
        balanced_accuracy = fairness_metrics['balanced_accuracy']
        fairness_score = fairness_metrics['fairness_score']
        
        ranking_score = (1 - alpha) * balanced_accuracy + alpha * fairness_score
        
        if verbose:
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
            print(f"Fairness Score (1-disparity): {fairness_score:.4f}")
            print(f"Ranking Score: {ranking_score:.4f}")
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
                line = f"{epoch},{validation_loss:.4f},{precision:.4f},{recall:.4f},{balanced_accuracy:.4f}," \
                       f"{fairness_metrics['fairness_score']:.4f},{ranking_score:.4f}"
                for var in selected_fairness_variables:
                    disparity = fairness_metrics['equalized_odds_disparities'].get(var, 0.0)
                    line += f",{disparity:.4f}"
                f.write(line + "\n")
        
        return validation_loss, ranking_score, fairness_metrics
    
    else:
        # Fallback to basic validation loss
        print(f"Validation Loss: {validation_loss:.4f}")
        return validation_loss, validation_loss, None