import os
import json
import numpy as np
import torch
from ray import tune
from ray.tune import Tuner, RunConfig
from classification_training.train import train_model

def run_fold(sampled_config, fold_idx):
    with open(os.environ["BASE_CONFIG"], 'r') as f:
        config = json.load(f)

    config.update({
        "learning_rate": sampled_config["learning_rate"],
        "final_learning_rate": sampled_config["final_learning_rate"],
        "momentum": sampled_config["momentum"],
        "weight_decay": sampled_config["weight_decay"],
        "batch_size": int(sampled_config["batch_size"]),
        "label_smoothing": sampled_config["label_smoothing"],
        "target_size": (int(sampled_config["z_resolution"]),
                        int(sampled_config["x_y_resolution"]),
                        int(sampled_config["x_y_resolution"])),
        "encoder_key": sampled_config["encoder_key"],
        "attention_type": sampled_config["attention_type"],
        "data_split_file": f"fold_{fold_idx}.json",
    })

    output_dir = os.path.join(os.environ["TUNE_TRIALS"],
                              tune.get_context().get_experiment_name(),
                              f"trial_{tune.get_context().get_trial_id()}_fold_{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
    best_ranking_score, balanced_accuracy = train_model(config, output_dir)
    return best_ranking_score, balanced_accuracy

def train_func(config):
    try:
        fold_scores = []
        fold_balanced_accuracies = []
        for fold in range(1, 6):
            score, balanced_accuracy = run_fold(config, fold)
            fold_scores.append(score)
            fold_balanced_accuracies.append(balanced_accuracy)
        avg_score = np.mean(fold_scores)
        avg_balanced_accuracy = np.mean(fold_balanced_accuracies)
        tune.report({"mean_5_fold_ranking_score": avg_score, "balanced_accuracy": avg_balanced_accuracy})
    except torch.OutOfMemoryError:
        # Handle OOM configurations by reporting a small metric value
        tune.report({"mean_5_fold_ranking_score": 0, "balanced_accuracy": 0})

if __name__ == "__main__":
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "final_learning_rate": tune.loguniform(1e-8, 1e-4),
        "momentum": tune.uniform(0.8, 0.99),
        "weight_decay": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.quniform(8, 48, 1),
        "label_smoothing": tune.loguniform(0.0001, 0.1),
        "x_y_resolution": tune.quniform(50, 200, 1),
        "z_resolution": tune.quniform(8, 50, 1),
        "encoder_key": tune.choice(["convnext_tiny", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
                                    "efficientnet_v2_s", "resnet18", "resnet34", "swin_t", "swin_v2_t"]),
        "attention_type": tune.choice(["simple_attention", "transformer_encoder"])
    }

    trainable_with_resources = tune.with_resources(train_func, {"cpu": 4, "gpu": 1})
    name = os.environ["TUNING_RUN_NAME"]
    storage_path = os.environ["TUNE_RESULTS"]
    exp_dir = os.path.join(storage_path, name)
    if Tuner.can_restore(exp_dir):
        tuner = Tuner.restore(
            exp_dir,
            trainable=trainable_with_resources,
            param_space=search_space,
            resume_errored=True,
        )
    else:
        tuner = Tuner(
            trainable=trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=400,
                max_concurrent_trials=int(os.environ.get("MAX_CONCURRENT_TRIALS", 2))
            ),
            run_config=RunConfig(
                name=name,
                storage_path=storage_path,
                verbose=1
            ),
        )
    tuner.fit()
