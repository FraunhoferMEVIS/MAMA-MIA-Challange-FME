import os
import json
import numpy as np
import torch
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune import Tuner, RunConfig
from classification_training.train import train_model

def transform_value(value, lower, upper):
    return (value / 10) * (upper - lower) + lower

def run_fold(sampled_config, fold_idx):
    with open(os.environ["BASE_CONFIG"], 'r') as f:
        config = json.load(f)

    config.update({
        "learning_rate": sampled_config["learning_rate"],
        "weight_decay": sampled_config["weight_decay"],
        "batch_size": int(sampled_config["batch_size"]),
        "label_smoothing": sampled_config["label_smoothing"],
        "target_size": (int(sampled_config["z_resolution"]),
                        int(sampled_config["x_y_resolution"]),
                        int(sampled_config["x_y_resolution"])),
        "normalization": sampled_config["normalization"],
        "model_key": sampled_config["model_key"],
        "data_split_file": f"fold_{fold_idx}.json",
    })

    output_dir = os.path.join(os.environ["TUNE_TRIALS"],
                              tune.get_context().get_experiment_name(),
                              f"trial_{tune.get_context().get_trial_id()}_fold_{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
    best_ranking_score = train_model(config, output_dir)
    return best_ranking_score

def train_func(config):
    try:
        fold_scores = []
        for fold in range(1, 6):
            score = run_fold(config, fold)
            fold_scores.append(score)
        avg_score = np.mean(fold_scores)
        tune.report({"mean_5_fold_ranking_score": avg_score})
    except torch.OutOfMemoryError:
        # Handle OOM configurations by reporting a small metric value
        tune.report({"mean_5_fold_ranking_score": 0})

search_space = {
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.quniform(16, 50, 1),
    "label_smoothing": tune.uniform(0, 0.1),
    "x_y_resolution": tune.quniform(50, 150, 1),
    "z_resolution": tune.quniform(16, 50, 1),
    "normalization": tune.choice(["none", "zScoreFirstChannelBased"]),
    "model_key": tune.choice(["swin3d_t", "mc3_18", "mvit_v2_s", "r2plus1d_18", "s3d"])
}

if __name__ == "__main__":
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
                num_samples=100,
                max_concurrent_trials=int(os.environ.get("MAX_CONCURRENT_TRIALS", 2))
            ),
            run_config=RunConfig(
                name=name,
                storage_path=storage_path,
                verbose=1
            ),
        )
    tuner.fit()
