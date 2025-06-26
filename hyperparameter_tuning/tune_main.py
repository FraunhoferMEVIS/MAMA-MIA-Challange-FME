import os
import json
import numpy as np
from ray import tune, train
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune import Tuner, RunConfig
from ray.tune.search import ConcurrencyLimiter
from classification_training.train import train_model


def run_fold(sampled_config, fold_idx):
    with open(sampled_config["base_config_path"], 'r') as f:
        config = json.load(f)

    config.update({
        "learning_rate": sampled_config["learning_rate"],
        "weight_decay": sampled_config["weight_decay"],
        "batch_size": int(sampled_config["batch_size"]),
        "label_smoothing": sampled_config["label_smoothing"],
        "target_size": (int(sampled_config["z_resolution"]),
                        int(sampled_config["x_y_resolution"]),
                        int(sampled_config["x_y_resolution"])),
        "data_split_file": f"fold_{fold_idx}.json",
    })

    output_dir = os.path.join(sampled_config["output_root"], f"trial_{tune.get_context().get_trial_id()}_fold_{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
    best_ranking_score = train_model(config, output_dir)
    return best_ranking_score

def train_func(config):
    fold_scores = []
    for fold in range(1, 6):
        score = run_fold(config, fold)
        fold_scores.append(score)
    avg_score = np.mean(fold_scores)
    train.report({"mean_5_fold_ranking_score": avg_score})

search_space = {
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.uniform(16, 48),
    "label_smoothing": tune.uniform(0.0, 0.1),
    "x_y_resolution": tune.uniform(50, 150),
    "z_resolution": tune.uniform(16, 50),
    "base_config_path": os.environ["BASE_CONFIG"],
    "output_root": os.environ["TUNE_OUTPUTS"]
}

if __name__ == "__main__":
    bayesopt = BayesOptSearch(metric="mean_5_fold_ranking_score", mode="max")

    tuner = Tuner(
        trainable=train_func,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=50,
            search_alg=bayesopt,
            max_concurrent_trials=2
        ),
        run_config=RunConfig(
            name="swin3d_t_hyperparam_tuning",
            storage_path=os.environ["RAY_RESULTS"],
            verbose=1
        ),
    )
    tuner.fit()
