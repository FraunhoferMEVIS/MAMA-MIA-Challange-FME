import argparse
import json
import os
import shutil
from pathlib import Path
from classification_training.onnx_export import  export_model

def main(args):
    # Load trial list
    with open(args.trial_list, 'r') as f:
        trial_ids = json.load(f)

    os.makedirs(args.output_folder, exist_ok=True)

    for trial_id in trial_ids:
        for fold in range(1, 6):  # Folds 1 to 5
            folder_name = f"trial_{trial_id}_fold_{fold}"
            src_folder = Path(args.trial_folder) / folder_name

            if not src_folder.exists():
                print(f"Skipping missing folder: {src_folder}")
                continue

            dst_folder = Path(args.output_folder) / folder_name
            dst_folder.mkdir(parents=True, exist_ok=True)

            # Copy and read config.json
            config_path = src_folder / "config.json"
            if not config_path.exists():
                print(f"Warning: config.json not found in {src_folder}")
                continue
            shutil.copy(config_path, dst_folder / "config.json")

            with open(config_path, "r") as f:
                config = json.load(f)

            model_key = config.get("model_key")
            if not isinstance(model_key, str):
                print(f"Warning: 'model_key' key missing or invalid in config.json for {folder_name}")
                continue

            target_size = config.get("target_size")
            if not isinstance(target_size, list) or not all(isinstance(x, int) for x in target_size):
                print(f"Warning: 'target_size' missing or invalid in config.json for {folder_name}")
                continue

            target_size_onnx = [1, 3] + target_size

            # Export ONNX
            weights_path = src_folder / "best_model.pth"
            if not weights_path.exists():
                print(f"Warning: best_model.pth not found in {src_folder}")
                continue

            onnx_output_path = dst_folder / "model.onnx"

            if os.path.exists(onnx_output_path):
                continue

            print(f"Exporting ONNX for {folder_name} with model '{model_key}' and input size {target_size_onnx}")
            export_model(model_key, weights_path, onnx_output_path, target_size_onnx, half_precision=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export selected trials to ONNX format")
    parser.add_argument("--trial-list", type=str, required=True, help="Path to JSON file with trial IDs")
    parser.add_argument("--trial-folder", type=str, required=True, help="Path to folder containing all trial subfolders")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to output folder")
    args = parser.parse_args()
    main(args)
