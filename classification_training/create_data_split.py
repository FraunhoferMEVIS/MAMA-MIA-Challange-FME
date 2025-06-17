#!/usr/bin/env python3
"""
N-Fold Dataset Splitter for Deep Learning

This script scans image files in a folder and creates n-fold cross-validation splits,
saving each fold as a separate JSON file with training and validation sets.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random


def get_image_stems(images_folder: Path) -> List[str]:
    """
    Scan the images folder and extract filename stems from image files.
    
    Args:
        images_folder: Path to the images subfolder
        
    Returns:
        List of filename stems (without extensions)
    """
    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    # Common medical imaging and general image extensions
    image_extensions = {'.nii.gz', '.nii', '.dcm', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    stems = []
    for file_path in images_folder.iterdir():
        if file_path.is_file():
            # Handle .nii.gz files specially (double extension)
            if file_path.name.endswith('.nii.gz'):
                stem = file_path.name[:-7]  # Remove .nii.gz
            else:
                # Check if it's an image file
                if file_path.suffix.lower() in image_extensions:
                    stem = file_path.stem
                else:
                    continue
            stems.append(stem)
    
    return sorted(stems)


def create_nfold_splits(stems: List[str], n_folds: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Create n-fold cross-validation splits.
    
    Args:
        stems: List of filename stems
        n_folds: Number of folds to create
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing fold information
    """
    if n_folds <= 1:
        raise ValueError("Number of folds must be greater than 1")
    
    if len(stems) < n_folds:
        raise ValueError(f"Cannot create {n_folds} folds with only {len(stems)} samples")
    
    # Set random seed for reproducibility
    random.seed(seed)
    shuffled_stems = stems.copy()
    random.shuffle(shuffled_stems)
    
    # Calculate fold sizes
    fold_size = len(shuffled_stems) // n_folds
    remainder = len(shuffled_stems) % n_folds
    
    folds = []
    start_idx = 0
    
    for fold_idx in range(n_folds):
        # Distribute remainder across first few folds
        current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + current_fold_size
        
        # Validation set is the current fold
        validation = shuffled_stems[start_idx:end_idx]
        
        # Training set is all other samples
        training = shuffled_stems[:start_idx] + shuffled_stems[end_idx:]
        
        fold_data = {
            "fold": fold_idx + 1,
            "total_samples": len(shuffled_stems),
            "training_samples": len(training),
            "validation_samples": len(validation),
            "training": training,
            "validation": validation
        }
        
        folds.append(fold_data)
        start_idx = end_idx
    
    return folds


def save_fold_json(fold_data: Dict[str, Any], output_folder: Path, fold_idx: int):
    """
    Save a single fold to a JSON file.
    
    Args:
        fold_data: Dictionary containing fold information
        output_folder: Path to output folder
        fold_idx: Fold index (1-based)
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"fold_{fold_idx}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fold_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved fold {fold_idx}: {output_file}")
    print(f"  Training samples: {fold_data['training_samples']}")
    print(f"  Validation samples: {fold_data['validation_samples']}")


def main():
    parser = argparse.ArgumentParser(
        description="Create n-fold cross-validation splits from image files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset
  %(prog)s /path/to/dataset --folds 10
  %(prog)s /path/to/dataset --folds 3 --output ./splits --seed 123
        """
    )
    
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to the main dataset folder containing the "images" subfolder'
    )
    
    parser.add_argument(
        '--folds', '-f',
        type=int,
        default=5,
        help='Number of folds to create (default: 5)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output folder for JSON files (default: same as input folder)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--images-subfolder',
        type=str,
        default='images',
        help='Name of the images subfolder (default: "images")'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        main_folder = Path(args.folder_path)
        if not main_folder.exists():
            raise FileNotFoundError(f"Main folder not found: {main_folder}")
        
        images_folder = main_folder / args.images_subfolder
        output_folder = Path(args.output) if args.output else main_folder
        
        # Get image stems
        print(f"Scanning images in: {images_folder}")
        stems = get_image_stems(images_folder)
        
        if not stems:
            print("No image files found!")
            return
        
        print(f"Found {len(stems)} image files")
        print(f"Creating {args.folds}-fold cross-validation splits...")
        
        # Create n-fold splits
        folds = create_nfold_splits(stems, args.folds, args.seed)
        
        # Save each fold
        print(f"\nSaving splits to: {output_folder}")
        for fold_data in folds:
            save_fold_json(fold_data, output_folder, fold_data['fold'])
        
        print(f"\nSuccessfully created {args.folds} fold files!")
        print(f"Total samples: {len(stems)}")
        print(f"Random seed used: {args.seed}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())