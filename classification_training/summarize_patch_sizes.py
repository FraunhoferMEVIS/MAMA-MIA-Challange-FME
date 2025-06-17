#!/usr/bin/env python3
"""
NIfTI Image Statistics Script

Analyzes a folder of NIfTI images (.nii.gz) and prints summary statistics
for image extents (dimensions) and voxel sizes.

Usage: python nifti_stats.py <folder_path>
"""

import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def get_nifti_info(filepath):
    """Extract image dimensions and voxel sizes from a NIfTI file."""
    try:
        img = nib.load(filepath)
        header = img.header
        
        # Get image dimensions (shape)
        shape = img.shape
        
        # Get voxel sizes (pixel dimensions)
        voxel_sizes = header.get_zooms()
        
        return shape, voxel_sizes
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def analyze_folder(folder_path):
    """Analyze all NIfTI files in the given folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Find all .nii.gz files
    nifti_files = list(folder_path.glob("*.nii.gz"))
    nifti_files.extend(folder_path.glob("*.nii"))  # Also include .nii files
    
    if not nifti_files:
        print(f"No NIfTI files found in '{folder_path}'")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files in '{folder_path}'\n")
    
    # Collect data
    dimensions = []
    voxel_sizes_data = []
    valid_files = 0
    
    print("Processing files...")
    for filepath in nifti_files:
        print(f"  {filepath.name}")
        shape, voxel_sizes = get_nifti_info(filepath)
        
        if shape is not None and voxel_sizes is not None:
            dimensions.append(shape)
            voxel_sizes_data.append(voxel_sizes)
            valid_files += 1
    
    if valid_files == 0:
        print("No valid NIfTI files found.")
        return
    
    print(f"\nSuccessfully processed {valid_files} files\n")
    
    # Analyze dimensions
    print("=" * 60)
    print("IMAGE EXTENT STATISTICS")
    print("=" * 60)
    
    # Convert to numpy array for easier analysis
    dims_array = np.array(dimensions)
    max_dims = dims_array.shape[1]  # Maximum number of dimensions
    
    for dim in range(max_dims):
        dim_values = dims_array[:, dim]
        print(f"\nDimension {dim + 1}:")
        print(f"  Min: {np.min(dim_values)}")
        print(f"  Max: {np.max(dim_values)}")
        print(f"  Mean: {np.mean(dim_values):.2f}")
        print(f"  Std: {np.std(dim_values):.2f}")
        print(f"  Median: {np.median(dim_values):.2f}")
    

    
    # Analyze voxel sizes
    print("\n" + "=" * 60)
    print("VOXEL SIZE STATISTICS")
    print("=" * 60)
    
    # Convert to numpy array
    voxel_array = np.array(voxel_sizes_data)
    max_voxel_dims = voxel_array.shape[1]
    
    for dim in range(max_voxel_dims):
        voxel_values = voxel_array[:, dim]
        print(f"\nVoxel dimension {dim + 1} (mm):")
        print(f"  Min: {np.min(voxel_values):.4f}")
        print(f"  Max: {np.max(voxel_values):.4f}")
        print(f"  Mean: {np.mean(voxel_values):.4f}")
        print(f"  Std: {np.std(voxel_values):.4f}")
        print(f"  Median: {np.median(voxel_values):.4f}")
    

    
    # Calculate total volume statistics
    print("\n" + "=" * 60)
    print("VOLUME STATISTICS")
    print("=" * 60)
    
    volumes = []
    for i, (shape, voxel_sizes) in enumerate(zip(dimensions, voxel_sizes_data)):
        # Calculate volume in mm³
        volume = np.prod(shape[:3]) * np.prod(voxel_sizes[:3])  # Use first 3 dimensions
        volumes.append(volume)
    
    volumes = np.array(volumes)
    print(f"\nTotal volume (mm³):")
    print(f"  Min: {np.min(volumes):.2f}")
    print(f"  Max: {np.max(volumes):.2f}")
    print(f"  Mean: {np.mean(volumes):.2f}")
    print(f"  Std: {np.std(volumes):.2f}")
    print(f"  Median: {np.median(volumes):.2f}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze NIfTI images and print summary statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nifti_stats.py /path/to/nifti/folder
  python nifti_stats.py ./brain_images/
        """
    )
    
    parser.add_argument(
        "folder_path",
        help="Path to folder containing NIfTI files (.nii.gz or .nii)"
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        analyze_folder(args.folder_path)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()