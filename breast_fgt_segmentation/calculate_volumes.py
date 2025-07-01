import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd

def compute_volumes_and_density(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    voxel_volume = np.prod(img.header.get_zooms())  # in mm^3

    breast_mask = (data == 1) | (data == 2)
    parenchyma_mask = (data == 2)

    breast_volume = np.sum(breast_mask) * voxel_volume
    parenchyma_volume = np.sum(parenchyma_mask) * voxel_volume
    breast_density = parenchyma_volume / breast_volume if breast_volume > 0 else 0

    return breast_volume, parenchyma_volume, breast_density

def main(input_folder, output_csv):
    results = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_folder, filename)
            breast_vol, parenchyma_vol, density = compute_volumes_and_density(file_path)
            results.append({
                'filename': filename,
                'breast_volume_mm3': breast_vol,
                'parenchyma_volume_mm3': parenchyma_vol,
                'breast_density': density
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute breast and parenchyma volumes and density from NIfTI segmentations.")
    parser.add_argument('input_folder', type=str, help="Folder with .nii.gz breast segmentation files")
    parser.add_argument('output_csv', type=str, help="Output CSV file path")
    args = parser.parse_args()
    main(args.input_folder, args.output_csv)
