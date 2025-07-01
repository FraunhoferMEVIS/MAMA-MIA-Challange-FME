import os
import argparse
import subprocess
import numpy as np
import nibabel as nib
import csv

# Define global parameters
VOXEL_VOLUME_MM3 = 1.0  # Change if voxel size is known
PREDICT_SCRIPT = "../../3D-Breast-FGT-and-Blood-Vessel-Segmentation/predict.py"
BREAST_MODEL_PATH = "../../3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/breast_model.pth"
FGT_MODEL_PATH = "../../3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/dv_model.pth"

def run_prediction(image_path, target, save_dir, model_path, input_mask=None):
    cmd = [
        "python", PREDICT_SCRIPT,
        "--target-tissue", target,
        "--image", image_path,
        "--save-masks-dir", save_dir,
        "--model-save-path", model_path
    ]
    if input_mask:
        cmd += ["--input-mask", input_mask]
    subprocess.run(cmd, check=True)

def convert_numpy_to_nifti(npy_path, nifti_path):
    arr = np.load(npy_path)
    nifti_img = nib.Nifti1Image(arr.astype(np.uint8), affine=np.eye(4))
    nib.save(nifti_img, nifti_path)

def compute_volume(mask_array):
    return np.sum(mask_array > 0) * VOXEL_VOLUME_MM3

def main(base_dir):
    # Define folders
    preprocessed_dir = os.path.join(base_dir, "preprocessed_images")
    breast_numpy_dir = os.path.join(base_dir, "breast_segmentations_numpy")
    fgt_numpy_dir = os.path.join(base_dir, "fgt_segmentation_numpy")
    breast_nifti_dir = os.path.join(base_dir, "breast_segmentations_nifti")
    fgt_nifti_dir = os.path.join(base_dir, "fgt_segmentation_nifti")
    os.makedirs(breast_numpy_dir, exist_ok=True)
    os.makedirs(fgt_numpy_dir, exist_ok=True)
    os.makedirs(breast_nifti_dir, exist_ok=True)
    os.makedirs(fgt_nifti_dir, exist_ok=True)

    volume_csv_path = os.path.join(base_dir, "volumes.csv")
    with open(volume_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Patient", "Breast Volume (mm³)", "FGT Volume (mm³)"])

        for patient_id in os.listdir(preprocessed_dir):
            patient_folder = os.path.join(preprocessed_dir, patient_id)
            if not os.path.isdir(patient_folder):
                continue

            print(f"Processing {patient_id}...")
            image_dir = os.path.join(patient_folder)
            breast_save_dir = os.path.join(breast_numpy_dir, patient_id)
            fgt_save_dir = os.path.join(fgt_numpy_dir, patient_id)
            os.makedirs(breast_save_dir, exist_ok=True)
            os.makedirs(fgt_save_dir, exist_ok=True)

            # Step 1: Predict Breast Mask
            run_prediction(
                image_path=image_dir,
                target="breast",
                save_dir=breast_save_dir,
                model_path=BREAST_MODEL_PATH
            )
            breast_mask_file = os.path.join(breast_save_dir, f"{patient_id}_preprocessed.npy")
            convert_numpy_to_nifti(breast_mask_file, os.path.join(breast_nifti_dir, f"{patient_id}_breast_mask.nii.gz"))

            # Step 2: Predict FGT Mask (with breast mask as input)
            run_prediction(
                image_path=image_dir,
                target="dv",
                save_dir=fgt_save_dir,
                model_path=FGT_MODEL_PATH,
                input_mask=breast_save_dir
            )
            fgt_mask_file = os.path.join(fgt_save_dir, f"{patient_id}_preprocessed.npy")
            convert_numpy_to_nifti(fgt_mask_file, os.path.join(fgt_nifti_dir, f"{patient_id}_fgt_mask.nii.gz"))

            # Step 4: Compute volumes
            breast_mask = np.load(breast_mask_file)
            fgt_mask = np.load(fgt_mask_file)
            breast_vol = compute_volume(breast_mask)
            fgt_vol = compute_volume(fgt_mask)
            writer.writerow([patient_id, breast_vol, fgt_vol])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation pipeline for breast and FGT tissue.")
    parser.add_argument("input_dir", help="Base input directory containing all data folders.")
    args = parser.parse_args()
    main(args.input_dir)
