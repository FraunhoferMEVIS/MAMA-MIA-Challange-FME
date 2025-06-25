# ==================== MAMA-MIA CHALLENGE SAMPLE SUBMISSION ====================
#
# This is the official sample submission script for the **MAMA-MIA Challenge**, 
# covering both tasks:
#
#   1. Primary Tumour Segmentation (Task 1)
#   2. Treatment Response Classification (Task 2)
#
# ----------------------------- SUBMISSION FORMAT -----------------------------
# Participants must implement a class `Model` with one or two of these methods:
#
#   - `predict_segmentation(output_dir)`: required for Task 1
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#
#   - `predict_classification(output_dir)`: required for Task 2
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
#   - `predict_classification(output_dir)`: if a single model handles both tasks
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
# You can submit:
#   - Only Task 1 (implement `predict_segmentation`)
#   - Only Task 2 (implement `predict_classification`)
#   - Both Tasks (implement both methods independently or define `predict_segmentation_and_classification` method)
#
# ------------------------ SANITY-CHECK PHASE ------------------------
#
# ✅ Before entering the validation or test phases, participants must pass the **Sanity-Check phase**.
#   - This phase uses **4 samples from the test set** to ensure your submission pipeline runs correctly.
#   - Submissions in this phase are **not scored**, but must complete successfully within the **20-minute timeout limit**.
#   - Use this phase to debug your pipeline and verify output formats without impacting your submission quota.
#
# 💡 This helps avoid wasted submissions on later phases due to technical errors.
#
# ------------------------ SUBMISSION LIMITATIONS ------------------------
#
# ⚠️ Submission limits are strictly enforced per team:
#   - **One submission per day**
#   - **Up to 15 submissions total on the validation set**
#   - **Only 1 final submission on the test set**
#
# Plan your development and testing accordingly to avoid exhausting submissions prematurely.
#
# ----------------------------- RUNTIME AND RESOURCES -----------------------------
#
# > ⚠️ VERY IMPORTANT: Each image has a **timeout of 5 minutes** on the compute worker.
#   - **Validation Set**: 58 patients → total budget ≈ 290 minutes
#   - **Test Set**: 516 patients → total budget ≈ 2580 minutes
#
# > The compute worker environment is based on the Docker image:
#       `lgarrucho/codabench-gpu:latest`
#
# > You can install additional dependencies via `requirements.txt`.
#   Please ensure all required packages are listed there.
#
# ----------------------------- SEGMENTATION DETAILS -----------------------------
#
# This example uses `nnUNet v2`, which is compatible with the GPU compute worker.
# Note the following nnUNet-specific constraints:
#
# ✅ `predict_from_files_sequential` MUST be used for inference.
#     - This is because nnUNet’s multiprocessing is incompatible with the compute container.
#     - In our environment, a single fold prediction using `predict_from_files_sequential` 
#       takes approximately **1 minute per patient**.
#
# ✅ The model uses **fold 0 only** to reduce runtime.
# 
# ✅ Predictions are post-processed by applying a breast bounding box mask using 
#    metadata provided in the per-patient JSON file.
#
# ----------------------------- CLASSIFICATION DETAILS -----------------------------
#
# If using predicted segmentations for Task 2 classification:
#   - Save them in `self.predicted_segmentations` inside `predict_segmentation()`
#   - You can reuse them in `predict_classification()`
#   - Or perform Task 1 and Task 2 inside `predict_segmentation_and_classification`
#
# ----------------------------- DATASET INTERFACE -----------------------------
# The provided `dataset` object is a `RestrictedDataset` instance and includes:
#
#   - `dataset.get_patient_id_list() → list[str]`  
#         Patient IDs for current split (val/test)
#
#   - `dataset.get_dce_mri_path_list(patient_id) → list[str]`  
#         Paths to all image channels (typically pre and post contrast)
#         - iamge_list[0] corresponds to the pre-contrast image path
#         - iamge_list[1] corresponds to the first post-contrast image path and so on
#
#   - `dataset.read_json_file(patient_id) → dict`  
#         Metadata dictionary per patient, including:
#         - breast bounding box (`primary_lesion.breast_coordinates`)
#         - scanner metadata (`imaging_data`), etc...
#
# Example JSON structure:
# {
#   "patient_id": "XXX_XXX_SXXXX",
#   "primary_lesion": {
#     "breast_coordinates": {
#         "x_min": 1, "x_max": 158,
#         "y_min": 6, "y_max": 276,
#         "z_min": 1, "z_max": 176
#     }
#   },
#   "imaging_data": {
#     "bilateral": true,
#     "dataset": "HOSPITAL_X",
#     "site": "HOSPITAL_X",
#     "scanner_manufacturer": "SIEMENS",
#     "scanner_model": "Aera",
#     "field_strength": 1.5,
#     "echo_time": 1.11,
#     "repetition_time": 3.35
#   }
# }
#
# ----------------------------- RECOMMENDATIONS -----------------------------
# ✅ We recommend to always test your submission first in the Sanity-Check Phase.
#    As in Codabench the phases need to be sequential and they cannot run in parallel,
#    we will open a secondary MAMA-MIA Challenge Codabench page with a permanen Sanity-Check phase.
#   That way you won't lose submission trials to the validation or even wore, the test set.
# ✅ We recommend testing your solution locally and measuring execution time per image.
# ✅ Use lightweight models or limit folds if running nnUNet.
# ✅ Keep all file paths, patient IDs, and formats **exactly** as specified.
# ✅ Ensure your output folders are created correctly (e.g. `pred_segmentations/`)
# ✅ For faster runtime, only select a single image for segmentation.
#
# ------------------------ COPYRIGHT ------------------------------------------
#
# © 2025 Lidia Garrucho. All rights reserved.
# Unauthorized use, reproduction, or distribution of any part of this competition's 
# materials is prohibited without explicit permission.
#
# ------------------------------------------------------------------------------

# === MANDATORY IMPORTS ===
import os
import pandas as pd
import shutil

# === OPTIONAL IMPORTS: only needed if you modify or extend nnUNet input/output handling ===
# You can remove unused imports above if not needed for your solution
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
# === nnUNetv2 IMPORTS ===
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class Model:
    def __init__(self, dataset):
        """
        Initializes the model with the restricted dataset.
        
        Args:
            dataset (RestrictedDataset): Preloaded dataset instance with controlled access.
        """
        # MANDATOR
        self.dataset = dataset  # Restricted Access to Private Dataset
        self.predicted_segmentations = None  # Optional: stores path to predicted segmentations
        self.classification_model_folder = "/app/ingested_program/code_submission_1/classification_models"
        

    def predict_segmentation(self, output_dir):
        """
        Task 1 — Predict tumor segmentation with nnUNetv2.
        You MUST define this method if participating in Task 1.

        Args:
            output_dir (str): Directory where predictions will be stored.

        Returns:
            str: Path to folder with predicted segmentation masks.
        """

        # === Set required nnUNet paths ===
        # Not strictly mandatory if pre-set in Docker env, but avoids missing variable warnings
        os.environ['nnUNet_raw'] = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_preprocessed'] = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_results'] = "/app/ingested_program/sample_code_submission"

        # Usage: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/inference
        # === Instantiate nnUNet Predictor ===
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        # === Load your trained model from a specific fold ===
        predictor.initialize_from_trained_model_folder(
            '/app/ingested_program/code_submission_1/Dataset125_MAMA_MIA_expert_segmentations_1_subtraction/nnUNetTrainer__nnUNetResEncUNetLPlans24GB__3d_fullres',
            use_folds=(0,1,2,3,4), checkpoint_name='checkpoint_best.pth')
        
        # === Build nnUNet-compatible input images folder ===
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        os.makedirs(nnunet_input_images, exist_ok=True)

        # === Participants can modify how they prepare input ===
        patient_ids = self.dataset.get_patient_id_list()
        for patient_id in patient_ids:
            images = self.dataset.get_dce_mri_path_list(patient_id)

            pre_contrast_image_path = images[0]
            first_post_contrast_image_path = images[1]

            nii_precontrast = nib.load(str(pre_contrast_image_path))
            data_precontrast = nii_precontrast.get_fdata().astype(np.int16)
            nii_postcontrast = nib.load(str(first_post_contrast_image_path))
            data_postcontrast = nii_postcontrast.get_fdata().astype(np.int16)

            subtraction_data = data_postcontrast - data_precontrast

            subtraction_nii = nib.Nifti1Image(subtraction_data.astype(np.int16), nii_postcontrast.affine, nii_postcontrast.header)
            subtraction_nii.set_data_dtype(np.int16)
            nnunet_image_path = os.path.join(nnunet_input_images, f"{patient_id}_0001_0000.nii.gz")
            nib.save(subtraction_nii, nnunet_image_path)

        # === Output folder for raw nnUNet segmentations ===
        output_dir_nnunet = os.path.join(output_dir, 'nnunet_seg')
        os.makedirs(output_dir_nnunet, exist_ok=True)

        # === Call nnUNetv2 prediction ===
        nnunet_images = [[os.path.join(nnunet_input_images, f)] for f in os.listdir(nnunet_input_images)]
        # IMPORTANT: the only method that works inside the Docker container is predict_from_files_sequential
        # This method will predict all images in the list and save them in the output directory
        ret = predictor.predict_from_files_sequential(nnunet_images, output_dir_nnunet, save_probabilities=False,
                                                       overwrite=True, folder_with_segs_from_prev_stage=None)
        print("Predictions saved to:", os.listdir(output_dir_nnunet))
        
       # === Final output folder (MANDATORY name) ===
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        os.makedirs(output_dir_final, exist_ok=True)

        # === Optional post-processing step ===
        # For example, you can threshold the predictions or apply morphological operations
        # Here, we iterate through the predicted segmentations and apply the breast mask to each segmentation
        # to remove false positives outside the breast region
        for patient_id in self.dataset.get_patient_id_list():
            seg_path = os.path.join(output_dir_nnunet, f"{patient_id}_0001.nii.gz")
            if not os.path.exists(seg_path):
                print(f'{seg_path} NOT FOUND!')
                continue
            
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            
            patient_info = self.dataset.read_json_file(patient_id)
            if not patient_info or "primary_lesion" not in patient_info:
                continue
            
            coords = patient_info["primary_lesion"]["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]
            
            masked_segmentation = np.zeros_like(segmentation_array)
            masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
                segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]            
            masked_seg_image = sitk.GetImageFromArray(masked_segmentation)
            masked_seg_image.CopyInformation(segmentation)

            # MANDATORY: the segmentation masks should be named using the patient_id
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            sitk.WriteImage(masked_seg_image, final_seg_path)

        # Save path for Task 2 if needed
        self.predicted_segmentations = output_dir_final

        return output_dir_final
    
    def predict_classification(self, output_dir):
        """
        Task 2 — Predict treatment response (pCR).
        You MUST define this method if participating in Task 2.

        Args:
            output_dir (str): Directory to save output predictions.

        Returns:
            pd.DataFrame: DataFrame with patient_id, pcr prediction, and score.
        """
        patient_ids = self.dataset.get_patient_id_list()
        predictions = []
        
        for patient_id in patient_ids:
            image_paths = self.dataset.get_dce_mri_path_list(patient_id)

            images = []
            for image_index in range(3):
                image_path = image_paths[image_index]
                nii_image = nib.load(image_path)
                image_data = nii_image.get_fdata().astype(np.int16)
                image_data = image_data.transpose((2,1,0))
                image_data = np.expand_dims(image_data, axis=0)
                images.append(image_data)
            image_array = np.concatenate(images, axis=0)
            print('Image shape:', image_array.shape)

            seg_path = os.path.join(self.predicted_segmentations, f"{patient_id}.nii.gz")
            if not os.path.exists(seg_path):
                continue
            
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            print('Segmentation array shape:', segmentation_array.shape)
            segmentation_empty = not segmentation_array.any()
            if segmentation_empty:
                probability = 0
                pcr_prediction = 0
            else:
                cropped_image, _ = self._crop_to_largest_component(image_array, segmentation_array)
                input_image = torch.from_numpy(cropped_image)
                input_image.to(torch.device('cuda'))
                input_image = input_image.unsqueeze(0)
                input_image = torch.nn.functional.interpolate(input_image, (24, 75, 75), mode='trilinear')
                input_image = input_image.squeeze(0)
                results = np.zeros(5, 2)
                for index, model_path in enumerate(os.listdir(self.classification_model_folder)):
                    model_path_global = os.path.join(self.classification_model_folder, model_path)
                    model = torch.load(model_path_global)
                    model.eval()
                    model.to(torch.device('cuda'))
                    result = model(input_image)
                    results[index] = result.cpu().to_numpy()
                    print(result)
                mean_result = results.mean(axis=0)
                probability = mean_result[1]
                pcr_prediction = int(probability > 0.5)
            
            # === MANDATORY output format ===
            predictions.append({
                "patient_id": patient_id,
                "pcr": pcr_prediction,
                "score": probability
            })

        return pd.DataFrame(predictions)

    def _get_largest_component_crop(self, mask: np.ndarray) -> tuple[tuple, np.ndarray]:
        binary_mask = mask.astype(bool)
        
        # Find connected components
        labeled_array, num_components = ndimage.label(binary_mask, 
                                                      structure=ndimage.generate_binary_structure(mask.ndim, connectivity=3))
        
        if num_components == 0:
            empty_slice = tuple(slice(0, 0) for _ in range(mask.ndim))
            empty_mask = np.zeros_like(mask, dtype=bool)
            return empty_slice, empty_mask
        
        # Find the largest component by counting pixels in each label
        component_sizes = np.bincount(labeled_array.ravel())
        # Skip background (label 0)
        component_sizes[0] = 0
        largest_component_label = np.argmax(component_sizes)
        
        largest_component_mask = (labeled_array == largest_component_label)
        
        # Find bounding box of the largest component
        coords = np.where(largest_component_mask)
        
        # Create slice objects for each dimension
        crop_slice = tuple(
            slice(int(np.min(coord)), int(np.max(coord)) + 1) 
            for coord in coords
        )
        
        return crop_slice, largest_component_mask.astype(mask.dtype)


    def _crop_to_largest_component(self, array: np.ndarray, mask: np.ndarray = None) -> tuple[np.ndarray, tuple]:
        if mask is None:
            mask = array
        
        crop_slice, _ = self._get_largest_component_crop(mask)
        print(crop_slice)
        slice_with_channels = (slice(None),) + crop_slice
        print(slice_with_channels)
        print(array.shape)
        cropped_array = array[slice_with_channels]
        print(cropped_array.shape)
        
        return cropped_array, crop_slice