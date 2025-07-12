import os
import pandas as pd
import shutil
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib
import onnxruntime
import json
import scipy.special
import time
from scipy import ndimage
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
        self.nnunet_model_folder = '/app/ingested_program/Dataset125_MAMA_MIA_expert_segmentations_1_subtraction/nnUNetTrainer__nnUNetResEncUNetLPlans24GB__3d_fullres'
        self.classification_model_folders = "/app/ingested_program/classification_models"
        

    def predict_segmentation(self, output_dir):
        """
        Task 1 — Predict tumor segmentation with nnUNetv2.
        You MUST define this method if participating in Task 1.

        Args:
            output_dir (str): Directory where predictions will be stored.

        Returns:
            str: Path to folder with predicted segmentation masks.
        """

        if os.environ.get('DEBUG_MAMA_MIA') == "True":
            segmentation_start_time = time.time()

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
            self.nnunet_model_folder,
            use_folds=(0,1,2,3,4),
            checkpoint_name='checkpoint_best.pth')
        
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
            
            # 1. Apply breast mask first
            patient_info = self.dataset.read_json_file(patient_id)
            if not patient_info or "primary_lesion" not in patient_info:
                # If no patient info or primary lesion, just proceed with the raw segmentation
                masked_segmentation = segmentation_array
            else:
                coords = patient_info["primary_lesion"]["breast_coordinates"]
                x_min, x_max = coords["x_min"], coords["x_max"]
                y_min, y_max = coords["y_min"], coords["y_max"]
                z_min, z_max = coords["z_min"], coords["z_max"]
                
                masked_segmentation = np.zeros_like(segmentation_array)
                masked_segmentation[x_min:x_max, y_min:y_max, z_min:z_max] = \
                    segmentation_array[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # 2. Keep only the largest connected component
            labeled_array, num_components = ndimage.label(masked_segmentation, 
                                                          structure=ndimage.generate_binary_structure(masked_segmentation.ndim, connectivity=3))

            if num_components > 0:
                component_sizes = np.bincount(labeled_array.ravel())
                component_sizes[0] = 0  # Ignore background
                largest_component_label = np.argmax(component_sizes)
                final_segmentation_array = (labeled_array == largest_component_label).astype(np.uint8)
            else:
                final_segmentation_array = np.zeros_like(masked_segmentation, dtype=np.uint8)

            masked_seg_image = sitk.GetImageFromArray(final_segmentation_array.astype(np.uint8)) # Ensure type is uint8 for binary
            masked_seg_image.CopyInformation(segmentation)

            # MANDATORY: the segmentation masks should be named using the patient_id
            final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
            sitk.WriteImage(masked_seg_image, final_seg_path)

        # Save path for Task 2 if needed
        self.predicted_segmentations = output_dir_final

        if os.environ.get('DEBUG_MAMA_MIA') == "True":
            segmentation_stop_time = time.time()
            segmentation_duration = segmentation_stop_time - segmentation_start_time
            print(f'Segmentation took {segmentation_duration} seconds.')

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
        
        if os.environ.get('DEBUG_MAMA_MIA') == "True":
            classification_start_time = time.time()
        
        patient_ids = self.dataset.get_patient_id_list()
        predictions = []
        if os.environ.get('DEBUG_MAMA_MIA') == "True":
            os.makedirs(os.path.join(output_dir, 'classification_inputs'), exist_ok=True)
        
        model_sessions = []
        model_resampling = []

        for model_folder in os.listdir(self.classification_model_folders):
            with open(os.path.join(self.classification_model_folders, model_folder, "config.json"), 'r') as file:
                config = json.load(file)
            resampling_size = config['target_size']
            model_path_global = os.path.join(self.classification_model_folders, model_folder, 'model.onnx')
            session = onnxruntime.InferenceSession(model_path_global, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            model_sessions.append(session)
            model_resampling.append(resampling_size)

        for patient_id in patient_ids:
            print(f'Classifying patient {patient_id}...')
            image_paths = self.dataset.get_dce_mri_path_list(patient_id)

            images = []
            for image_index in range(3):
                image_path = image_paths[image_index]
                nii_image = nib.load(image_path)
                image_data = nii_image.get_fdata().astype(np.float32)
                image_data = image_data.transpose((2,1,0))
                image_data = np.expand_dims(image_data, axis=0)
                images.append(image_data)
            image_array = np.concatenate(images, axis=0)

            seg_path = os.path.join(self.predicted_segmentations, f"{patient_id}.nii.gz")
            if not os.path.exists(seg_path):
                continue
            
            segmentation = sitk.ReadImage(seg_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation)
            segmentation_empty = not segmentation_array.any()
            if segmentation_empty:
                probability = 0
                pcr_prediction = 0
            else:
                cropped_image, _ = self._crop_to_largest_component(image_array, segmentation_array)
                mean = cropped_image[0].mean()
                std = cropped_image[0].std()
                cropped_image = (cropped_image - mean ) / std
                input_image = torch.from_numpy(cropped_image)
                input_image = input_image.unsqueeze(0)

                if os.environ.get('DEBUG_MAMA_MIA') == "True":
                    cropped_image_transposed = cropped_image.transpose((3,2,1,0))
                    cropped_nii = nib.Nifti1Image(cropped_image_transposed, nii_image.affine, nii_image.header)
                    cropped_nii_path = os.path.join(output_dir, "classification_inputs", f"{patient_id}.nii.gz")
                    nib.save(cropped_nii, cropped_nii_path)

                results = []
                for session, resampling_size in zip(model_sessions, model_resampling):
                    resampled_image = torch.nn.functional.interpolate(input_image, resampling_size, mode='trilinear').half()
                    flipping_dimensions = [tuple(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
                    for dimensions in flipping_dimensions:
                        flipped_image = torch.flip(resampled_image, dims=dimensions)
                        input_image_onnx = flipped_image.numpy()
                        logits = session.run(None, {'input': input_image_onnx})[0]
                        result = scipy.special.softmax(logits)
                        results.append(result)
                results_array = np.concatenate(results, axis=0)
                if os.environ.get('DEBUG_MAMA_MIA') == "True":
                    print(results_array)
                mean_result = results_array.mean(axis=0)
                probability = mean_result[1]
                pcr_prediction = int(probability > 0.5)
            
            # === MANDATORY output format ===
            predictions.append({
                "patient_id": patient_id,
                "pcr": pcr_prediction,
                "score": probability
            })
        prediction_df = pd.DataFrame(predictions)
        prediction_df.to_csv(os.path.join(output_dir, 'predictions.csv'))

        if os.environ.get('DEBUG_MAMA_MIA') == "True":
            classification_stop_time = time.time()
            classification_duration = classification_stop_time - classification_start_time
            print(f'Classification took {classification_duration} seconds.')

        return prediction_df

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
        slice_with_channels = (slice(None),) + crop_slice
        cropped_array = array[slice_with_channels]
        
        return cropped_array, crop_slice