import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects

def process_patient_data(patient_id, input_folder, output_folder):
    """
    Loads MRI timepoints and segmentation for a patient, crops them,
    and saves the stacked result.
    """
    image_folder = os.path.join(input_folder, 'images', patient_id)
    segmentation_path = os.path.join(input_folder, 'segmentations', 'expert', f'{patient_id}.nii.gz')

    if not os.path.exists(segmentation_path):
        print(f"Warning: Segmentation for {patient_id} not found. Skipping.")
        return

    segmentation_image = nib.load(segmentation_path)
    segmentation_data = segmentation_image.get_fdata()

    bounding_box = find_objects(segmentation_data > 0)[0]

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.nii.gz')])
    if len(image_files) < 3:
        print(f"Warning: Less than 3 timepoints for {patient_id}. Skipping.")
        return

    first_three_timepoints = image_files[:3]
    cropped_images = []

    for i, image_file in enumerate(first_three_timepoints):
        image_path = os.path.join(image_folder, image_file)
        mri_image = nib.load(image_path)
        mri_data = mri_image.get_fdata()
        if i == 0:
            mean = mri_data.mean()
            std = mri_data.std()
        cropped_mri = mri_data[bounding_box]
        cropped_mri_normalized = (cropped_mri - mean) / std
        cropped_images.append(cropped_mri_normalized)

    stacked_cropped_images = np.stack(cropped_images, axis=-1)

    output_nifti_image = nib.Nifti1Image(stacked_cropped_images, segmentation_image.affine)
    output_filename = os.path.join(output_folder, f'{patient_id}_cropped_stacked.nii.gz')
    nib.save(output_nifti_image, output_filename)
    print(f"Processed and saved data for {patient_id}")


def main():
    """
    Main function to parse arguments and process all patients.
    """
    parser = argparse.ArgumentParser(description='Crop and stack breast MRI timepoints based on tumor segmentations.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    images_base_folder = os.path.join(args.input_folder, 'images')
    patient_ids = [d for d in os.listdir(images_base_folder) if os.path.isdir(os.path.join(images_base_folder, d))]

    for patient_id in patient_ids:
        process_patient_data(patient_id, args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()