import argparse
import os
import nibabel as nib
import numpy as np
from pathlib import Path


def normalize_image(image_array, min_cutoff=0.001, max_cutoff=0.001):
    """
    Normalize the intensity of an image array by cutting off min and max values
    to a certain percentile and set all values above/below that percentile to
    the new max/min.

    Parameters
    ----------
    image_array: np.array
        3D numpy array
    min_cutoff: float
        Minimum percentile of image to keep. (0.1% = 0.001)
    max_cutoff: float
        Maximum percentile of image to keep. (0.1% = 0.001)

    Returns
    -------
    np.array
        Normalized image

    """
    # Sort image values
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * (1 - max_cutoff))
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def zscore_image(image_array):
    """
    Convert intensity values in an image to zscores:
    zscore = (intensity_value - mean) / standard_deviation

    Parameters
    ----------
    image_array: np.array
        3D numpy array
    Returns
    -------
    np.array
        Image with zscores for values

    """
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)
    return image_array


def process_nifti_to_npy(input_dir, output_dir):
    """
    Processes NIfTI MRI data, normalizes intensity, and saves as .npy files
    for use with the breast segmentation model.

    Args:
        input_dir (str): Path to the input directory containing subject folders.
                         Each subject folder should contain a precontrast .nii.gz file.
        output_dir (str): Path to the output directory where processed .npy files will be saved.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting processing from: {input_path}")
    print(f"Saving processed files to: {output_path}")

    for subject_folder in os.listdir(input_path):
        subject_path = input_path / subject_folder
        if not subject_path.is_dir():
            continue

        print(f"\nProcessing subject: {subject_folder}")
        precontrast_nifti_found = False

        for nifti_file in os.listdir(subject_path):
            if '0000.nii.gz' in nifti_file.lower():
                nifti_filepath = subject_path / nifti_file
                precontrast_nifti_found = True
                break
        
        if not precontrast_nifti_found:
            print(f"  No precontrast NIfTI file found in {subject_folder}. Skipping.")
            continue

        try:
            nifti_img = nib.load(nifti_filepath)
            image_array = nifti_img.get_fdata().astype(np.float32)

            original_affine = nifti_img.affine
            original_orientation = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(original_affine))
            target_orientation = nib.orientations.axcodes2ornt(('L', 'P', 'I'))
            transform = nib.orientations.ornt_transform(original_orientation, target_orientation)
            image_array = nib.apply_orientation(image_array, transform)

            # Intensity Normalization
            normalized_image_array = zscore_image(normalize_image(image_array))
            
            os.mkdir(output_path / subject_folder)
            output_filename = f"{subject_folder}_preprocessed.npy"
            output_filepath = output_path / subject_folder / output_filename

            np.save(output_filepath, normalized_image_array)
            print(f"  Successfully processed and saved {output_filename}")

        except Exception as e:
            print(f"  Error processing {subject_folder}: {e}")

    print("\nPreprocessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert NIfTI breast MRI data to .npy format for deep learning model input.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir', type=str, required=True,
        help='Path to the input directory containing subject folders (e.g., /path/to/my_mri_data/). '
             'Each subfolder should contain the precontrast NIfTI file (e.g., DUKE_001/duke_001_0000.nii.gz).'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Path to the output directory where processed .npy files will be saved.'
    )

    args = parser.parse_args()

    process_nifti_to_npy(args.input_dir, args.output_dir)