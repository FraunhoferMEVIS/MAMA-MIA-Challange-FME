import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm 

FILENAME_PATTERN = "{case_name}_000{channel:d}.nii.gz" # Using 4 digits as per example
EPSILON = 1e-8 # To avoid division by zero if standard deviation is zero


def zscore_normalize(img_data, mean, std):
    """Applies Z-score normalization."""
    return (img_data - mean) / (std + EPSILON)


def process_images(source_dir: Path, target_dir: Path):
    """
    Finds cases, normalizes images based on the first timepoint,
    copies them to the target directory, and pads missing channels.

    Args:
        source_dir: Path object for the source directory.
        target_dir: Path object for the target directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source directory: {source_dir.resolve()}")
    print(f"Target directory: {target_dir.resolve()}")

    case_paths = [d for d in source_dir.iterdir() if d.is_dir()]
    if not case_paths:
        print(f"No case subdirectories found in {source_dir}")
        return

    print(f"Found {len(case_paths)} potential cases.")

    processed_case_info = {} # Store info needed for padding: {case_name: {'count': int, 'mean0': float, 'affine': np.array, 'shape': tuple}}
    max_channels = 0

    print("\n--- Starting Pass 1: Normalization and Copying ---")
    for case_path in tqdm(case_paths, desc="Processing Cases"):
        case_name = case_path.name
        image_files = sorted(list(case_path.glob(f"{case_name}_*.nii.gz")))

        if not image_files:
            print(f"Warning: No images found for case {case_name}. Skipping.")
            continue

        first_channel_path = case_path / FILENAME_PATTERN.format(case_name=case_name, channel=0)

        if first_channel_path not in image_files:
             image_file_strs = [str(p) for p in image_files]
             if str(first_channel_path) not in image_file_strs:
                 print(f"Error: First channel ({first_channel_path.name}) "
                       f"not found for case {case_name}. Cannot calculate normalization stats. Skipping.")
                 continue

        try:
            nii_first = nib.load(str(first_channel_path)) # nib.load usually expects string path
            data_first = nii_first.get_fdata().astype(np.float32)

            mean_first = np.mean(data_first)
            std_first = np.std(data_first)
            affine_first = nii_first.affine
            shape_first = nii_first.shape

            print(f"  Case: {case_name} | Channels Found: {len(image_files)} | Mean (Ch 0): {mean_first:.4f} | Std Dev (Ch 0): {std_first:.4f}")

            if std_first < EPSILON:
                print(f"Warning: Standard deviation for first channel of {case_name} is close to zero ({std_first}). Normalization might produce NaNs or Infs.")

            processed_case_info[case_name] = {
                'count': len(image_files),
                'mean0': mean_first,
                'affine': affine_first,
                'shape': shape_first
            }
            max_channels = max(max_channels, len(image_files))

            for i, img_path in enumerate(image_files): 
                nii_img = nib.load(str(img_path)) 
                img_data = nii_img.get_fdata().astype(np.float32)

                normalized_data = zscore_normalize(img_data, mean_first, std_first)

                target_filename = FILENAME_PATTERN.format(case_name=case_name, channel=i)
                target_filepath = target_dir / target_filename 

                normalized_nii = nib.Nifti1Image(normalized_data.astype(np.float32), nii_img.affine, nii_img.header)
                normalized_nii.set_data_dtype(np.float32)

                nib.save(normalized_nii, str(target_filepath)) 

        except Exception as e:
            print(f"Error processing case {case_name} file {first_channel_path.name}: {e}")
            if case_name in processed_case_info:
                del processed_case_info[case_name]
            continue 

    if not processed_case_info:
        print("\nNo cases were successfully processed in Pass 1.")
        return

    print(f"\n--- Pass 1 Complete. Max channels found across all cases: {max_channels} ---")

    print("\n--- Starting Pass 2: Padding Missing Channels ---")
    padding_needed = any(info['count'] < max_channels for info in processed_case_info.values())

    if not padding_needed:
        print("No padding required, all processed cases have the maximum number of channels.")
        return

    for case_name, info in tqdm(processed_case_info.items(), desc="Padding Cases"):
        num_existing = info['count']
        num_to_pad = max_channels - num_existing

        if num_to_pad <= 0:
            continue

        print(f"  Padding Case: {case_name} | Adding {num_to_pad} channels (from {num_existing} to {max_channels})")

        mean_first = info['mean0']
        affine = info['affine']
        shape = info['shape']
        padding_value = -mean_first

        padding_data = np.full(shape, padding_value, dtype=np.float32)

        padding_nii = nib.Nifti1Image(padding_data, affine)
        padding_nii.set_data_dtype(np.float32) 

        for i in range(num_existing, max_channels):
            pad_filename = FILENAME_PATTERN.format(case_name=case_name, channel=i)
            pad_filepath = target_dir / pad_filename # Use Path object division
            try:
                nib.save(padding_nii, str(pad_filepath))
            except Exception as e:
                print(f"Error saving padding file {pad_filepath} for case {case_name}: {e}")

    print("\n--- Pass 2 Complete. Padding finished. ---")
    print(f"\nProcessing finished. Normalized images saved in {target_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize NIfTI images using Z-score based on the first channel, "
                    "copy to a target directory, and pad missing channels."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to the source directory containing case subfolders (e.g., ./images)"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Path to the target directory where normalized images will be saved (e.g., ./imagesTr)"
    )

    args = parser.parse_args()
    process_images(args.source_dir, args.target_dir)

    