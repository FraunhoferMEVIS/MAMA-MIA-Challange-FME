import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures 
from functools import partial

FILENAME_PATTERN = "{case_name}_000{channel:d}.nii.gz" # Using 4 digits as per example
EPSILON = 1e-8 # To avoid division by zero if standard deviation is zero


def zscore_normalize(img_data, mean, std):
    """Applies Z-score normalization."""
    return (img_data - mean) / (std + EPSILON)


def process_single_case(case_path: Path, target_dir: Path, filename_pattern: str, epsilon: float):
    """
    Processes a single case: loads images, calculates stats from the first channel,
    normalizes, saves images, and returns collected info.

    Args:
        case_path: Path to the case directory.
        target_dir: Path to the target directory for saving normalized images.
        filename_pattern: String pattern for image filenames.
        epsilon: Small value to prevent division by zero.

    Returns:
        tuple: (case_name: str, success: bool, info: dict | None)
               info dict contains {'count', 'mean0', 'affine', 'shape'} on success.
    """
    case_name = case_path.name.lower()
    image_files = sorted(list(case_path.glob(f"{case_name}_*.nii.gz")))

    if not image_files:
        return case_name, False, {"error": "No images found"}

    first_channel_path = case_path / filename_pattern.format(case_name=case_name, channel=0)

    if first_channel_path not in image_files:
         image_file_strs = [str(p) for p in image_files]
         if str(first_channel_path) not in image_file_strs:
            return case_name, False, {"error": f"First channel {first_channel_path.name} not found"}

    try:
        nii_first = nib.load(str(first_channel_path)) # nib.load usually expects string path
        data_first = nii_first.get_fdata().astype(np.float32)

        mean_first = np.mean(data_first)
        std_first = np.std(data_first)
        affine_first = nii_first.affine
        shape_first = nii_first.shape

        if std_first < epsilon:
            print(f"Warning: Standard deviation for first channel of {case_name} is close to zero ({std_first}). Normalization might produce NaNs or Infs.")

        for i, img_path in enumerate(image_files):
            nii_img = nib.load(str(img_path))
            img_data = nii_img.get_fdata().astype(np.float32)

            if nii_img.shape != shape_first:
                 raise ValueError(f"Shape mismatch in {img_path.name} ({nii_img.shape}) vs first channel ({shape_first})")
            if not np.allclose(nii_img.affine, affine_first):
                 print(f"Warning: Affine mismatch in {img_path.name} vs first channel for case {case_name}. Using affine from current image.")
                 current_affine = nii_img.affine
            else:
                 current_affine = affine_first


            normalized_data = zscore_normalize(img_data, mean_first, std_first)

            target_filename = filename_pattern.format(case_name=case_name, channel=i)
            target_filepath = target_dir / target_filename

            normalized_nii = nib.Nifti1Image(normalized_data.astype(np.float32), current_affine, nii_img.header)
            normalized_nii.set_data_dtype(np.float32)

            nib.save(normalized_nii, str(target_filepath))

        case_info = {
            'count': len(image_files),
            'mean0': mean_first,
            'std0': std_first,
            'affine': affine_first,
            'shape': shape_first
        }
        return case_name, True, case_info

    except Exception as e:
        print(f"Error processing case {case_name} in worker: {e}")
        return case_name, False, {"error": str(e)}


def process_images(source_dir: Path, target_dir: Path):
    """
    Finds cases, normalizes images based on the first timepoint (in parallel),
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

    print("\n--- Starting Pass 1: Normalization and Copying (Parallel) ---")
    processed_case_info = {}
    max_channels = 0
    results = []

    # Use functools.partial to fix arguments for the worker function
    # This makes it easy to use with executor.submit or executor.map
    worker_func = partial(process_single_case,
                          target_dir=target_dir,
                          filename_pattern=FILENAME_PATTERN,
                          epsilon=EPSILON)

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(worker_func, case_path) for case_path in case_paths]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(case_paths), desc="Processing Cases (Pass 1)"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error retrieving result for a task: {e}")


    print("\n--- Aggregating results from Pass 1 ---")
    successful_cases = 0
    failed_cases = []
    for case_name, success, info in results:
        if success and info:
            processed_case_info[case_name] = info
            max_channels = max(max_channels, info['count'])
            successful_cases += 1
            print(f"  Aggregated success: {case_name} | Channels: {info['count']} | Mean (Ch 0): {info.get('mean0', 'N/A'):.4f}")
        elif case_name:
             failed_cases.append((case_name, info.get('error', 'Unknown error') if isinstance(info, dict) else 'Unknown state'))
             print(f"  Aggregated failure: {case_name} | Reason: {info.get('error', 'Unknown error') if isinstance(info, dict) else 'Unknown state'}")
        else:
             print(f"  Aggregated failure: Unknown case | Reason: {info.get('error', 'Unknown error') if isinstance(info, dict) else 'Unknown state'}")

    if not processed_case_info:
        print("\nNo cases were successfully processed in Pass 1.")
        if failed_cases:
             print("Failed cases:", failed_cases)
        return

    print(f"\n--- Pass 1 Complete. Successfully processed {successful_cases}/{len(case_paths)} cases. Max channels found: {max_channels} ---")
    if failed_cases:
        print(f"Failed to process {len(failed_cases)} cases.")


    print("\n--- Starting Pass 2: Padding Missing Channels ---")
    padding_needed = any(info['count'] < max_channels for info in processed_case_info.values())

    if not padding_needed:
        print("No padding required, all processed cases have the maximum number of channels.")
        print(f"\nProcessing finished. Normalized images saved in {target_dir.resolve()}")
        return

    for case_name, info in tqdm(processed_case_info.items(), desc="Padding Cases (Pass 2)"):
        num_existing = info['count']
        num_to_pad = max_channels - num_existing

        if num_to_pad <= 0:
            continue

        print(f"  Padding Case: {case_name} | Adding {num_to_pad} channels (from {num_existing} to {max_channels})")

        mean_first = info['mean0']
        std_first = info['std0']
        affine = info['affine']
        shape = info['shape']

        padding_value = -mean_first / (std_first + EPSILON)

        padding_data = np.full(shape, padding_value, dtype=np.float32)
        padding_nii = nib.Nifti1Image(padding_data, affine)
        padding_nii.set_data_dtype(np.float32)

        for i in range(num_existing, max_channels):
            pad_filename = FILENAME_PATTERN.format(case_name=case_name, channel=i)
            pad_filepath = target_dir / pad_filename
            try:
                nib.save(padding_nii, str(pad_filepath))
            except Exception as e:
                print(f"Error saving padding file {pad_filepath} for case {case_name}: {e}")

    print("\n--- Pass 2 Complete. Padding finished. ---")
    print(f"\nProcessing finished. Normalized images saved in {target_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize NIfTI images using Z-score based on the first channel (parallel pass 1), "
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

    