import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from functools import partial
import sys

FILENAME_PATTERN = "{case_name}_000{channel:d}.nii.gz"

def process_single_case_subtract_all_timepoints(case_path: Path, target_dir: Path):
    case_name = case_path.name.lower()
    nii_files = sorted(case_path.glob("*.nii.gz"))

    if not nii_files or len(nii_files) < 2:
        return case_name, False, {"error": "Not enough timepoints to subtract."}

    ref_nii = nib.load(str(nii_files[0]))
    ref_data = ref_nii.get_fdata().astype(np.int16)
    ref_affine = ref_nii.affine
    ref_shape = ref_nii.shape

    errors = []
    processed_count = 0

    for i, timepoint_path in enumerate(nii_files[1:], start=1):  # skip first timepoint
        try:
            nii_tp = nib.load(str(timepoint_path))
            data_tp = nii_tp.get_fdata().astype(np.int16)

            if data_tp.shape != ref_shape:
                raise ValueError(f"Shape mismatch: {timepoint_path.name} vs reference.")

            affine = ref_affine if np.allclose(nii_tp.affine, ref_affine) else nii_tp.affine

            subtraction = data_tp - ref_data

            sub_id = f"{case_name}_t{i}"

            native_img = nib.Nifti1Image(data_tp.astype(np.int16), affine, nii_tp.header)
            native_img.set_data_dtype(np.int16)
            nib.save(native_img, target_dir / f"{sub_id}_0000.nii.gz")

            subtract_img = nib.Nifti1Image(subtraction.astype(np.int16), affine, nii_tp.header)
            subtract_img.set_data_dtype(np.int16)
            nib.save(subtract_img, target_dir / f"{sub_id}sub_0000.nii.gz")

            processed_count += 1

        except Exception as e:
            errors.append(f"Timepoint {i}: {e}")

    if processed_count:
        return case_name, True, {
            "subtractions_created": processed_count,
            "total_timepoints": len(nii_files),
            "errors": errors
        }
    else:
        return case_name, False, {"error": "; ".join(errors)}



def process_subtraction_images(source_dir: Path, target_dir: Path):
    """
    Finds cases, calculates subtraction images (parallel),
    and saves them to the target directory.

    Args:
        source_dir: Path object for the source directory containing case subfolders.
        target_dir: Path object for the target directory to save subtraction images.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source directory: {source_dir.resolve()}")
    print(f"Target directory: {target_dir.resolve()}")

    case_paths = [d for d in source_dir.iterdir() if d.is_dir()]
    if not case_paths:
        print(f"No case subdirectories found in {source_dir}")
        return

    print(f"Found {len(case_paths)} potential cases.")

    print("\n--- Starting Subtraction Image Calculation (Parallel) ---")
    processed_case_info = {}
    results = []

    # Use functools.partial to fix arguments for the worker function
    worker_func = partial(process_single_case_subtract_all_timepoints,
                          target_dir=target_dir)

    max_workers = 16
    try:
         import os
         available_workers = len(os.sched_getaffinity(0))
         max_workers = min(max_workers, available_workers)
    except (ImportError, AttributeError, NotImplementedError):
         print(f"Could not determine available CPUs. Using default max_workers={max_workers}.")
         pass


    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_func, case_path) for case_path in case_paths]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(case_paths), desc="Processing Cases"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\nCritical error retrieving result for a task: {e}", file=sys.stderr)
                # We might not know which case failed here, add placeholder
                results.append((None, False, {"error": f"Future retrieval failed: {e}"}))


    print("\n--- Aggregating Results ---")
    successful_cases = 0
    partially_successful_cases = 0
    failed_cases = []
    for case_name, success, info in results:
        if success and info:
            processed_case_info[case_name] = info
            created = info.get('subtractions_created', 0)
            if created >= 1:
                successful_cases += 1
                status = "Success"

            error_info = ""
            if info.get('errors'):
                error_info = f" | Issues: {'; '.join(info['errors'])}"

            print(f"  Aggregated {status}: {case_name} | Subtractions Created: {created}{error_info}")

        elif case_name:
            failed_cases.append((case_name, info.get('error', 'Unknown error') if isinstance(info, dict) else 'Unknown state'))
            print(f"  Aggregated failure: {case_name} | Reason: {info.get('error', 'Unknown error') if isinstance(info, dict) else 'Unknown state'}")
        else:
             # Handle cases where the future itself failed, case_name might be None
             failed_cases.append(("Unknown Case", info.get('error', 'Future retrieval failed') if isinstance(info, dict) else 'Unknown state'))
             print(f"  Aggregated failure: Unknown Case | Reason: {info.get('error', 'Future retrieval failed') if isinstance(info, dict) else 'Unknown state'}")


    total_processed = successful_cases + partially_successful_cases + len(failed_cases)
    print(f"\n--- Processing Complete ---")
    print(f"Attempted to process {len(case_paths)} cases.")
    if total_processed != len(case_paths):
         print(f"Warning: Mismatch in processed count ({total_processed}) vs initial count ({len(case_paths)}).")

    print(f"Fully successful cases: {successful_cases}")
    print(f"Failed cases (0 subtractions): {len(failed_cases)}")

    if failed_cases:
        print("\nFailed case details:")
        for name, reason in failed_cases:
            print(f"  - {name}: {reason}")

    print(f"\nSubtraction images saved in {target_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate subtraction NIfTI images (Ch1-Ch0, optionally Ch2-Ch0) "
                    "from case subfolders and save them to a target directory."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to the source directory containing case subfolders (e.g., ./images)"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Path to the target directory where subtraction images will be saved (e.g., ./imagesSubtracted)"
    )
    args = parser.parse_args()

    # Basic validation
    if not args.source_dir.is_dir():
        print(f"Error: Source directory not found or is not a directory: {args.source_dir}")
        sys.exit(1)

    process_subtraction_images(args.source_dir, args.target_dir)