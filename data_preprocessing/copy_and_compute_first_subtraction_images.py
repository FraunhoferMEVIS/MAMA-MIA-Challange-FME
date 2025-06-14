import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from functools import partial
import sys

FILENAME_PATTERN = "{case_name}_000{channel:d}.nii.gz"

def process_single_case_subtract(case_path: Path, target_dir: Path, filename_pattern: str, num_subtractions: int):
    """
    Processes a single case: loads required channels, calculates subtraction images
    (channel N - channel 0), saves them, and returns collected info.

    Args:
        case_path: Path to the case directory.
        target_dir: Path to the target directory for saving subtraction images.
        filename_pattern: String pattern for image filenames.
        num_subtractions: Number of subtraction images to compute (1 or 2).
                          1: Computes (channel 1 - channel 0)
                          2: Computes (channel 1 - channel 0) and (channel 2 - channel 0)

    Returns:
        tuple: (case_name: str, success: bool, info: dict | None)
               info dict contains {'subtractions_created', 'affine0', 'shape0'} on success,
               or {'error'} on failure.
    """
    case_name = case_path.name.lower()

    # --- 1. Load Channel 0 (Reference) ---
    ch0_filename = filename_pattern.format(case_name=case_name, channel=0)
    ch0_path = case_path / ch0_filename
    if not ch0_path.exists():
        potential_ch0 = list(case_path.glob(f"{case_name}_*0000.nii.gz"))
        if potential_ch0:
             ch0_path = potential_ch0[0]
             print(f"Warning: Using {ch0_path.name} as channel 0 for case {case_name}.")
        else:
             potential_ch0_alt = sorted(list(case_path.glob(f"{case_name}_*.nii.gz")))
             if potential_ch0_alt:
                 ch0_path = potential_ch0_alt[0]
                 print(f"Warning: Assuming first file {ch0_path.name} is channel 0 for case {case_name}.")
             else:
                 return case_name, False, {"error": f"Channel 0 ({ch0_filename}) not found."}

    try:
        nii_ch0 = nib.load(str(ch0_path))
        data_ch0 = nii_ch0.get_fdata().astype(np.int16)
        affine_ch0 = nii_ch0.affine
        shape_ch0 = nii_ch0.shape
    except Exception as e:
        return case_name, False, {"error": f"Error loading channel 0 ({ch0_path.name}): {e}"}

    subtractions_created = 0
    errors = []

    # --- 2. Process Subtraction Channels ---
    for i in range(1, num_subtractions + 1):
        chN_filename = filename_pattern.format(case_name=case_name, channel=i)
        chN_path = case_path / chN_filename

        if not chN_path.exists():
             all_files = sorted([p for p in case_path.glob(f"{case_name}_*.nii.gz") if p != ch0_path])
             if len(all_files) >= i:
                 chN_path = all_files[i-1] # 0-based index for files after channel 0
                 print(f"Warning: Using {chN_path.name} as channel {i} for case {case_name}.")
             else:
                 error_msg = f"Channel {i} ({chN_filename}) not found."
                 print(f"Error for case {case_name}: {error_msg}")
                 errors.append(error_msg)
                 continue # Skip to next required channel if possible

        try:
            nii_chN = nib.load(str(chN_path))
            data_chN = nii_chN.get_fdata().astype(np.int16)

            # --- Sanity Checks ---
            if nii_chN.shape != shape_ch0:
                raise ValueError(f"Shape mismatch in {chN_path.name} ({nii_chN.shape}) vs channel 0 ({shape_ch0})")

            current_affine = nii_chN.affine
            if not np.allclose(current_affine, affine_ch0):
                print(f"Warning: Affine mismatch in {chN_path.name} vs channel 0 for case {case_name}. Using affine from channel {i}.")
            else:
                current_affine = affine_ch0 # Prefer channel 0 affine if they match

            subtraction_data = data_chN - data_ch0

            target_filename = filename_pattern.format(case_name=case_name, channel=i-1)
            target_filepath = target_dir / target_filename

            subtraction_nii = nib.Nifti1Image(subtraction_data.astype(np.int16), current_affine, nii_chN.header)
            subtraction_nii.set_data_dtype(np.int16)

            nib.save(subtraction_nii, str(target_filepath))
            subtractions_created += 1

        except Exception as e:
            error_msg = f"Error processing channel {i} ({chN_path.name}): {e}"
            print(f"Error for case {case_name}: {error_msg}")
            errors.append(error_msg)

    # --- 3. Return Results ---
    if subtractions_created > 0:
        case_info = {
            'subtractions_created': subtractions_created,
            'affine0': affine_ch0,
            'shape0': shape_ch0,
            'errors': errors # Include any non-fatal errors encountered
        }
        return case_name, True, case_info
    else:
         return case_name, False, {"error": "; ".join(errors) if errors else "Failed to create any subtraction images."}


def process_subtraction_images(source_dir: Path, target_dir: Path, num_subtractions: int):
    """
    Finds cases, calculates subtraction images (parallel),
    and saves them to the target directory.

    Args:
        source_dir: Path object for the source directory containing case subfolders.
        target_dir: Path object for the target directory to save subtraction images.
        num_subtractions: Number of subtraction images to compute (1 or 2).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source directory: {source_dir.resolve()}")
    print(f"Target directory: {target_dir.resolve()}")
    print(f"Number of subtractions requested: {num_subtractions} (Ch1-Ch0{', Ch2-Ch0' if num_subtractions == 2 else ''})")

    case_paths = [d for d in source_dir.iterdir() if d.is_dir()]
    if not case_paths:
        print(f"No case subdirectories found in {source_dir}")
        return

    print(f"Found {len(case_paths)} potential cases.")

    print("\n--- Starting Subtraction Image Calculation (Parallel) ---")
    processed_case_info = {}
    results = []

    # Use functools.partial to fix arguments for the worker function
    worker_func = partial(process_single_case_subtract,
                          target_dir=target_dir,
                          filename_pattern=FILENAME_PATTERN,
                          num_subtractions=num_subtractions)

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
            if created == num_subtractions:
                successful_cases += 1
                status = "Success"
            else:
                partially_successful_cases +=1
                status = f"Partial Success ({created}/{num_subtractions})"

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

    print(f"Fully successful cases ({num_subtractions}/{num_subtractions} subtractions): {successful_cases}")
    print(f"Partially successful cases (<{num_subtractions}/{num_subtractions} subtractions): {partially_successful_cases}")
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
    parser.add_argument(
        "--num_subtractions",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of subtraction images to compute: 1 for (Ch1-Ch0), 2 for (Ch1-Ch0) and (Ch2-Ch0). Default is 2."
    )
    args = parser.parse_args()

    # Basic validation
    if not args.source_dir.is_dir():
        print(f"Error: Source directory not found or is not a directory: {args.source_dir}")
        sys.exit(1)

    process_subtraction_images(args.source_dir, args.target_dir, args.num_subtractions)