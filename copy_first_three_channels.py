import argparse
import os
import shutil
import re
from multiprocessing import Pool
import sys

def copy_file(src_path, dst_path):
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"Error copying {src_path} to {dst_path}: {e}", file=sys.stderr)

def main(input_folder, output_folder, num_workers):
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found: {input_folder}", file=sys.stderr)
        sys.exit(1)

    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output folder {output_folder}: {e}", file=sys.stderr)
        sys.exit(1)

    file_pattern = re.compile(r".*_000[012]\.nii\.gz$")

    tasks = []
    print(f"Scanning input folder: {input_folder}")
    try:
        for filename in os.listdir(input_folder):
            if file_pattern.match(filename):
                src_path = os.path.join(input_folder, filename)
                dst_path = os.path.join(output_folder, filename)
                if os.path.isfile(src_path):
                    tasks.append((src_path, dst_path))
    except OSError as e:
         print(f"Error reading input folder {input_folder}: {e}", file=sys.stderr)
         sys.exit(1)

    if not tasks:
        print("No matching files found to copy.")
        return

    print(f"Found {len(tasks)} files to copy. Starting copy process with {num_workers} workers...")

    try:
        with Pool(processes=num_workers) as pool:
            pool.starmap(copy_file, tasks)
    except Exception as e:
         print(f"An error occurred during multiprocessing: {e}", file=sys.stderr)
         sys.exit(1)

    print("Copy process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy NIfTI files ending in _000[012].nii.gz from an input folder to an output folder using multiple processes."
    )
    parser.add_argument("input_folder", help="Path to the input folder containing source files.")
    parser.add_argument("output_folder", help="Path to the output folder where files will be copied.")
    args = parser.parse_args()

    num_workers = 6
    main(args.input_folder, args.output_folder, num_workers)