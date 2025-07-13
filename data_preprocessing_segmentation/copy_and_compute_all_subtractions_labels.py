import argparse
from pathlib import Path
import shutil
import re

def extract_base_case_name(image_filename: str) -> str:
    """
    Extract original case name (e.g., 'casea') from filename like 'casea_t1_0000.nii.gz'
    """
    match = re.match(r"([a-zA-Z0-9]+_[0-9]+)_t\d+(sub)?_0000\.nii\.gz", image_filename)
    return match.group(1) if match else None

def get_new_label_name(image_filename: str) -> str:
    """
    Given 'casea_t1_0000.nii.gz', return 'casea_t1.nii.gz'
    """
    return image_filename.replace("_0000", "")

def copy_and_rename_labels(generated_image_dir: Path, labels_source_dir: Path, labels_target_dir: Path):
    labels_target_dir.mkdir(parents=True, exist_ok=True)

    new_images = list(generated_image_dir.glob("*_0000.nii.gz"))
    if not new_images:
        print(f"No new images found in {generated_image_dir}")
        return

    copied = 0
    skipped = 0

    for img_path in new_images:
        img_name = img_path.name
        base_case = extract_base_case_name(img_name)
        if not base_case:
            print(f"Warning: Could not extract original case name from {img_name}")
            skipped += 1
            continue

        label_source_path = labels_source_dir / f"{base_case}.nii.gz"
        if not label_source_path.exists():
            print(f"Label not found for {base_case}, skipping.")
            skipped += 1
            continue

        new_label_name = get_new_label_name(img_name)
        target_label_path = labels_target_dir / new_label_name
        shutil.copy(str(label_source_path), str(target_label_path))
        copied += 1

    print(f"\n--- Label Copy Summary ---")
    print(f"Copied: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {labels_target_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy segmentation masks to match renamed subtraction images (without _0000)."
    )
    parser.add_argument("generated_image_dir", type=Path, help="Directory with newly created images (e.g., ./imagesSubtracted)")
    parser.add_argument("labels_source_dir", type=Path, help="Directory with original labels (e.g., ./labelsTr)")
    parser.add_argument("labels_target_dir", type=Path, help="Target directory to store renamed labels (e.g., ./labelsSubtracted)")

    args = parser.parse_args()

    if not args.generated_image_dir.is_dir():
        print(f"Error: Generated image directory not found: {args.generated_image_dir}")
        exit(1)
    if not args.labels_source_dir.is_dir():
        print(f"Error: Label source directory not found: {args.labels_source_dir}")
        exit(1)

    copy_and_rename_labels(args.generated_image_dir, args.labels_source_dir, args.labels_target_dir)
