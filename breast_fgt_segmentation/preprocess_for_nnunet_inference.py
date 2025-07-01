import argparse
import shutil
import os


def main(input_folder: str, output_folder: str):
    for patient_folder in os.listdir(input_folder):
        first_timepoint_path = os.path.join(input_folder, patient_folder, f'{patient_folder.lower()}_0000.nii.gz')
        shutil.copy2(first_timepoint_path, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)