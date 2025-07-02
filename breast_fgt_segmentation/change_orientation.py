import argparse
import os
import nibabel as nib


def change_orientation(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for nifti_file in os.listdir(input_dir):
        nifti_filepath = os.path.join(input_dir, nifti_file)
        nifti_img = nib.load(nifti_filepath)

        original_affine = nifti_img.affine
        original_orientation = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(original_affine))
        target_orientation = nib.orientations.axcodes2ornt(('L', 'P', 'I'))
        transform = nib.orientations.ornt_transform(original_orientation, target_orientation)

        output_image = nifti_img.as_reoriented(transform)
        output_filepath = os.path.join(output_dir, nifti_file)
        nib.save(output_image, output_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalize NIfTI orientation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir', type=str, required=True,
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Path to the output directory where processed .nii.gz files will be saved.'
    )
    args = parser.parse_args()

    change_orientation(args.input_dir, args.output_dir)

