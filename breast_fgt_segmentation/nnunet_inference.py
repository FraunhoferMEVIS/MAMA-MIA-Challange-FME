import argparse
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main(input_folder: str, output_folder: str, nnunet_model_folder: str):
    predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

    predictor.initialize_from_trained_model_folder(
        nnunet_model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth')
    
    predictor.predict_from_files(input_folder, output_folder)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', help='Input folder for segmentation')
    parser.add_argument('output_folder', help= 'Output folder for segmentation masks')
    parser.add_argument('model_folder', help= 'Path of nnU-Net model folder')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.model_folder)

    