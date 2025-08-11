# MAMA-MIA-Challenge (Team FME)

This repository collects the preprocessing, training and inference code for the [MAMA-MIA challenge](https://www.ub.edu/mama-mia/) submission of team FME.

Team Members: Kai Geissler, Jan Raphael Schäfer

## Final Submission

### Inference code
The inference code of our final submission for both tasks is in the directory `inference/code_submission_7`. 

### Training code

The data preparation for the segmentation task is the script `data_preprocessing_segmentation/copy_and_compute_first_subtraction_images.py` which we used to compute the first subtraction images from the MAMA-MIA dataset, which we than trained a 5-fold ensemble of nnU-Nets on:

```bash
nnUNetv2_plan_experiment -d 125 -pl nnUNetPlannerResEncL -c 3d_fullres
nnUNetv2_train 125 3d_fullres 0,1,2,3,4 -p nnUNetResEncUNetLPlans --c
```

The data preparation for the classification task was done using the script `data_prepocessing_classification/crop_and_copy_first_three_timepoints.py`. The training was performed with the script `hyperparameter_tuning/tune_random_search_3d.py`.

### Description of segmentation approach

The segmentation approach employed a 5-fold ensemble of nnU-Net models configured with the 3d_fullres setup. Data preprocessing involved computing the first difference DCE-MRI image, while nnU-Net handled normalization, resampling, and patch generation automatically. During inference, test-time augmentation via mirroring was applied, and connected component analysis was used as post-processing to retain only the largest connected lesion. The nnU-Net configuration included a voxel size of 2.0x0.7x0.7, a patch size of 80x256x256, and a batch size of 2.

### Description of classification approach

The classification approach used 3D models pretrained for video classification from the TorchVision library. The first three DCE-MRI images were selected as input, z-score normalized using the pre-contrast image, cropped to a bounding box around the lesion, and resampled to a fixed size. A hyperparameter search with Ray Tune optimized parameters such as learning rate, weight decay, and model architecture, resulting in an ensemble of 25 models trained with 5-fold cross-validation. Test-time augmentation was applied by flipping the input images along each axis, producing 8 augmented versions of each image, with predictions averaged across them. The classification models were trained using the AdamW optimizer, cosine annealing learning rate scheduler, and CCE loss function.

## Set up training dependencies
Install your favorite python package manager (e.g. miniconda).

```bash
conda create -n 'mamamia' python==3.12.9
conda activate mamamia
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Test inference Docker container
```bash
docker run -it --rm --entrypoint /bin/bash --gpus all --shm-size=1024m `
--mount type=bind,source=<this_repository_checkout_directory>/mama-mia-challenge/inference,target=/app/ingested_program `
--mount type=bind,source=<test_data_directoy>,target=/dataset `
--mount type=bind,source=<test_output_directory>,target=/output `
lgarrucho/codabench-gpu:latest `
 -c "python /app/ingested_program/test_model.py /dataset /output"
```

## Folder structure
- `breast_fgt_segmentation`: Segment breast and FGT in breast MRI
- `classification_training`: Train classification models
- `data_prepocessing_classification`: Preprocessing of classification data
- `data_preprocessing_segmentation`: Preprocessing of segmentation data
- `hyperparameter_tuning`: Hyperparameter tuning for classification models
- `inference`: Submission code
- `tests`: Small tests for selected components

