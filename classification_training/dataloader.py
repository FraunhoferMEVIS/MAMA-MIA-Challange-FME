import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

class NiftiImageDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 data_split_file: str,
                 group: str,
                 target_size: tuple[int],
                 transforms: list = [],
                 normalization: str | None = None):
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.target_size = target_size  # Fixed voxel size (x, y, z)
        self.transforms = transforms
        self.normalization = normalization
        data_split_file_path = os.path.join(data_dir, data_split_file)
        with open(data_split_file_path, 'r') as file:
            data_split = json.load(file)
        self.case_names = data_split[group]

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        
        image_path = os.path.join(self.image_dir, f"{case_name}.nii.gz")
        label_path = os.path.join(self.label_dir, f"{case_name}.json")

        nii_image = nib.load(image_path)
        image_data = nii_image.get_fdata().astype(np.float32)

        image = torch.from_numpy(image_data).permute((3, 2, 1, 0))

        if self.target_size:
            image = image.unsqueeze(0)
            image = interpolate(image, self.target_size, mode='trilinear')  # Bilinear interpolation
            image = image.squeeze(0)

        if self.normalization == "zScoreFirstChannelBased":
            mean = image[0].mean()
            std = image[0].std()
            image = (image - mean) / std


        with open(label_path, 'r') as file:
            label_dict = json.load(file)
            label = label_dict['label']
            if label_dict['age'] == None:
                # Fill with mean value
                label_dict['age'] = float('nan')
            if label_dict['menopausal_status'][:3] == 'pre':
                label_dict['menopausal_status'] = 'pre'
            elif label_dict['menopausal_status'][:4] == 'peri':
                label_dict['menopausal_status'] = 'peri'
        
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms:
            for transform in self.transforms:
                image = transform(image)

        return image, label, label_dict

def main():
    parser = argparse.ArgumentParser(description="PyTorch Dataloader for NIfTI images with fixed image size.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the base directory containing 'images' and 'labels' subfolders.")
    parser.add_argument('--target_size', type=int, nargs=3, default=None,
                        help="Optional: Target image size (z, y, x) in voxels. E.g., --target_size 64 128 128")
    args = parser.parse_args()

    image_directory = os.path.join(args.dataset_path, 'images')
    label_directory = os.path.join(args.dataset_path, 'labels')

    if not os.path.isdir(image_directory):
        print(f"Error: Image directory not found at '{image_directory}'")
        return
    if not os.path.isdir(label_directory):
        print(f"Error: Label directory not found at '{label_directory}'")
        return

    dataset = NiftiImageDataset(
        image_dir=image_directory, 
        label_dir=label_directory,
        target_size=args.target_size
    )
    
    if len(dataset) == 0:
        print("No image files found in the specified directory.")
        return

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    images, labels = next(iter(dataloader))
    
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels: {labels}")

    fig, axes = plt.subplots(1, images.shape[0], figsize=(10, 5))
    if images.shape[0] == 1:
        axes = [axes]
        
    for i, (image, label) in enumerate(zip(images, labels)):
        image_numpy = image.squeeze().numpy()
        print(image_numpy.shape)
        central_slice_idx = image_numpy.shape[1] // 2
        ax = axes[i]
        image_for_display = image_numpy[:, central_slice_idx]
        image_for_display = image_for_display.swapaxes(0, 2)
        image_for_display = (image_for_display - image_for_display.min()) / (image_for_display.max() - image_for_display.min())
        ax.imshow(image_for_display, origin='lower')
        ax.set_title(f"Label: {label.item()}\nShape: {image_numpy.shape}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
