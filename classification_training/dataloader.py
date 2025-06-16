import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

class NiftiImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_size=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size  # Fixed voxel size (x, y, z)
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        patient_name = image_name.split('.nii.gz')[0]
        
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, f"{patient_name}.txt")

        nii_image = nib.load(image_path)
        image_data = nii_image.get_fdata().astype(np.float32)

        image = torch.from_numpy(image_data).permute((3, 2, 1, 0))

        if self.target_size:
            image = image.unsqueeze(0)
            image = interpolate(image, self.target_size, mode='trilinear')  # Bilinear interpolation
            image = image.squeeze(0)

        with open(label_path, 'r') as f:
            label = int(f.read().strip())
        
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

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
