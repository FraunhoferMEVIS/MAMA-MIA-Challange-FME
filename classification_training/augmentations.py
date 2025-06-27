import random
import numpy as np
import torch
from typing import List
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform, BrightnessAdditiveTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

# TODO: Add random zoom transformation

def random_mirroring_2d(image: torch.Tensor,
                     mirror_axes: list[str]=['y', 'x'],
                     apply_probability=0.5,
                     probability_per_axis=0.5) -> torch.Tensor:
    if random.random() < apply_probability:
        flip_dims = []
        for axis in mirror_axes:
            if axis == 'y' and random.random() < probability_per_axis:
                flip_dims.append(1)
            elif axis == 'x' and random.random() < probability_per_axis:
                flip_dims.append(2)
        if flip_dims:
            image = torch.flip(image, dims=flip_dims)
    return image

def random_mirroring(image: torch.Tensor,
                     mirror_axes: list[str]=['z', 'y', 'x'],
                     apply_probability=0.5,
                     probability_per_axis=0.5) -> torch.Tensor:
    if random.random() < apply_probability:
        flip_dims = []
        for axis in mirror_axes:
            if axis == 'z' and random.random() < probability_per_axis:
                flip_dims.append(1)
            elif axis == 'y' and random.random() < probability_per_axis:
                flip_dims.append(2)
            elif axis == 'x' and random.random() < probability_per_axis:
                flip_dims.append(3)
        if flip_dims:
            image = torch.flip(image, dims=flip_dims)
    return image

def batch_generators_spatial_augmentations(image: torch.Tensor) -> torch.Tensor:
    dimensions = len(image.shape) - 1
    transforms: List[BasicTransform] = [
        RandomTransform(
            SpatialTransform(
                patch_size=image.shape[1:],
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.25,
                p_scaling=0.3,
                # Need to synchronize axis scaling in 2D case because of bug in batchgeneratorsv2
                p_synchronize_scaling_across_axes=1 if dimensions==2 else 0
            ),
            apply_probability=0.1
        ),
    ]
    transforms = ComposeTransforms(transforms)
    bg_result = transforms(image=image)
    augmented_image = bg_result['image']
    return augmented_image


def batch_generators_intensity_augmentations(image: torch.Tensor,
                                             do_intensity_augmentations_per_channel: bool = True) -> torch.Tensor:
    synchronize_channels = not do_intensity_augmentations_per_channel
    transforms: List[BasicTransform] = [
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 1),
                synchronize_channels=synchronize_channels
            ),
            apply_probability=0.1
        ),
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=synchronize_channels,
                p_per_channel=1,
            ),
            apply_probability=0.2
        ),
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=(0.75, 1.25),
                synchronize_channels=synchronize_channels
            ),
            apply_probability=0.15
        ),
        RandomTransform(
           BrightnessAdditiveTransform(
            mu=0,
            sigma=10,
            per_channel=do_intensity_augmentations_per_channel,
            p_per_channel=1
            ),
            apply_probability=0.15
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=(0.75, 1.25),
                preserve_range=False,
                synchronize_channels=synchronize_channels
            ),
            apply_probability=0.15
        ),
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=synchronize_channels,
                synchronize_axes=False,
                ignore_axes=tuple(),
                p_per_channel=1
            ),
            apply_probability=0.25
        ),
        RandomTransform(
            GammaTransform(
                gamma=(0.7, 1.5),
                p_invert_image=0.25,
                synchronize_channels=synchronize_channels,
                p_per_channel=1,
                p_retain_stats=1
            ),
            apply_probability=0.4
        )
    ]
    transforms = ComposeTransforms(transforms)
    bg_result = transforms(image=image)
    augmented_image = bg_result['image']
    return augmented_image
