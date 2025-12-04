import json
import cv2
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, segmentations_path, depths_path, images_path,  mode='training', subset_dim = None):

        self.depths = []
        for dirpath, _, filenames in os.walk(depths_path):
            for file in filenames:
                self.depths.append(os.path.join(dirpath, file))
        self.depths.sort()

        self.segmentations = []
        for dirpath, _, filenames in os.walk(segmentations_path):
            for file in filenames:
                if file.endswith("_gtFine_labelIds.png"):
                    self.segmentations.append(os.path.join(dirpath, file))
        self.segmentations.sort()

        self.images = []
        for dirpath, _, filenames in os.walk(images_path):
            for file in filenames:
                self.images.append(os.path.join(dirpath, file))
        self.images.sort()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 768), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 768), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        self.mode=mode

        # if subset_dim is not None:
            # self.segmentations

    def __len__(self):
        # return 300
        return len(self.segmentations)

    def __getitem__(self, idx):
        if idx > len(self.segmentations):
            raise IndexError("Out of bounds")
        segmentation_path = self.segmentations[idx]
        depth_path = self.depths[idx]
        image_path = self.images[idx]

        image = cv2.imread(image_path, -1)
        # print(image_path.split('/')[-1])
        # print(segmentation_path.split('/')[-1])    
        # print(depth_path.split('/')[-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation = np.repeat(np.expand_dims(cv2.imread(segmentation_path, -1), axis=-1), repeats=3, axis=-1)
        depth = np.repeat(np.expand_dims(cv2.imread(depth_path, -1), axis=-1), repeats=3, axis=-1)

        prompt = "A high quality image from a city street with cars, pedestrians, buildings, sky, and trees. 4K, Full HD, High Quality"

        if self.mode == 'validation':
            return dict(pixel_values=image_path, input_ids=prompt, conditioning_pixel_values_seg=segmentation_path, conditioning_pixel_values_depth=depth_path, hint_depth=depth)
        # (1024, 2048, 3)
        # print(image.shape, segmentation.shape, depth.shape)

        # Normalize source images to [0, 1].
        # segmentation = np.expand_dims(segmentation.astype(np.float32) / segmentation.max(), axis=-1)
        # depth = depth.astype(np.float32) / 255.
        # segmentation = segmentation.astype(np.float32) / 33.

        # Normalize target images to [-1, 1].
        # image = image.astype(np.float32) # - 128.) / 128. 
        # print(image.dtype, segmentation.dtype, depth.dtype)

        # image = torch.from_numpy(image)
        # segmentation = torch.from_numpy(segmentation)
        # depth = torch.from_numpy(depth)


        input_tokenizer = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        image = self.image_transforms(image)
        segmentation = self.conditioning_image_transforms(segmentation)
        depth = self.conditioning_image_transforms(depth)

        return dict(pixel_values=image, input_ids=input_tokenizer, conditioning_pixel_values_seg=segmentation, conditioning_pixel_values_depth=depth)
