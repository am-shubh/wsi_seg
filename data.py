import os
import cv2
import albumentations
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from constants import *


class Dataset(BaseDataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    # Classes defined as RGB color values
    BLACK_PIXEL = [0, 0, 0]
    YELLOW_PIXEL = [255, 255, 0]
    RED_PIXEL = [255, 0, 0]

    CLASSES = [BLACK_PIXEL, YELLOW_PIXEL, RED_PIXEL]

    COLOR_MAP = {index: color for index, color in enumerate(CLASSES)}

    def __init__(
        self,
        images_dir,
        masks_dir,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.ids = [ids.replace(".png", "") for ids in self.ids]
        self.images_fps = [
            os.path.join(images_dir, image_id + ".png") for image_id in self.ids
        ]
        self.masks_fps = [
            os.path.join(masks_dir, image_id + "_mask.png") for image_id in self.ids
        ]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.mean = np.array([0.6075])
        self.std = np.array([0.3584])

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OR use pre-processing functions from Model itself
        image = image / 255
        image = (image - self.mean) / self.std

        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        label_seg = np.zeros(mask.shape, dtype=np.uint8)
        label_seg[np.all(mask == self.BLACK_PIXEL, axis=-1)] = 0
        label_seg[np.all(mask == self.YELLOW_PIXEL, axis=-1)] = 1
        label_seg[np.all(mask == self.RED_PIXEL, axis=-1)] = 2

        # Just take the first channel, no need for all 3 channels
        mask = label_seg[:, :, 0].astype("float")
        mask = np.expand_dims(mask, axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.ShiftScaleRotate(
        #     scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0
        # ),
    ]
    return albumentations.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # albumentations.Lambda(image=preprocessing_fn),
        albumentations.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albumentations.Compose(_transform)


def get_data_loader(config, preprocessing_fn):
    logging.info("Creating Dataset loaders for Train and Valid")

    preprocessing = get_preprocessing(preprocessing_fn)

    train_dataset = Dataset(
        os.path.join(TRAIN_DIRECTORY, "Image"),
        os.path.join(TRAIN_DIRECTORY, "Mask"),
        augmentation=get_training_augmentation(),
        preprocessing=preprocessing,
    )

    valid_dataset = Dataset(
        os.path.join(VAL_DIRECTORY, "Image"),
        os.path.join(VAL_DIRECTORY, "Mask"),
        augmentation=None,
        preprocessing=preprocessing,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["valid_batch_size"],
        shuffle=False
    )

    return train_loader, valid_loader