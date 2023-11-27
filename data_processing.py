import os, json, cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from albumentations import (
    VerticalFlip,
    HorizontalFlip,
    SafeRotate,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    Compose,
)

BLACK = [0, 0, 0]
YELLOW = [255, 255, 0]
RED = [255, 0, 0]


def preprocess_masks(root_dir: str):
    for folder in ["Train", "Valid"]:
        print("Preprocessing masks for {} dataset".format(folder))

        image_path = os.path.join(root_dir, folder, "Image")
        annotation_path = os.path.join(root_dir, folder, "Mask")

        # Get the list of image file names
        files = os.listdir(image_path)

        for fil in files:
            image_file = os.path.join(image_path, fil)
            img_id = image_file.split(os.path.sep)[-1].replace(".png", "")

            ann_file = os.path.join(annotation_path, img_id + "_mask.png")

            # check the image file and corresponding annotation file exists or not
            if os.path.exists(ann_file):
                mask = cv2.imread(ann_file)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                label_seg = np.zeros(mask.shape, dtype=np.uint8)
                label_seg[np.all(mask == [0, 0, 0], axis=-1)] = 0
                label_seg[np.all(mask == [255, 255, 0], axis=-1)] = 1
                label_seg[np.all(mask == [255, 0, 0], axis=-1)] = 2

                # Just take the first channel, no need for all 3 channels
                mask = label_seg[:, :, 0]

                cv2.imwrite(ann_file.replace("_mask", "_modified_mask"), mask)


class DataGenerator:
    def __init__(
        self, root, classes, resize, _type="Train", aug_params=None, preprocess_input=None
    ) -> None:
        self.root = root
        self.aug_params = aug_params
        self.classes = classes
        self.resize = resize
        self._type = _type
        self.preprocess_input = preprocess_input

        if self.aug_params:

            # augmentation params
            self.vertical_flip = aug_params.get("vertical_flip", False)
            self.horizontal_flip = aug_params.get("horizontal_flip", False)
            self.brightness = float(aug_params.get("brightness", 0.0))
            self.contrast = float(aug_params.get("contrast", 0.0))
            self.shift = float(aug_params.get("shift", 0.0))
            self.scale = float(aug_params.get("scale", 0.0))
            self.rotation = int(aug_params.get("rotation", 0))

            self.transforms = self.augmentation()

        if self._type == "Train":
            self.apply_augmentation = True
        else:
            self.apply_augmentation = False

        self.len_data = 0
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.img_shape = [self.resize[0], self.resize[1], 3]
        # After one hot encoding
        self.label_shape = [self.resize[0], self.resize[1], len(self.classes)]

        self.images = []
        self.masks = []
        self.read_data(_type)

    def augmentation(self):
        augmentation_list = []
        # Vertical and Horizontal Flip
        if self.vertical_flip:
            augmentation_list.append(VerticalFlip(p=0.5))

        if self.horizontal_flip:
            augmentation_list.append(HorizontalFlip(p=0.5))

        if self.scale or self.shift or self.rotation:
            # Shift-Scale-Rotate using ShiftScaleRotate method of albumentations
            augmentation_list.append(
                ShiftScaleRotate(
                    shift_limit=self.shift,
                    scale_limit=self.scale,
                    rotate_limit=self.rotation,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                )
            )

        # Brighness-Contrast using RandomBrightnessContrast method of albumentation
        self.brightness = (0, self.brightness)
        self.contrast = (0, self.contrast)
        augmentation_list.append(
            RandomBrightnessContrast(
                brightness_limit=self.brightness, contrast_limit=self.contrast, p=0.5
            )
        )

        compose = Compose(
            random.sample(
                augmentation_list,
                k=random.randrange(len(augmentation_list)),
            ),
            p=0.5,
        )
        return compose

    def read_data(self, data_sub_folder):
        image_path = os.path.join(self.root, data_sub_folder, "Image")
        annotation_path = os.path.join(self.root, data_sub_folder, "Mask")

        # Get the list of image file names
        files = os.listdir(image_path)

        for fil in files:
            image_file = os.path.join(image_path, fil)
            img_id = image_file.split(os.path.sep)[-1].replace(".png", "")

            ann_file = os.path.join(annotation_path, img_id + "_modified_mask.png")

            # check the image file and corresponding annotation file exists or not
            if os.path.exists(ann_file):
                # appending image and mask and file name
                self.images.append(image_file)
                self.masks.append(ann_file)

        self.len_data = len(self.images)

    def read_image(self, image_path, mask=False):
        if mask == True:
            channels = 1
        else:
            channels = 3

        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=channels)
        return image

    def aug_fn(self, image, mask):
        data = {"image": image, "mask": mask}
        aug_data = self.transforms(**data)
        image, mask = aug_data["image"], aug_data["mask"]

        return image, mask

    def apply_aug(self, image, mask):
        if self.apply_augmentation:
            image, mask = self.aug_fn(image, mask)

        if self.preprocess_input:
            image = self.preprocess_input(image)
        else:
            image = image / 255.0
        mask = to_categorical(mask, num_classes=len(self.classes))

        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        # image = tf.cast(image / 255.0, tf.float32)
        image = tf.image.resize(
            images=image, size=[self.resize[0], self.resize[1]], method="nearest"
        )

        mask = tf.image.resize(
            images=mask, size=[self.resize[0], self.resize[1]], method="nearest"
        )

        return image, mask

    def set_shapes(self, img, label):
        img.set_shape(self.img_shape)
        label.set_shape(self.label_shape)
        return img, label

    def load_data(self, image_path, mask_path):
        image = self.read_image(image_path)
        mask = self.read_image(mask_path, mask=True)

        image, mask = tf.numpy_function(
            func=self.apply_aug,
            inp=[image, mask],
            # Tout=(tf.float32, tf.uint8),
            Tout=(tf.float32, tf.float32),
        )

        return image, mask

    def get_generator(self, batch_size, _type="Train"):
        data_gen = tf.data.Dataset.from_tensor_slices((self.images, self.masks))
        data_gen = data_gen.map(
            self.load_data,
            num_parallel_calls=self.AUTOTUNE,
        ).prefetch(self.AUTOTUNE)
        data_gen = data_gen.map(self.set_shapes, num_parallel_calls=self.AUTOTUNE)
        data_gen = data_gen.repeat()
        data_gen = data_gen.batch(batch_size, drop_remainder=False)

        return self.len_data, data_gen
