import os, shutil
import cv2
import random
import numpy as np
import json
from matplotlib import pyplot as plt
from patchify import patchify
from sklearn.model_selection import train_test_split

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image


root_directory = 'Data/'

output_dir = "unet_512/Dataset"
output_sub_dirs = ["Train", "Valid"]

config = {
    "patch_size": 512
}

patch_size = config["patch_size"]
step_size = int(patch_size * 0.8)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

for sub_dirs in output_sub_dirs:
    os.makedirs(os.path.join(output_dir, sub_dirs))
    os.makedirs(os.path.join(output_dir, sub_dirs, "Image"))
    os.makedirs(os.path.join(output_dir, sub_dirs, "Mask"))
os.makedirs(os.path.join(output_dir, "Test"))

def random_patches(sub_dir_name, n=5):
    imgs = random.sample(os.listdir(os.path.join(output_dir, sub_dir_name, "Image")), n)
    masks = [img.replace(".png", "") + "_mask.png" for img in imgs]

    images = [
        cv2.cvtColor(
            cv2.imread(os.path.join(output_dir, sub_dir_name, "Image", img)),
            cv2.COLOR_BGR2RGB,
        )
        for img in imgs
    ]
    masks = [
        cv2.cvtColor(
            cv2.imread(os.path.join(output_dir, sub_dir_name, "Mask", mask)),
            cv2.COLOR_BGR2RGB,
        )
        for mask in masks
    ]

    return images, masks

def create_n_save_patches(_type="Train"):

    image_path = os.path.join(root_directory, f"{_type}_Image")
    mask_path = os.path.join(root_directory, f"{_type}_Mask")

    file_names = os.listdir(image_path)

    for i, file_name in enumerate(file_names):
        file_id = file_name.replace(".png", "")
        print("Creating Patches for {}".format(file_id))

        image = cv2.imread(os.path.join(image_path, file_name))           

        mask = cv2.imread(os.path.join(mask_path, file_id + "_mask.png"))

        # e.g. Step = 256 for 256 patches means no overlap
        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :][0]

                single_patch_mask = patches_mask[i, j, :, :][0]

                if single_patch_img.shape == (patch_size, patch_size, 3):

                    cv2.imwrite(
                        os.path.join(output_dir, _type, "Image", f"{i}_{j}_{file_name}"),
                        single_patch_img,
                    )

                    cv2.imwrite(
                        os.path.join(output_dir, _type, "Mask", f"{i}_{j}_{file_id}" + "_mask.png"),
                        single_patch_mask,
                    )


# Creating Patches from Train and Valid Images and masks

create_n_save_patches("Train")
create_n_save_patches("Valid")


# Copy test image and mask as it is
shutil.copytree(os.path.join(root_directory, "Valid_Image"), os.path.join(output_dir, "Test", "Image"))
shutil.copytree(os.path.join(root_directory, "Valid_Mask"), os.path.join(output_dir, "Test", "Mask"))