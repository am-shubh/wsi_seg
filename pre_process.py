import os, shutil
import cv2
import random
import numpy as np
import json
from constants import *
from utils import visualize
from matplotlib import pyplot as plt
from patchify import patchify
from sklearn.model_selection import train_test_split

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image


root_directory = 'Data/'

output_dir = DATASET_LOCATION
output_sub_dirs = ["Train", "Valid"]

with open(CONFIG_FILE, "r") as fp:
    config = json.load(fp)

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

        # SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
        # SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
        # image = Image.fromarray(image)
        # image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
        # image = np.array(image)             

        mask = cv2.imread(os.path.join(mask_path, file_id + "_mask.png"))

        # mask = Image.fromarray(mask)
        # mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
        # mask = np.array(mask)

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

# images = os.listdir(os.path.join(output_dir, "Train", "Image"))
# masks = [img.replace(".png", "") + "_mask.png" for img in images]

# X_train, X_valid, y_train, y_valid = train_test_split(
#     images, masks, test_size=0.20, random_state=42, shuffle=True
# )

# print("Train - Image & Mask :: ", len(X_train), len(y_train))
# print("Valid - Image & Mask :: ", len(X_valid), len(y_valid))

# Moving splitted files from train to valid in WSI_patches folder
# for img, mask in zip(X_valid, y_valid):
#     shutil.move(
#         os.path.join(output_dir, "Train", "Image", img),
#         os.path.join(output_dir, "Valid", "Image", img),
#     )
#     shutil.move(
#         os.path.join(output_dir, "Train", "Mask", mask),
#         os.path.join(output_dir, "Valid", "Mask", mask),
#     )

# Copy test image and mask as it is
shutil.copytree(os.path.join(root_directory, "Valid_Image"), os.path.join(output_dir, "Test", "Image"))
shutil.copytree(os.path.join(root_directory, "Valid_Mask"), os.path.join(output_dir, "Test", "Mask"))

# Visualization
# Train
# images, masks = random_patches("Train", 5)
# for img, mask in zip(images, masks):
#     visualize(image=img, mask=mask)

# # Valid
# images, masks = random_patches("Valid", 5)
# for img, mask in zip(images, masks):
#     visualize(image=img, mask=mask)