import os, shutil, sys
import cv2
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from patchify import patchify

import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image


class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith("\n"):
            self.buf.append(msg.rstrip("\n"))
            self.logfct("".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass


def setup():
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)
    os.makedirs(EXP_DIR)
    os.makedirs(MODEL_PATH)
    os.makedirs(PREDICTION_DIR)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        filename=LOG_FILE,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    logging.info("Experiemnt Directory Created...")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("Device available : {}".format(DEVICE))

    with open(CONFIG_FILE, "r") as fp:
        config = json.load(fp)

    return DEVICE, config


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(10, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def get_patches(img_file, patch_size=256):
    # step_size = int(patch_size * 0.8)

    image = cv2.imread(img_file)
    img_shape = image.shape[:2]

    # SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
    # SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
    # image = cv2.resize(image, (SIZE_X, SIZE_Y))
    # rsz_shape = image.shape[:2]

    # image = Image.fromarray(image)
    # image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
    # image = np.array(image)

    # e.g. Step = 256 for 256 patches means no overlap
    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
    patches_img = patches_img[:, :, 0, :, :, :]

    img_patches = []

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            img_patches.append(single_patch_img)

    return img_patches, img_shape, patches_img.shape


def calculate_dice_coefficient(y_true, y_pred, class_id):
    true_mask = np.where(y_true == class_id, 1, 0)
    pred_mask = np.where(y_pred == class_id, 1, 0)

    tp = np.sum(np.logical_and(true_mask, pred_mask))
    fp = np.sum(np.logical_and(1 - true_mask, pred_mask))
    fn = np.sum(np.logical_and(true_mask, 1 - pred_mask))

    dice_coefficient = 2 * tp / (2 * tp + fp + fn)

    return dice_coefficient


def calculate_average_dice_coefficient(y_true, y_pred, num_classes):
    dice_coefficients = []
    for class_id in range(num_classes):
        dice_coefficient = calculate_dice_coefficient(y_true, y_pred, class_id)
        print(f"Dice coefficient for class {CLASS_LIST[class_id]}: {dice_coefficient}")
        dice_coefficients.append(dice_coefficient)

    average_dice_coefficient = np.mean(dice_coefficients)
    print(f"Average Dice coefficient: {average_dice_coefficient}")
