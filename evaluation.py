import os
import numpy as np
from PIL import Image
from patchify import patchify, unpatchify
import cv2

CLASS_LIST = ["Black", "Yellow", "Red"]


def get_patches(img_file, patch_size=256):
    # step_size = int(patch_size * 0.8)

    image = cv2.imread(img_file)
    img_shape = image.shape[:2]

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


def predict(model, X):
    pred = model.predict(X)
    pred = np.argmax(pred, axis=-1)
    return pred


def predict_test_samples(model, path, prediction_dir, preprocess_func, patch_size=512):
    image_path = os.path.join(path, "Image")
    mask_path = os.path.join(path, "Mask")

    file_names = os.listdir(image_path)

    for i, file_name in enumerate(file_names):
        file_id = file_name.replace(".png", "")
        print("Running Prediction for Test/Image/{}.png".format(file_id))

        print("Creating Patches for Test/Image/{}.png".format(file_id))

        img_file = os.path.join(image_path, file_name)
        mask_file = os.path.join(mask_path, file_id + "_mask.png")

        img_patches, img_shape, patches_img_shape = get_patches(img_file, patch_size)

        predictions = []
        for _patch in img_patches:
            if preprocess_func:
                _patch = preprocess_func(_patch)
            else:
                _patch = _patch / 255.0
                
            predictions.append(predict(model, np.expand_dims(_patch, axis=0)))

        predictions = np.array(predictions)
        predictions = np.reshape(
            predictions,
            [
                patches_img_shape[0],
                patches_img_shape[1],
                patches_img_shape[2],
                patches_img_shape[3],
            ],
        )
        pred_mask = unpatchify(predictions, img_shape)

        pred_mask_three = np.zeros(shape=(pred_mask.shape[0], pred_mask.shape[1], 3))

        pred_mask_three[pred_mask == 0] = [0, 0, 0]
        pred_mask_three[pred_mask == 1] = [255, 255, 0]
        pred_mask_three[pred_mask == 2] = [255, 0, 0]

        gt_mask_three = cv2.imread(mask_file)
        gt_mask_three = cv2.cvtColor(gt_mask_three, cv2.COLOR_BGR2RGB)

        label_seg = np.zeros(gt_mask_three.shape, dtype=np.uint8)
        label_seg[np.all(gt_mask_three == [0, 0, 0], axis=-1)] = 0
        label_seg[np.all(gt_mask_three == [255, 255, 0], axis=-1)] = 1
        label_seg[np.all(gt_mask_three == [255, 0, 0], axis=-1)] = 2

        # Just take the first channel, no need for all 3 channels
        gt_mask = label_seg[:, :, 0]

        # Saving predictions
        pred_mask_three = Image.fromarray(pred_mask_three.astype(np.uint8))
        pred_mask_three.save(os.path.join(prediction_dir, file_name))

        # Calculating Dice loss
        calculate_average_dice_coefficient(gt_mask, pred_mask, 3)
