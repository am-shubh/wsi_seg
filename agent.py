import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from patchify import unpatchify
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from constants import *
from utils import get_patches, visualize, calculate_average_dice_coefficient
from data import get_preprocessing


class TrainAgent:
    def __init__(
        self, device, model, optimizer, criterion, scheduler=None, logger=None
    ) -> None:
        self.device = device

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model = model.to(device)

        self.snapshots_root = MODEL_PATH

        self.jaccard = MulticlassJaccardIndex(num_classes=3, average="none").to(device)
        self.accuracy = MulticlassAccuracy(
            num_classes=3, average="none", multidim_average="global"
        ).to(device)

        self.logger = logger
        self.current_epoch = 1
        self.best_metric = 0

    def train(self, n_epochs, train_loader, valid_loader) -> None:
        for epoch in range(1, n_epochs + 1):
            ## Train
            self.stage = "train"
            self.model.train()

            self._init_records()
            self._run_epoch(train_loader)
            self._log_scalars()

            ## Validation
            self.stage = "valid"
            self.model.eval()

            self._init_records()
            with torch.no_grad():
                self._run_epoch(valid_loader)
            self._log_scalars()

            ## Creating snapshots
            self._save_snapshot("LATEST.PTH")

            cur_metric = (
                self.records["IoU_black"]
                * self.records["IoU_yellow"]
                * self.records["IoU_red"]
            ) / (
                self.records["IoU_black"]
                + self.records["IoU_yellow"]
                + self.records["IoU_red"]
                + 1e-10
            )
            cur_metric = cur_metric.item() / self.n_samples.item()
            if cur_metric > self.best_metric:
                self._save_snapshot("BEST.PTH")
                self.best_metric = cur_metric

            self.current_epoch += 1

    def _run_epoch(self, loader) -> None:
        for i, (features, target) in enumerate(loader):
            features = features.to(self.device)
            target = target.to(self.device)

            output = self.model(features)
            loss = self.criterion(output, target)

            if self.stage == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            target = target.detach()
            target = torch.squeeze(target, dim=1)
            output = output.detach()
            probabilities = nn.Softmax(dim=1)(output)
            prediction = torch.argmax(probabilities, dim=1)

            record = self._compute_metrics(prediction, target)
            record["loss"] = loss.item()

            batch_size = features.shape[0]
            self._update_records(record, batch_size)

        if not self.scheduler is None and self.stage == "train":
            self.scheduler.step()

    def _compute_metrics(self, prediction, target):
        record = {}
        record.update(
            {
                f"IoU_{key}": torch.tensor(0.0, device=self.device)
                for key in LABELS.keys()
            }
        )
        record.update(
            {
                f"acc_{key}": torch.tensor(0.0, device=self.device)
                for key in LABELS.keys()
            }
        )
        record.update({"IoU_mean": torch.tensor(0.0, device=self.device)})
        record.update({"acc_mean": torch.tensor(0.0, device=self.device)})

        # This part is necessary for correct calculation of IoU and accuracy metrics in case
        # of absence of pos. classes in target and prediction masks

        prediction[:, 0, 0], prediction[:, 0, 1] = 1, 2
        target[:, 0, 0], target[:, 0, 1] = 1, 2

        for i in range(prediction.shape[0]):
            pred_s = torch.unsqueeze(prediction[i], dim=0)
            target_s = torch.unsqueeze(target[i], dim=0)

            iou = self.jaccard(pred_s, target_s)
            acc = self.accuracy(pred_s, target_s)

            for key, value in LABELS.items():
                record[f"IoU_{key}"] += iou[value]
                record[f"acc_{key}"] += acc[value]

            record["IoU_mean"] += iou.mean()
            record["acc_mean"] += acc.mean()

        return record

    def _init_records(self) -> None:
        self.records = {}
        self.records.update(
            {
                f"IoU_{key}": torch.tensor(0.0, device=self.device)
                for key in LABELS.keys()
            }
        )
        self.records.update(
            {
                f"acc_{key}": torch.tensor(0.0, device=self.device)
                for key in LABELS.keys()
            }
        )

        self.records["IoU_mean"] = torch.tensor(0.0, device=self.device)
        self.records["acc_mean"] = torch.tensor(0.0, device=self.device)
        self.records["loss"] = torch.tensor(0.0, device=self.device)

        self.n_samples = torch.tensor(0)

    def _update_records(self, record, samples) -> None:
        for key in self.records.keys():
            self.records[key] += record[key]
        self.n_samples += samples

    def _log_scalars(self) -> None:
        for key in self.records.keys():
            self.logger.log(
                {
                    f"metrics/{self.stage}/{key}": torch.nan_to_num(
                        self.records[key] / self.n_samples, 0
                    ).item()
                },
                step=self.current_epoch,
            )

        if self.stage == "valid":
            self.logger.log(
                {"metrics/lr": self.optimizer.param_groups[0]["lr"]},
                step=self.current_epoch,
            )

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=self.device)

        self.model.load_state_dict(snapshot["PARAMS"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        if not self.scheduler is None and not snapshot.get("SCHEDULER") is None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER"])

        self.current_epoch = 1

        print(f"Checkpoint loaded from {snapshot_path}")

    def _save_snapshot(self, snapshot_name):
        snapshot = {
            "PARAMS": self.model.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict(),
            "CURRENT_EPOCH": self.current_epoch,
        }
        if not self.scheduler is None:
            snapshot["SCHEDULER"] = self.scheduler.state_dict()
        snapshot_path = os.path.join(self.snapshots_root, snapshot_name)

        torch.save(snapshot, snapshot_path)
        print(
            f"Epoch {self.current_epoch} | Training snapshot saved at {snapshot_path}"
        )


class InferenceAgent:
    def __init__(self, device, model, config, preprocessing_fn) -> None:
        self.device = device
        self.model = model

        if self.model is None:
            print("Model is not initialized")
        else:
            self.model = self.model.to(self.device)
            self.model.eval()

        self.mean = np.array([0.6075])
        self.std = np.array([0.3584])

        self.patch_size = config["patch_size"]

        self.preprocessing = get_preprocessing(preprocessing_fn)

    def load_weights(self, path):
        if self.model is None:
            print("It is not possible to load weights without model")
            return None

        weights = torch.load(path, map_location=self.device)
        self.model.load_state_dict(weights["PARAMS"])
        print(f"Weights are loaded from {path}")

    def predict(self, X):
        if self.model is None:
            print("Prediction is not possible without model")
            return None

        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            output = self.model(X)
            probabilities = nn.Softmax(dim=1)(output)
            prediction = torch.argmax(probabilities, dim=1)
            prediction = torch.squeeze(prediction, dim=0)

        return prediction.cpu().numpy()

    def predict_test_samples(self, path):
        image_path = os.path.join(path, "Image")
        mask_path = os.path.join(path, "Mask")

        file_names = os.listdir(image_path)

        for i, file_name in enumerate(file_names):
            file_id = file_name.replace(".png", "")
            print("Running Prediction for Test/Image/{}.png".format(file_id))

            print("Creating Patches for Test/Image/{}.png".format(file_id))

            img_file = os.path.join(image_path, file_name)
            mask_file = os.path.join(mask_path, file_id + "_mask.png")

            img_patches, img_shape, patches_img_shape = get_patches(
                img_file, self.patch_size
            )

            predictions = []
            for _patch in img_patches:
                sample = self.preprocessing(image=_patch)
                _patch = sample["image"]
                _patch = _patch / 255
                _patch = (_patch - self.mean) / self.std

                predictions.append(self.predict(np.expand_dims(_patch, axis=0)))

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

            pred_mask_three = np.zeros(
                shape=(pred_mask.shape[0], pred_mask.shape[1], 3)
            )

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

            # visualize(gt=gt_mask_three, pred=pred_mask_three)

            # Saving predictions
            pred_mask_three = Image.fromarray(pred_mask_three.astype(np.uint8))
            pred_mask_three.save(os.path.join(PREDICTION_DIR, file_name))

            # Calculating Dice loss
            calculate_average_dice_coefficient(gt_mask, pred_mask, 3)
