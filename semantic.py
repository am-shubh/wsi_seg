import logging
import traceback
import sys, os, glob, shutil
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

from loss import DiceLoss, freezeGraph
from callbacks import customCallbacks
from utils import LoggerWriter
from data_processing import preprocess_masks, DataGenerator
from evaluation import predict_test_samples
from simple_multi_unet_model import multi_unet_model

from tensorflow import keras

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

keras.backend.set_image_data_format("channels_last")

class SemanticModel:
    """
    Class containing methods for semantic segmentation models training and evaluation.
    """

    def __init__(self) -> None:
        self.k_sess_value = None

        self.path = "/home/rnd/Documents/backup/DL/exp/unet_512"

        # Get the data paths
        self.dataset_path = os.path.join(self.path, "Dataset")
        self.model_path = os.path.join(self.path, "Weights")
        self.train_model_path = os.path.join(self.path, "Weights", "Trained")
        self.best_weights_path = os.path.join(self.train_model_path, "best_weight.hdf5")

        self.logfile_path = os.path.join(self.path, "Logs")

        self.config_path = os.path.join(self.path, "Config")

        self.tfrecord_path = os.path.join(self.path, "tf_records")
        self.tfrecord_file = os.path.join(self.path, "tf_records", "TFInfo.json")

        self.prediction_save_path = os.path.join(self.path, "Predictions")

        # creating prediction dir
        if os.path.exists(self.prediction_save_path):
            shutil.rmtree(self.prediction_save_path)
        os.makedirs(self.prediction_save_path)

        with open(os.path.join(self.config_path, "config.json"), "r") as fp:
            self.config = json.load(fp)

        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(self.logfile_path, "wsi_seg.log"),
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )

        sys.stdout = LoggerWriter(logging.info)
        sys.stderr = LoggerWriter(logging.error)

        self.preprocess_input = sm.get_preprocessing("resnet34")

    def read_n_set_params(self) -> None:
        self.aug_params = self.config["aug_params"]
        self.batch_size = self.config["batch_size"]
        self.epochs = self.config["epochs"]
        self.model = self.config["model"]
        self.learningRate = self.config["learning_rate"]
        self.input_shape = self.config["input_shape"]
        self.classes = self.config["class_label"]
        self.output_shape = len(self.classes)

    def get_data_generator(self) -> None:
        logging.info("Pre-Processing individual masks")
        preprocess_masks(self.dataset_path)

        json_data = {}

        logging.info("Creating data generator")

        train_data_gen = DataGenerator(
            self.dataset_path,
            self.classes,
            (self.input_shape[0], self.input_shape[1]),
            _type="Train",
            aug_params=self.aug_params,
            # preprocess_input=self.preprocess_input,
        )

        (
            json_data["len_Train_data"],
            training_data,
        ) = train_data_gen.get_generator(self.batch_size)

        valid_data_gen = DataGenerator(
            self.dataset_path,
            self.classes,
            (self.input_shape[0], self.input_shape[1]),
            _type="Valid",
            # preprocess_input=self.preprocess_input,
        )

        (
            json_data["len_Valid_data"],
            validation_data,
        ) = valid_data_gen.get_generator(self.batch_size)

        logging.info("Data generator initialised ...")
        print("Train Dataset:", training_data)
        print("Val Dataset:", validation_data)
        print(json_data)

        len_train_data = json_data["len_Train_data"]
        len_validation_data = json_data["len_Valid_data"]
        steps_per_epoch = np.ceil(len_train_data // self.batch_size)
        validation_steps = np.ceil(len_validation_data // self.batch_size)

        return training_data, validation_data, steps_per_epoch, validation_steps

    def get_model(self, compile=True):
        logging.info("Getting model architecture")

        # if self.output_shape == 1:
        #     activation = "sigmoid"
        # else:
        #     activation = "softmax"

        # model = sm.Unet(
        #     "resnet34",
        #     input_shape=self.input_shape,
        #     classes=self.output_shape,
        #     activation=activation,
        # )

        model = multi_unet_model(len(self.classes), self.input_shape[0], self.input_shape[1], self.input_shape[2])

        if compile:
            # loss = (
            #     "binary_crossentropy"
            #     if self.output_shape == 1
            #     else SparseCategoricalCrossentropy(from_logits=True)
            # )

            # dice_loss = DiceLoss(self.output_shape)

            # optimizer = Adam(lr=self.learningRate)

            # metrics = ["accuracy", dice_loss.diceCoefLoss]

            # logging.info("Compiling model")
            # model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            optimizer = Adam(lr=self.learningRate)
            dice_loss = sm.losses.DiceLoss()
            focal_loss = sm.losses.CategoricalFocalLoss()
            total_loss = dice_loss + (1 * focal_loss)

            metrics = [
                sm.metrics.IOUScore(threshold=0.5),
                sm.metrics.FScore(threshold=0.5),
            ]

            model.compile(optimizer, total_loss, metrics=metrics)

        return model

    def get_callbacks(self) -> list:
        logging.info("Setting up Callbacks")

        model_checkpoints, early_stoping, reduce_LR, tensor_board = customCallbacks(
            self.logfile_path, self.train_model_path
        )
        callbacks = [model_checkpoints, early_stoping, reduce_LR, tensor_board]

        return callbacks

    def train(self, model, train_gen, valid_gen, steps_per_epoch, validation_steps):
        callbacks = self.get_callbacks()

        logging.info("Training the model")
        train_history = model.fit(
            train_gen,
            validation_data=valid_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        logging.info("Training completed")

        model.save_weights(os.path.join(self.train_model_path, "model.hdf5"))

        return train_history

    def get_best_weights(self):
        logging.info("Loading model with best saved weights")

        model = self.get_model(compile=False)
        model.load_weights(self.best_weights_path)
        return model

    def evaluate(self):
        model = self.get_best_weights()

        logging.info("Evaluating the model")
        predict_test_samples(
            model,
            os.path.join(self.dataset_path, "Test"),
            self.prediction_save_path,
            None,
            self.input_shape[0],
        )

    def train_evaluate(self):
        try:
            self.read_n_set_params()

            (
                train_gen,
                valid_gen,
                steps_per_epoch,
                validation_steps,
            ) = self.get_data_generator()

            model = self.get_model()

            train_history = self.train(
                model, train_gen, valid_gen, steps_per_epoch, validation_steps
            )

            self.evaluate()

        except Exception as e:
            print(traceback.format_exc())
            logging.error(e)
            logging.error(traceback.format_exc())
