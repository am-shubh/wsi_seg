import os

EXP_DIR = "size512_deeplabV3Plus"

CONFIG_FILE = "config.json"
LOG_FILE = os.path.join(EXP_DIR, f"{EXP_DIR}.log")
DATASET_LOCATION = "Dataset"
MODEL_PATH = os.path.join(EXP_DIR, "Weights")
BEST_MODEL_PATH = os.path.join(MODEL_PATH, "BEST.PTH")
TRAIN_DIRECTORY = os.path.join(DATASET_LOCATION, "Train")
VAL_DIRECTORY = os.path.join(DATASET_LOCATION, "Valid")
TEST_DIRECTORY = os.path.join(DATASET_LOCATION, "Test")
PREDICTION_DIR = os.path.join(EXP_DIR, "Predictions")

CLASSES = 3
CLASS_LIST = ["Black", "Yellow", "Red"]
LABELS = {"black": 0, "yellow": 1, "red": 2}