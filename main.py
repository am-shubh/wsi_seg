import logging
import traceback
import wandb
from utils import setup
from data import get_data_loader
from model import get_model, get_loss_optimizer
from agent import TrainAgent, InferenceAgent
from constants import *


if __name__ == "__main__":
    try:
        device, config = setup()

        model, preprocessing_fn = get_model(config)

        train_loader, valid_loader = get_data_loader(config, preprocessing_fn)

        optimizer, scheduler, criterion = get_loss_optimizer(device, model, config)

        logging.info("###")
        logging.info("WANDB INITIALIZATION")
        logging.info("###")

        logger = wandb.init(project="exp", config=config)

        # Train and Valid
        agent = TrainAgent(device, model, optimizer, criterion, scheduler, logger)
        agent.train(config["epochs"], train_loader, valid_loader)

        # Inference
        infer_agent = InferenceAgent(device, model, config, preprocessing_fn)
        infer_agent.load_weights(BEST_MODEL_PATH)
        infer_agent.predict_test_samples(TEST_DIRECTORY)

        logger.save(LOG_FILE)

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

        logger.save(LOG_FILE)
