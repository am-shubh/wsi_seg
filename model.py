import segmentation_models_pytorch as smp
import torch
import logging
import monai


def get_model(config):
    logging.info("Getting DeeplabV3Plus Model and preprocessing function")

    net = {
        "name"      : "DeepLabV3Plus",
        "loss"      : "DiceLoss",
        "optimizer" : "Adam",
        "scheduler" : "OneCycleLR",
        "shape"     : (config["patch_size"], config["patch_size"]),

        "parameters" : {
            "encoder_name"          : config["encoder"],
            "encoder_depth"         : 5,
            "encoder_weights"       : config["encoder_weights"],
            "encoder_output_stride" : 16,
            "decoder_channels"      : 256,
            "in_channels"           : 3,
            "classes"               : 3,
            "activation"            : None,
        }
    }

    model = smp.DeepLabV3Plus(**net['parameters'])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config["encoder"], config["encoder_weights"]
    )

    # logging.info("Getting Model and preprocessing function")

    # model = smp.UnetPlusPlus(
    #     encoder_name=config["encoder"],
    #     encoder_weights=config["encoder_weights"],
    #     classes=config["classes"],
    #     activation=config["activation"]
    # )

    # model = smp.FPN(
    #     encoder_name=config["encoder"],
    #     encoder_weights=config["encoder_weights"],
    #     classes=config["classes"],
    #     activation=config["activation"]
    # )

    # model = smp.Unet(
    #     encoder_name=config["encoder"],
    #     encoder_weights=config["encoder_weights"],
    #     classes=config["classes"],
    #     activation=config["activation"],
    # )

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(
    #     config["encoder"], config["encoder_weights"]
    # )

    return model, preprocessing_fn


def get_loss_optimizer(device, model, config):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"])
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=2 * config["learning_rate"], total_steps=config["epochs"]
    # )
    scheduler = None

    # criterion = monai.losses.DiceLoss(
    #     include_background=True, reduction="mean", to_onehot_y=True
    # ).to(device)

    criterion = monai.losses.DiceFocalLoss(
        include_background=True, reduction="mean", to_onehot_y=True
    ).to(device)

    return optimizer, scheduler, criterion
