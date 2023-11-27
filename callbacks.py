# Dependencies
import io, json, os, shutil
import tensorflow as tf
from tensorflow.python.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)


def customCallbacks(log_path, path):
    """
    log_dir : string , must exist
    histogram_freq : freq in epochs
    write_graph : true or false for seeing the graph in TB
    write_grads :writing the gradients not working 2.04
    batch_size : size of batches for histogram
    write_images :whether to write model weights to see
    """

    tensor_board = TensorBoard(
        log_dir=log_path, histogram_freq=1, write_graph=True, write_images=False
    )

    """
    filepath - str can use formating to put epoch number etc {epoc}
    monitor - qunatity to monitor
    verbose - 0 or 1
    save_best_only - only save if better than before
    mode - use 'auto'
    save_weights_only - if false then will save whole model
    period - the interval between epochs
    """

    model_checkpoints = ModelCheckpoint(
        os.path.join(path, "best_weight.hdf5"),
        monitor="val_loss",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode="min",
    )

    """
        monitor : what to monitor 'val_loss', 'acc'
        min_delta :minimum change to qualify as improvement
        patience : number of epocs with no improvement before you stop
        verbose : verbosity mode
        mode : 'auto' , can be 'min' or 'max' determines direction of improvement. Auto = based on monitor
    """

    early_stoping = EarlyStopping(
        monitor="val_loss", min_delta=0.000001, patience=30, verbose=0, mode="min"
    )

    """
    monitor : quality to be monitored eg. 'val_loss' , 'val_acc'
    factor : the factor by which the current LR be multiplied
    patience : number of epochs with no improvement
    verbose : 1 = update messages 0 nothing
    mode : 'auto' eg. is improvment up or down 'min' 'max'
    epsilon : threshold for measuring the new optimum, to only focus on significant changes
    cooldown :number of epochs to wait before any new changes
    min_lr: the lowest lr allowed
    """
    reduce_LR = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.9,
        patience=10,
        cooldown=2,
        min_lr=0.000001,
        mode="min",
    )

    return model_checkpoints, early_stoping, reduce_LR, tensor_board
