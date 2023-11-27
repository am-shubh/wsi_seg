import json
import logging
import os, sys
import cv2
import numpy as np


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


def read_img(img, crop, mask, rsz=None):
    try:
        imag = cv2.imread(img)
        orig_ht, orig_wd = imag.shape[:2]

    except OSError as e:
        print(
            "OSError: image file {} is not found -{}",
            sys.exc_info()[0],
            sys.exc_info()[1],
        )

    if len(imag.shape) == 2:
        imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)
    else:
        channels = imag.shape[2]
        if channels == 4:
            imag = cv2.cvtColor(imag, cv2.COLOR_RGBA2RGB)
        else:
            imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

    if crop != None:
        imag = imag[crop[0] : crop[1], crop[2] : crop[3]]

    if type(mask) == np.ndarray:
        imag = cv2.bitwise_and(imag, imag, mask=mask)

    if rsz is not None:
        imag = cv2.resize(imag, (rsz[1], rsz[0]))

    if len(imag.shape) == 2:
        imag = np.expand_dims(imag, axis=2)

    imag = imag.astype("uint8")

    return imag, orig_ht, orig_wd