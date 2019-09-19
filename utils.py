import cv2
import numpy as np
import os

def load_image(path, color_channel='rgb'):
    tmp = cv2.imread(path)
    tmp = cv2.resize(tmp, (int(128), int(128)))
    if color_channel == 'ycbcr':
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2YCR_CB)
    if color_channel == 'hsv':
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    return tmp

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]
