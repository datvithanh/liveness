import cv2
import numpy as np
import os

def load_image(path):
    tmp = cv2.imread(path)
    return cv2.resize(tmp, (int(128), int(128)))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]
