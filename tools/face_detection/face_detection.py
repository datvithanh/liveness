from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tools.face_detection.align.detect_face as mtcnn
import numpy as np
import os
import imutils
import cv2


gpu_memory_fraction = 0.2
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = mtcnn.create_mtcnn(sess, \
                os.path.join(os.path.abspath(os.path.dirname(__file__)), 'align'))

threshold = [0.6, 0.75, 0.85]  # three steps's threshold
factor = 0.65  # scale factor


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def detect_face(img, min_size=40, max_dim=600, padding=80, ret_conf=False):
    if img.ndim < 2:
        print('Unable to detect face: img.ndim < 2')
        if ret_conf:
            return [], [], []
        else:
            return [], []
    if img.ndim == 2:
        img = to_rgb(img)
    img = img[:, :, 0:3].copy()
    h, w = img.shape[:2]
    if w > h and w > max_dim:
        img = imutils.resize(img, width=max_dim)
    if h > w and h > max_dim:
        img = imutils.resize(img, height=max_dim)
    nh, nw = img.shape[:2]
    img=cv2.copyMakeBorder(img, padding, padding, padding, padding,cv2.BORDER_CONSTANT,value=(100, 100, 100))
    bounding_boxes, points = mtcnn.detect_face(img, min_size, pnet, rnet, onet, threshold, factor)
    bbs, confs = [], []
    for idx, face_position in enumerate(bounding_boxes):
        # face_position = face_position.astype(int)
        l, t, r, b = face_position[:4]
        if b - t < 2 or r - l < 2 or b < 0 or l < 0 or t < 0 or r < 0: continue
        bbs.append((l, t, r, b))
        confs.append(face_position[4])
    if len(bbs) > 0:
        points = np.array([points[:,i].reshape((2,5)).T for i in range(points.shape[1])])
        points = ((float(w) / nw) * (points - padding).clip(min=0)).astype(np.int32)
        bbs = np.array(bbs)
        bbs = ((float(w) / nw) * (bbs - padding).clip(min=0)).astype(np.int32)
    if ret_conf:
        return bbs, points, np.array(confs)
    else:
        return bbs, points


def detect_face_all_directions(img, min_size=40, max_dim=600, padding=80):
    rotation_angle = None
    for angle in [0, 180, 270, 90]:
        # print("Try to rotate: ", angle)
        image = imutils.rotate_bound(img.copy(), angle)
        ret, points = detect_face(image, min_size, max_dim, padding)
        if len(ret) > 0:
            rotation_angle = angle
            break
    return ret, points, rotation_angle


def detect_face_and_rotate(img, min_size=40, max_dim=600, padding=80, angles=None):
    tries, rets = [], []
    if angles is None:
        angles = [0, 180, 270, 90]
    for idx, angle in enumerate(angles):
        image = imutils.rotate_bound(img.copy(), angle)
        ret, points, confs = detect_face(image, min_size, max_dim, padding, ret_conf=True)
        rets.append((ret, points, confs))
        tries.append((len(ret), max(confs) if len(confs) > 0 else 0, idx))
    tries.sort()
    best_try = tries[-1]
    best_angle = angles[best_try[2]]
    return rets[best_try[2]], best_angle
