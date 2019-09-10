import cv2

def load_image(path):
    tmp = cv2.imread(path)
    return cv2.resize(tmp, (int(128), int(128)))
