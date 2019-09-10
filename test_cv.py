import numpy as np
import cv2
from trainer import Trainer

# gpu = True
# model_path = 'result/init/model_epoch40'
# trainer = Trainer('data', model_path, gpu)

cap = cv2.VideoCapture(0)

frame_list = []
cnt = 0
while(True):
    # Capture frame-by-frame
    cnt += 1
    ret, frame = cap.read()

    # Our operations on the frame come here
    # cv2 video cap frame per sec
    # print(cap.get(cv2.CAP_PROP_FPS))
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # 128 by 128 image
    if len(frame_list) < 8:
        frame_list.append(frame)

    if frame_list == 8:
        if cnt % 3 == 0:
            frame_list.append(frame)
            frame_list = frame_list[1:]

    print(len(frame_list))
    # rescaled_image = cv2.resize(frame, (int(128), int(128)))

    print(frame.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
