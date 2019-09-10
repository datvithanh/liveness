import numpy as np
import cv2
from trainer import Trainer
from tools.extract_data import detect_face_expanded
import torch

gpu = True
model_path = 'result/init/model_epoch40'
trainer = Trainer('data', model_path, gpu)
trainer.set_model()

cap = cv2.VideoCapture(0)

frame_list = []
cnt = 0

label = 0
while(True):
    # Capture frame-by-frame
    cnt += 1
    ret, frame = cap.read()

    # Our operations on the frame come here
    # cv2 video cap frame per sec
    # print(cap.get(cv2.CAP_PROP_FPS))
    # Display the resulting frame

    detected_frames, box = detect_face_expanded(frame_list)

    if label == 0:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    else:    
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    # 128 by 128 image
    if len(frame_list) < 8:
        frame_list.append(frame)

    if len(frame_list) == 8:
        if cnt % 3 == 0:
            frame_list.append(frame)
            frame_list = frame_list[1:]

            cut_frame = [cv2.resize(tmp[box[0]:box[2], box[1]:box[3], :], (int(128), int(128))) for tmp in frame_list]
            X = torch.Tensor(np.array([cut_frame]).transpose(0,4,1,2,3))
            label = trainer.predict(X)[0]

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
