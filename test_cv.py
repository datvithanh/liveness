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
box = ()
while(True):
    # Capture frame-by-frame
    cnt += 1
    ret, frame = cap.read()

    frame = cv2.resize(frame, (int(640), int(480)))
    # Our operations on the frame come here
    # cv2 video cap frame per sec
    print(cap.get(cv2.CAP_PROP_FPS))
    # Display the resulting frame
    

    print(label, box)

    # 128 by 128 image
    if len(frame_list) < 8:
        frame_list.append(frame)

    if len(frame_list) == 8:
        if cnt % 5  == 0:
            frame_list.append(frame)
            frame_list = frame_list[1:]

            detected_frames, box = detect_face_expanded(frame_list)
            if len(detected_frames) != 8:
                continue
            cut_frame = [cv2.resize(tmp[box[1]:box[3], box[0]:box[2], :], (int(128), int(128))) for tmp in detected_frames]
            X = torch.Tensor(np.array([cut_frame]).transpose(0,4,1,2,3))
            cv2.imwrite(f'tmp/{cnt}.jpg', cut_frame[-1])
            label = trainer.predict(X)[0]

    if len(box) != 0:
        if label == 0:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        else:    
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
