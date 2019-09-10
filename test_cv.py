import numpy as np
import cv2
from utils import 

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #cv2 video cap frame per sec
    print(cap.get(cv2.CAP_PROP_FPS))
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # 128 by 128 image

    rescaled_image = cv2.resize(frame, (int(128), int(128)))

    

    print(frame.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()