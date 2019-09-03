import os
import glob
import cv2
import random
import imutils
from face_detection.face_detection import detect_face_and_rotate, to_rgb
import pdb

data_dir = '/home/common_gpu0/corpora/vision/liveness/rose/videos/'
out_dir = '/home/common_gpu0/corpora/vision/liveness/rose/images/'
miss_dir = '/home/common_gpu0/corpora/vision/liveness/rose/miss/'
debug = True
example_duration = 3.0 #second
example_frame_per_second = 3

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def read_all_frames(video_file):
    video = cv2.VideoCapture(video_file)
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS) 
    ret = True
    while(ret):
        ret, frame = video.read()
        frames.append(frame)
    video.release()
    return fps, frames

def random_select(frames, fps, second_frame):
    selected_frames = []
    sec_num = int(len(frames)/fps)
    for sec in range(sec_num):
        sec_frames = frames[int(sec*fps):int((sec+1)*fps)]
        frame_ids = sorted(random.sample(range(int(fps)), second_frame))
        for idx in frame_ids:
            selected_frames.append(sec_frames[idx])
        #print(int(sec*fps), '\t', int((sec+1)*fps))
    return selected_frames

def write_frame(frame, path):
    cv2.imwrite(path, frame)

def extract_face(frame):
    #ret = detect_face_and_rotate(frame)
    ret = detect_face_and_rotate(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    box = ret[0][0]
    if isinstance(box, list) or (len(box.shape)!=2 and box.shape[1]!= 4):
        return None
    else:
        angle = ret[1]
        if angle != 0:
                frame = imutils.rotate_bound(frame.copy(),angle)
        left, top, right, bottom = box[0]
        return frame[top:bottom, left:right]

def read_dir():
    for sub in glob.glob(os.path.join(data_dir, '*')):
        sub_name = os.path.basename(sub)
        sub_out_dir = os.path.join(out_dir, sub_name)
        make_dir(sub_out_dir)
        video_count = 0
        for fv in glob.glob(os.path.join(sub, '*')):
            fps, frames = read_all_frames(fv)
            num_examples = int(len(frames)/(fps*example_duration))
            print('{}s\t{}\t{}'.format(round(len(frames)/fps,3), num_examples, fv))
            example_count = 0
            for eid in range(num_examples):
                fv_name, _ = os.path.splitext(os.path.basename(fv))
                fv_dir = os.path.join(sub_out_dir, '{}__{}'.format(fv_name, eid))
                make_dir(fv_dir)
                exam_frames = frames[int(eid*example_duration*fps):int((eid+1)*example_duration*fps)]
                exam_frames = random_select(exam_frames, fps, example_frame_per_second)
                #print(eid*example_duration*fps, (eid+1)*example_duration*fps)
                for idx, frame in enumerate(exam_frames):
                    frame_path = os.path.join(fv_dir, '{}__{}_{}.jpg'.format(fv_name, eid,idx))
                    face_frame = extract_face(frame)
                    #ret1, ret2  = extract_face(frame)
                    if face_frame is None: #len(ret2[0][0])==0:
                        #print('{}\t{}\t{}'.format(frame_path, ret1, ret2))
                        print('Detect Face error {}'.format(frame_path))
                        miss_path = os.path.join(miss_dir,'{}__{}_{}.jpg'.format(fv_name, eid,idx))
                        write_frame(frame, miss_path)
                    else:
                        write_frame(face_frame, frame_path)
                    #pdb.set_trace()
                example_count += 1
                if debug and example_count>=2:
                    break
            video_count += 1
            if debug and video_count>=2:
                break

if __name__ == '__main__':
    read_dir()
#TODO:
# expand face bounding box
# Move the box
# Correct gamut
