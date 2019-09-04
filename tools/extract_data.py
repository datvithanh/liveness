import os
import glob
import cv2
import random
from tqdm import tqdm
import imutils
import numpy as np
from face_detection.face_detection import detect_face_and_rotate, to_rgb
import pdb

data_dir = '/home/common_gpu0/corpora/vision/liveness/rose/videos/'
out_dir = '/home/common_gpu0/corpora/vision/liveness/rose/images/'
miss_dir = '/home/common_gpu0/corpora/vision/liveness/rose/miss/'
debug = False
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

#def extract_face(frame):
#    #ret = detect_face_and_rotate(frame)
#    ret = detect_face_and_rotate(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
#    box = ret[0][0]
#    if isinstance(box, list) or (len(box.shape)!=2 and box.shape[1]!= 4):
#        return None
#    else:
#        angle = ret[1]
#        if angle != 0:
#                frame = imutils.rotate_bound(frame.copy(),angle)
#        left, top, right, bottom = box[0]
#        return frame[top:bottom, left:right]

def detect_face_expanded(frames):
    lefts, rights, tops, bottoms = [], [], [], []
    detected_frames = []
    for frame in frames:
        ret = detect_face_and_rotate(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        box = ret[0][0]
        if isinstance(box, list) or (len(box.shape)!=2 and box.shape[1]!= 4):
            pass
        else:
            angle = ret[1]
            if angle != 0:
                frame = imutils.rotate_bound(frame.copy(),angle)
            left, top, right, bottom = box[0]
            lefts.append(left)
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)
            detected_frames.append(frame)
    
    if len(lefts)==0:
        return [], ()

    mleft = min(lefts)
    mtop = min(tops)
    mright = max(rights)
    mbottom = max(bottoms)
    
    return detected_frames, (mleft, mtop, mright, mbottom)

def extract_face(detected_frames, box):
    left, top, right, bottom = box
    ret_frames = []
    for frame in detected_frames:
        ret_frames.append(frame[top:bottom, left:right])
    return ret_frames

def augument_move(box, move_type):
    move_rate = 0.2
    left, top, right, bottom = box
    width = right-left
    height = bottom-top
    px_xmove = move_rate*width
    px_ymove = move_rate*height
    if move_type == 'right':
        left += px_xmove
        right += px_xmove
    elif move_type == 'left':
        left -= px_xmove
        right -= px_xmove
    elif move_type == 'up':
        top -= px_ymove 
        bottom -= px_ymove 
    elif move_type == 'down':
        top += px_ymove 
        bottom += px_ymove 
    else:
        pdb.set_trace()
    return (int(left), int(top), int(right), int(bottom))

def write_processed_videos(processed_videos):
    with open(os.path.join(data_dir, 'processed_videos.txt'), 'wt') as f:
        f.write('\n'.join(processed_videos)) 

def read_processed_videos():
    with open(os.path.join(data_dir, 'processed_videos.txt'), 'rt') as f:
        videos = f.readlines() 
        videos = [v.strip() for v in videos]
        return videos

def extract_dir():
    #processed_videos = []
    processed_videos = read_processed_videos()
    for sub in glob.glob(os.path.join(data_dir, '*')):
        sub_name = os.path.basename(sub)
        if sub_name in ['15', '17']:
            continue
        print('-'*20, sub_name)
        sub_out_dir = os.path.join(out_dir, sub_name)
        make_dir(sub_out_dir)
        video_count = 0
        for fv in tqdm(glob.glob(os.path.join(sub, '*'))):
            if fv in processed_videos:
                print('Processed: ', fv)
                continue
            fps, frames = read_all_frames(fv)
            num_examples = int(len(frames)/(fps*example_duration))
            #print('{}s\t{}\t{}'.format(round(len(frames)/fps,3), num_examples, fv))
            example_count = 0
            for eid in range(num_examples):
                fv_name, _ = os.path.splitext(os.path.basename(fv))
                fv_dir = os.path.join(sub_out_dir, '{}__{}'.format(fv_name, eid))
                make_dir(fv_dir)
                exam_frames = frames[int(eid*example_duration*fps):int((eid+1)*example_duration*fps)]
                exam_frames = random_select(exam_frames, fps, example_frame_per_second)
                #print(eid*example_duration*fps, (eid+1)*example_duration*fps)

                ### Without face bounding box expandation
                #for idx, frame in enumerate(exam_frames):
                #    frame_path = os.path.join(fv_dir, '{}__{}_{}.jpg'.format(fv_name, eid,idx))
                #    face_frame = extract_face(frame)
                #    #ret1, ret2  = extract_face(frame)
                #    if face_frame is None: #len(ret2[0][0])==0:
                #        #print('{}\t{}\t{}'.format(frame_path, ret1, ret2))
                #        print('Detect Face error {}'.format(frame_path))
                #        miss_path = os.path.join(miss_dir,'{}__{}_{}.jpg'.format(fv_name, eid,idx))
                #        write_frame(frame, miss_path)
                #    else:
                #        write_frame(face_frame, frame_path)
                #    #pdb.set_trace()

                ###with face bounding box expandation
                detected_frames, box = detect_face_expanded(exam_frames)
                if len(detected_frames)==0:
                    for idx, frame in enumerate(exam_frames):
                        frame_path = os.path.join(miss_dir, '{}__{}_{}.jpg'.format(fv_name, eid, idx))
                        write_frame(frame, frame_path)
                else:
                    face_path = fv_dir
                    face_frames = extract_face(detected_frames, box)
                    for idx, frame in enumerate(face_frames):
                        frame_path = os.path.join(face_path, '{}__{}_{}.jpg'.format(fv_name, eid, idx))
                        write_frame(frame, frame_path)

                    #Augumentation with move
                    for move_type in ['right', 'left', 'up', 'down']:
                        box = augument_move(box, move_type)
                        face_frames = extract_face(detected_frames, box)
                        move_face_path = '{}_{}'.format(face_path, move_type)
                        make_dir(move_face_path)
                        for idx, frame in enumerate(face_frames):
                            frame_path = os.path.join(move_face_path, '{}__{}_{}.jpg'.format(fv_name, eid, idx))
                            write_frame(frame, frame_path)

                example_count += 1
                if debug and example_count>=2:
                    break
            processed_videos.append(fv)
            video_count += 1
            if debug and video_count>=2:
                break
            if video_count%3==0:
                write_processed_videos(processed_videos) 
    write_processed_videos(processed_videos)

def correct_gamma(y2, y1):
    for sub in glob.glob(os.path.join(out_dir, '*')):
        for example in glob.glob(os.path.join(sub, '*')):
            for img_path in glob.glob(os.path.join(example, '*')):
                img = cv2.imread(img_path)
                img = np.divide(img, 255.0)
                pow_rate = y2/y1
                img = np.power(img, pow_rate)
                img = img*255.0
                img = np.trunc(img)
                img[img>255] = 255
                img = img.astype('uint8')
                #continue here: create the example directory and push these correct gramma into

if __name__ == '__main__':
    extract_dir()
    #correct_gamma(1.0, 2.2)
    #correct_gamma(2.2, 1.0)

#TODO:
# Correct gamut

#Notice:
#some example might not have enough 9 faces - no detects found
#changes: spartial augumentation we also keep move which move outside of image but only keep pixels inside image
