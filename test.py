from trainer import Trainer
from utils import load_image
import torch
import numpy as np
import time
import pandas as pd
import cv2 
import os
from tqdm import tqdm
import time 

gpu = False

model_path = 'result/model_epoch40'
trainer = Trainer('data', model_path, gpu)
trainer.set_model()


path = '../live_examples/Webcam'

f = open('out.txt', 'w+')

t = time.time()

for example in tqdm(os.listdir(path)):
    imgs = sorted(os.listdir(os.path.join(path, example)))
    X = [load_image(os.path.join(path, example, tmp)) for tmp in imgs[:8]]
    X = np.array([X]).transpose(0,4,1,2,3)
    X = torch.Tensor(X)
    label = trainer.predict(X)[0]
    f.write(f'{label} {example}\n')

print(f'{time.time() - t} seconds')
