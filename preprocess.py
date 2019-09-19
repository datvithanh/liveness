import os
import argparse
import pandas as pd
from tqdm import tqdm
from random import shuffle
parser = argparse.ArgumentParser(description='Preprocess program for the dataset.')
parser.add_argument('--data_path', type=str, help='Path to the dataset')
parser.add_argument('--test', action='store_true', help='prepare test set')
parser.add_argument('--outdir', type=str, help='path to store ds metadata')

params = parser.parse_args()

root = params.data_path

images, labels = [], []
valid_images, valid_labels = [], []

train_subjects = ['2','3','4','5','6','7','9','10','11','12']

valid_subjects = [tmp for tmp in os.listdir(root) if tmp not in train_subjects]
train_subjects = [tmp for tmp in os.listdir(root) if tmp in train_subjects]

subjects = {'train': train_subjects, 'valid': valid_subjects}

for k, v in subjects.items():
    for fn in v:
        print(fn)
        for data_point in tqdm(os.listdir(os.path.join(root, fn))):
            image_fns = sorted(os.listdir(os.path.join(root, fn, data_point)))
            if len(image_fns) < 8:
                continue
            image_paths = ','.join([os.path.join(root,fn, data_point, tmp) for tmp in image_fns])
            if k == 'train':
                images.append(image_paths)
                labels.append(data_point)
            else: 
                valid_images.append(image_paths)
                valid_labels.append(data_point)

os.makedirs(params.outdir, exist_ok=True)


total = [[tmp1, tmp2] for tmp1, tmp2 in zip(valid_images, valid_labels)]

shuffle(total)

valid_images = [tmp[0] for tmp in total]
valid_labels = [tmp[1] for tmp in total]

valid_size = len(valid_images)//10

train = pd.DataFrame.from_dict({'image': images, 'label': labels})
valid = pd.DataFrame.from_dict({'image': valid_images[:valid_size],'label': valid_labels[:valid_size]})

train.to_csv(os.path.join(params.outdir, 'train.csv'))
valid.to_csv(os.path.join(params.outdir, 'valid.csv'))
