import os
import argparse
import pandas as pd
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Preprocess program for the dataset.')
parser.add_argument('--data_path', type=str, help='Path to the dataset')

params = parser.parse_args()

root = params.data_path

images = []
labels = []
train_subjects = ['2','3','4','5','6','7','9','10','11','12']
for fn in [tmp for tmp in os.listdir(root) if tmp in train_subjects]:
    print(fn)
    for data_point in tqdm(os.listdir(os.path.join(root, fn))):
        image_fns = sorted(os.listdir(os.path.join(root, fn, data_point)))
        if len(image_fns) < 8:
            continue
        image_paths = ','.join([os.path.join(root,fn, data_point, tmp) for tmp in image_fns])
        images.append(image_paths)
        labels.append(data_point)

if not os.path.exists('data'):
    os.mkdir('data')

train_size = len(images) - len(images)//10

train = pd.DataFrame.from_dict({'image': images[:train_size],'label': labels[:train_size]})

valid = pd.DataFrame.from_dict({'image': images[train_size:],'label': labels[train_size:]})

train.to_csv('data/train.csv')
valid.to_csv('data/valid.csv')
