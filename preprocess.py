import os
import argparse
import pandas as pd
from tqdm import tqdm
from random import shuffle
parser = argparse.ArgumentParser(description='Preprocess program for the dataset.')
parser.add_argument('--data_path', type=str, help='Path to the dataset')
parser.add_argument('--test', action='store_true', help='prepare test set')

params = parser.parse_args()

root = params.data_path

images = []
labels = []
train_subjects = ['2','3','4','5','6','7','9','10','11','12']
if params.test:
    subjects = [tmp for tmp in os.listdir(root) if tmp not in train_subjects]
else:
    subjects = [tmp for tmp in os.listdir(root) if tmp in train_subjects]
for fn in subjects:
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

if params.test:
    test = pd.DataFrame.from_dict({'image': images, 'label': labels})
    test.to_csv('data/test.csv')
else:
    total = [[tmp1, tmp2] for tmp1, tmp2 in zip(images, labels)]

    shuffle(total)

    images = [tmp[0] for tmp in total]
    labels = [tmp[1] for tmp in total]

    train_size = len(images) - len(images)//10

    train = pd.DataFrame.from_dict({'image': images[:train_size],'label': labels[:train_size]})
    valid = pd.DataFrame.from_dict({'image': images[train_size:],'label': labels[train_size:]})

    train.to_csv('data/train.csv')
    valid.to_csv('data/valid.csv')
