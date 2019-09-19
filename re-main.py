import argparse
import yaml
from src.estimator import Estimator


parser = argparse.ArgumentParser(description='Liveness estimator')

parser.add_argument('--config', type=str, help='Path to config of liveness')
parser.add_argument('--cuda', action='store_true', help='Use cuda for training/finetuning or not')
parser.add_argument('--epoch', default=0, type=int, help='Epoch to continue training/finetuning')
parser.add_argument('--model_path', default='', type=str, help='Path to pretrained model')

params = parser.parse_args()

config = yaml.load(params.config)

est = Estimator(params, config)
est.exec()
