import argparse
from trainer import Trainer

parser = argparse.ArgumentParser('Training anti-proofing model')

parser.add_argument('--data_path', type=str, help='Path to dataset')

param = parser.parse_args()


trainer = Trainer(param.data_path)
trainer.load_data()
trainer.set_model()
trainer.train()