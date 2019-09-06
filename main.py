import argparse
from trainer import Trainer, Finetuner

parser = argparse.ArgumentParser('Training anti-proofing model')

parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--finetuning', action='store_true', help='Finetuning for not')
parser.add_argument('--model_path', type=str, default='', help='Path to trained model')
param = parser.parse_args()


if param.finetuning:
    trainer = Finetuner(param.data_path, param.model_path)
    trainer.load_data()
    trainer.set_model()
    trainer.train()
else:    
    trainer = Trainer(param.data_path)
    trainer.load_data()
    trainer.set_model()
    trainer.train()