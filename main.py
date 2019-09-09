import argparse
from trainer import Trainer, Finetuner, Tester

parser = argparse.ArgumentParser('Training anti-proofing model')

parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--mode', type=str, help='Finetuning for not')
parser.add_argument('--model_path', type=str, default='', help='Path to trained model')
param = parser.parse_args()

if param.mode == 'test':
    trainer = Tester(param.data_path, param.model_path)
    trainer.load_data()
    trainer.set_model()
    trainer.test()
else:
    if param.mode == 'finetuning':
        trainer = Finetuner(param.data_path, param.model_path)
        trainer.load_data()
        trainer.set_model()
        trainer.finetune()
    else:    
        trainer = Trainer(param.data_path)
        trainer.load_data()
        trainer.set_model()
        trainer.train()
