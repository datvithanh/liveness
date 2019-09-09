import argparse
from trainer import Trainer, Finetuner, Tester

parser = argparse.ArgumentParser('Training anti-proofing model')

parser.add_argument('--data_path', type=str, help='Path to dataset')
parser.add_argument('--mode', type=str, help='Finetuning for not')
parser.add_argument('--extract_feature', action='store_true', help='extract feature on test mode')
parser.add_argument('--model_path', type=str, default='', help='Path to trained model')
params = parser.parse_args()

if params.mode == 'test':
    trainer = Tester(params.data_path, params.model_path, params.extract_feature)
    trainer.load_data()
    trainer.set_model()
    trainer.exec()
else:
    if params.mode == 'finetuning':
        trainer = Finetuner(params.data_path, params.model_path)
        trainer.load_data()
        trainer.set_model()
        trainer.exec()
    else:    
        trainer = Trainer(params.data_path)
        trainer.load_data()
        trainer.set_model()
        trainer.exec()
