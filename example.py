import argparse
from train_words import evaluate_setting

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')

args = parser.parse_args()
gpu_id = args.gpu_id

# name of the dataset and path to the dataset files
dataset = ('IAM', '../IAM')

# max epochs
N = 120

# experimental setting
setting = {
    'lowercase': True,
    'path_ctc': True,
    'path_s2s': True,
    'path_autoencoder': True,
    'train_external_lm': True,
    'start_external_lm': 0,
    'binarize': False,
    'start_binarize': 0,
    'feat_size': 512
}

# report name
rname = './tmp'

evaluate_setting(setting, dataset, gpu_id, report_name=rname, epochs=N)