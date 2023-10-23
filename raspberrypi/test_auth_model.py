# Usage
'''
python raspberrypi/test_auth_model.py -r 100
'''

import torch
import argparse

import sys, os; sys.path.append(os.path.abspath('.'))
from pipeline.pipe.auth_model import AuthModel
from raspberrypi.utils import repeat_n_times_and_analysis


parser = argparse.ArgumentParser(prog='Authentication model Testing')
parser.add_argument('-r', '--repeat', type=int, required=True)
args = parser.parse_args()

mock_input = torch.randn(1, 27, 128)
process_func = AuthModel().process

@repeat_n_times_and_analysis(args.repeat)
def test_processing():
    process_func(mock_input)
