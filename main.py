#!/data03/home/ruoqihuang/anaconda3/envs/tf/bin/python
import argparse
import importlib
import torch

# Define constants
BASE_PIPELINE_MODULE = "pipeline"

# Set up the argument parser
parser = argparse.ArgumentParser(
    prog='Pose Estimation',
    description='Process videos with pose estimation algorithms',
)
parser.add_argument('-p', '--pipeline', required=True)
# output file should append `.csv` to video file name (not enforced)
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-b', '--batchsize', type=int, default=1)


args = parser.parse_args()

#check for GPU, and initialize GPU using cuda command
if torch.cuda.device_count() > 0:
    print("GPU is compatible with CUDA.")

pipeline_module = importlib.import_module(f"{BASE_PIPELINE_MODULE}.{args.pipeline}")
process_file = pipeline_module.process_file
process_args = [args.input, args.output]
if args.batchsize > 1:
    process_file = pipeline_module.process_file_in_batch
    process_args = [args.input, args.output, args.batchsize]
process_file(*process_args)

