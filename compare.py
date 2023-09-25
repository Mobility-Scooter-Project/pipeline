import os
import argparse
import importlib

# Define constants
BASE_PIPELINE_MODULE = "pipeline"

# Set up the argument parser
parser = argparse.ArgumentParser(
    prog='Compare conversion and complete those missing',
    description='Process videos with pose estimation algorithms',
)
parser.add_argument('-p', '--pipeline', required=True)
parser.add_argument('-i', '--input', required=True, help="Video Folder")
parser.add_argument('-o', '--output', required=True, help="Result Folder")
parser.add_argument('-b', '--batchsize', type=int, default=1)

args = parser.parse_args()

pipeline_module = importlib.import_module(f"{BASE_PIPELINE_MODULE}.{args.pipeline}")
process_file = pipeline_module.process_file
process_args = []
if args.batchsize > 1:
    process_file = pipeline_module.process_file_in_batch
    process_args = [args.batchsize]

videos = os.listdir(args.input)
results = set(os.listdir(args.output))
for video in videos:
    if video not in results:
        in_path = os.path.join(args.input, video)
        out_path = os.path.join(args.output, video+'.csv')
        process_file(in_path, out_path, *process_args)
