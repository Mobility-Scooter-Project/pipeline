import argparse
import importlib

# Define constants
BASE_PIPELINE_MODULE = "pipeline"

# Set up the argument parser
parser = argparse.ArgumentParser(
    prog='Pose Estimation',
    description='Process videos with pose estimation algorithms',
)
parser.add_argument('-p', '--pipeline', required=True)
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-b', '--batchsize', type=int, default=1)


args = parser.parse_args()

pipeline_module = importlib.import_module(f"{BASE_PIPELINE_MODULE}.{args.pipeline}")
if args.batchsize == 1:
    process_file = pipeline_module.process_file
    process_file(args.input, args.output)
else:
    process_file = pipeline_module.process_file_in_batch
    process_file(args.input, args.output, args.batchsize)

