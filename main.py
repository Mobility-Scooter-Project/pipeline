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

args = parser.parse_args()

# Import module dynamically
try:
    pipeline_module = importlib.import_module(f"{BASE_PIPELINE_MODULE}.{args.pipeline}")
    process_file = pipeline_module.process_file
    process_file(args.input, args.output)
except ImportError:
    print(f"Pipeline '{args.pipeline}' not found!")
except AttributeError:
    print(f"'process_file' function not found in pipeline '{args.pipeline}'!")
