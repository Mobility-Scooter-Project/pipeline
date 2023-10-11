# Usage
'''
python raspberrypi/test_pose_estimation.py -i "video/test.mp4" -p movenet_pose -r 10
'''

import argparse
import importlib
import sys, os; sys.path.append(os.path.abspath('.'))
from raspberrypi.utils import repeat_n_times_and_analysis
from pipeline.pipe.video_input import VideoInput


BASE_PIPE_MODULE = 'pipeline.pipe'
NAME_TO_MODEL = {
    'bodypix_neck': 'BodypixNeck',
    'movenet_pose': 'MovenetPose',
    'yolov7_pose': 'Yolov7Pose',
    'mediapipe_pose': 'MediapipePose',
}

parser = argparse.ArgumentParser(prog='Pose Estimation Testing')
parser.add_argument('-i', '--video_path', required=True)
parser.add_argument('-p', '--pipe', required=True)
parser.add_argument('-r', '--repeat', type=int, required=True)
parser.add_argument('-s', '--skip', type=int, required=False, default=1000)
args = parser.parse_args()

pipe_module = importlib.import_module(f"{BASE_PIPE_MODULE}.{args.pipe}")
process_func = getattr(pipe_module, NAME_TO_MODEL[args.pipe])().process
cap = VideoInput(args.video_path)
frame = None

# skip frames
for i in range(args.skip):
    frame = cap.process()

@repeat_n_times_and_analysis(args.repeat)
def test_processing():
    process_func(frame)
