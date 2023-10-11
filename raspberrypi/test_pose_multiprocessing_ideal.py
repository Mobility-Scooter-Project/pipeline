# Usage
'''
python raspberrypi/test_pose_multiprocessing_ideal.py -i "video/test.mp4" -p movenet_pose -r 10
'''

import argparse
import importlib
import sys, os; sys.path.append(os.path.abspath('.'))
import time

from multiprocessing import Process, Manager
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
parser.add_argument('-n', '--num_of_processes', type=int, required=False, default=2)
args = parser.parse_args()

pipe_module = importlib.import_module(f"{BASE_PIPE_MODULE}.{args.pipe}")
model = getattr(pipe_module, NAME_TO_MODEL[args.pipe])
cap = VideoInput(args.video_path)
frame = None

# skip frames
for i in range(args.skip):
    frame = cap.process()

MAX_PROCESSES = args.num_of_processes
REPEAT = args.repeat

def func(model, frame, i, return_dict, repeat, remainder, max_processes):
    m = model()
    repeat += (i == max_processes) * remainder
    result = []
    for _ in range(repeat):
        result.append(m.process(frame))
    return_dict[i] = result

if __name__ == '__main__':
    process_count = 0
    repeat_per_process = REPEAT // MAX_PROCESSES
    remainder = REPEAT % MAX_PROCESSES
    jobs = []
    return_dict = Manager().dict()
    i = 0
            

    for i in range(MAX_PROCESSES):
        job = Process(target=func, args=(model, frame, i+1, return_dict, repeat_per_process, remainder, MAX_PROCESSES))
        jobs.append(job)
        job.start()

    start = time.time()
    for job in jobs:
        job.join()
    end = time.time()

    total = end - start
    average = total/args.repeat
    print(sum(len(i) for i in return_dict.values()))
    print(f"Total time   : {total} sec")
    print(f"Average time : {average} sec")