# Usage
'''
python raspberrypi/test_pose_multiprocessing_reality.py -i "video/test.mp4" -p movenet_pose -r 10 -b 20
'''

import argparse
import importlib
import sys, os; sys.path.append(os.path.abspath('.'))
import time
from contextlib import suppress

from multiprocessing import Process, Manager, Value, Queue
from queue import Empty
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
parser.add_argument('-b', '--batch_size', type=int, required=False, default=1)
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
BATCH_SIZE = args.batch_size

def func(model, return_dict, done, queue, batch_size):
    m = model()
    results = []
    while not done.value or not queue.empty():
        try:
            frame, index = queue.get(timeout=1)
            results.append((m.process(frame), index))
            if len(results) > batch_size*2:
                for result, index in results:
                    return_dict[index] = result
        except Empty:
            pass
    for result, index in results:
        return_dict[index] = result

if __name__ == '__main__':
    return_dict = Manager().dict()
    index = 0
    queues = [Queue() for _ in range(MAX_PROCESSES)]
    dones = [Value('b', False) for _ in range(MAX_PROCESSES)]
    processes = [
        Process(target=func, args=(
            model, 
            return_dict, 
            dones[i],
            queues[i],
            BATCH_SIZE
        )) for i in range(MAX_PROCESSES)
    ]
    for proc in processes:
        proc.start()

    # initialize subprocesses 
    time.sleep(20)
    
    start = time.time()
    while index < REPEAT:
        busy = True
        for i in range(MAX_PROCESSES):
            if queues[i].qsize() < BATCH_SIZE*2:
                for _ in range(BATCH_SIZE):
                    if index >= REPEAT:
                        break
                    queues[i].put((frame, index))
                    index += 1
                busy = False
        if busy:
            time.sleep(0.01)

    for done in dones:
        done.value = True
    for proc in processes:
        proc.join()
    end = time.time()

    total = end - start
    average = total/args.repeat
    print(f"Total time   : {total} sec")
    print(f"Average time : {average} sec")
    print(len(return_dict))
    assert REPEAT==len(return_dict)