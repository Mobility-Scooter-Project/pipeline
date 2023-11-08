# Usage
'''
python raspberrypi/test_pose_multiprocessing_reality.py -i "video/test.mp4" -p movenet_pose -r 10 -b 20
'''

import argparse
import importlib
import sys, os; sys.path.append(os.path.abspath('.'))
import time

from multiprocessing import Process, Manager, Value, Queue
from queue import Empty
from pipeline.pipe.video_input import VideoInput


BASE_PIPE_MODULE = 'pipeline.pipe'
NAME_TO_MODEL = {
    'bodypix_neck': 'BodypixNeck',
    'movenet_pose': 'MovenetPose',
    'yolov7_pose': 'Yolov7Pose',
    'mediapipe_pose': 'MediapipePose',
    'movenet_tpu': 'MovenetTPU'
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

def func(model, return_dict, state, queue, batch_size):
    m = model()
    buffer = []
    state.value = 1
    def flush_buffer():
        nonlocal buffer
        for result, index in buffer:
            return_dict[index] = result
        buffer = []
    while state.value != 2 or not queue.empty():
        try:
            frame, index = queue.get(timeout=1)
            buffer.append((m.process(frame), index))
            if len(buffer) > batch_size*2:
                flush_buffer()
        except Empty:
            pass
    flush_buffer()

if __name__ == '__main__':
    start = time.time()
    return_dict = Manager().dict()
    index = 0
    queues = [Queue() for _ in range(MAX_PROCESSES)]
    states = [Value('i', 0) for _ in range(MAX_PROCESSES)] # 0: pending | 1: running | 2: done
    processes = [
        Process(target=func, args=(
            model, 
            return_dict, 
            states[i],
            queues[i],
            BATCH_SIZE
        )) for i in range(MAX_PROCESSES)
    ]
    for proc in processes:
        proc.start()

    # wait for subprocess initialization
    while any([state.value==0 for state in states]):
        time.sleep(1)
    end = time.time()
    print(f"Initialized in {end - start}s")
    
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

    for state in states:
        state.value = 2
    for proc in processes:
        proc.join()
    end = time.time()

    total = end - start
    average = total/args.repeat
    print(f"Total time   : {total} sec")
    print(f"Average time : {average} sec")
    print(len(return_dict))
    assert REPEAT==len(return_dict)