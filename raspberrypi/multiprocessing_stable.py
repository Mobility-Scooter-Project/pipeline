# Usage
'''
python raspberrypi/multiprocessing_stable.py -i "video/test.mp4" -p mediapipe -r 1000 -b 20 -n 4
'''

import argparse
import importlib
import sys, os; sys.path.append(os.path.abspath('.'))
import time
from multiprocessing import Process, Value, Queue
from threading import Thread
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

def worker(model, state, in_queue, out_queue, batch_size):
    m = model()
    buffer = []
    state.value = 1
    def flush_buffer():
        for result, index in buffer:
            out_queue.put((result, index)) 
        buffer = []
    while state.value != 2 or not in_queue.empty():
        try:
            frame, index = in_queue.get()
            buffer.append((m.process(frame), index))
            if len(buffer) > batch_size*2:
                flush_buffer()
        except Empty:
            pass
    flush_buffer()


def grouper(in_queues, out_queue, done, batch_size):
    buffer = {}
    current_index = 0
    def flush_buffer():
        nonlocal current_index
        result = buffer.get(current_index)
        while result is not None:
            out_queue.append(result)
            del buffer[current_index]
            current_index += 1
            result = buffer.get(current_index)

    while not done.value or any([not q.empty() for q in in_queues]):
        for q in in_queues:
            try:
                for _ in range(batch_size):
                    frame, index = q.get(timeout=0.1)
                    buffer[index] = frame
            except Empty:
                pass
        flush_buffer()
    flush_buffer()

if __name__ == '__main__':
    start = time.time()
    index = 0
    in_queues = [Queue() for _ in range(MAX_PROCESSES)]
    out_queues = [Queue() for _ in range(MAX_PROCESSES)]
    states = [Value('i', 0) for _ in range(MAX_PROCESSES)] # 0: pending | 1: running | 2: done
    result_queue = []
    grouper_done = Value('b', False)
    grouper_thread = Thread(target=grouper, args=(out_queues, result_queue, grouper_done, BATCH_SIZE))
    grouper_thread.start()
    processes = [
        Process(target=worker, args=(
            model, 
            states[i],
            in_queues[i],
            out_queues[i],
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
            if in_queues[i].qsize() < BATCH_SIZE*2:
                for _ in range(BATCH_SIZE):
                    if index >= REPEAT:
                        break
                    in_queues[i].put((frame, index))
                    index += 1
                busy = False
        if busy:
            time.sleep(0.01)

    for state in states:
        state.value = 2
    for proc in processes:
        proc.join()
    grouper_done.value = True
    grouper_thread.join()
    end = time.time()

    total = end - start
    average = total/args.repeat
    print(f"Total time   : {total} sec")
    print(f"Average time : {average} sec")
    print(len(result_queue))
    assert REPEAT==len(result_queue)
