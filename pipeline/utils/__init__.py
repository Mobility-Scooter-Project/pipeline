import time
from ..pipe import VideoInput, VideoOutput, CSVOutput

def frame(file):
    cap = VideoInput(os.path.join("video", file))
    n = 0
    while 1:
        frame = cap.process()
        if frame is None:
            yield None
        n += 1
        if n%1000==0:
            print(f"{n} frames processed")
            yield frame

def time_func(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print(f"Elapsed time: {time.time()-start} seconds")
    return result
