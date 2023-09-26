from tqdm import tqdm
from .pipe import VideoInput, Yolov7Pose, CSVOutput
from .utils import time_func

@time_func
def process_file(in_file, out_file):
    failed_frames = []

    cap = VideoInput(in_file)
    column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
    pmodel = Yolov7Pose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for _ in tqdm(range(cap.total)):
        frame = cap.process()
        if frame is not None:
            landmarks = pmodel.process(frame)
            data_writer.process(landmarks)
        else:
            failed_frames.append(i)
    print(f"Saved {cap.total} estimations to {out_file}")
    if failed_frames:
        print(f"Estimation failed in frames: {failed_frames}")

@time_func
def process_file_in_batch(in_file, out_file, batch_size):
    cap = VideoInput(in_file)
    frame = True
    frame_buffer = []

    '''2d coordinates'''
    column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
    pmodel = Yolov7Pose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for i in tqdm(range(cap.total)):
        frame = cap.process()
        if len(frame_buffer) >= batch_size or (len(frame_buffer) and i==cap.total-1):
            landmarks_array = pmodel.process_batch(frame_buffer)
            frame_buffer.clear()
            [data_writer.process(lm) for lm in landmarks_array]
        else:
            frame_buffer.append(frame)
    print(f"Saved {cap.total} estimations to {out_file}")
