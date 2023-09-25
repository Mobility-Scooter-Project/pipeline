import os
from .pipe import VideoInput, MovenetPose, CSVOutput
# from .utils import time_func

def process_file(in_file, out_file):
    cap = VideoInput(in_file)
    column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
    pmodel = MovenetPose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for _ in range(cap.total):
        frame = cap.process()
        landmarks = pmodel.process(frame)
        data_writer.process(landmarks)
    print(f"Saved estimation to {out_file}")
