import numpy as np
import cv2
import sys
import os
from .pipe import CSVOutput, MovenetPose

'''2d coordinates'''
column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
pmodel = MovenetPose()

@time_func
def process_file(file):
    frames = frame(file)
    frame = next(frames)

    output_path = os.path.join("out", f"{file}.csv")
    if os.path.exists(output_path):
        os.remove(output_path)
    data_writer = CSVOutput(output_path, column_names)

    print(f"Estimation started for {file}")
    while frame:
        landmarks = pmodel.process(frame)
        if landmarks is not None:
            data_writer.process(landmarks)
        frame = next(frames)
    print(f"Finished {file}")
