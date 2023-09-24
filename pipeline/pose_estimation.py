import numpy as np
import cv2
import sys
import os
from .pipe import VideoInput, VideoOutput, MediapipePose, CSVOutput, MovenetPose

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
POINT_COLOR = (0, 0, 0) # black

# webcam
# VIDEO_SRC = 0 
# mp4 video
file = "test.mp4"


FPS = 100
WINDOW_NAME = "Face Patch"
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 540
SHOW_VIDEO = False


cap = VideoInput(os.path.join("video", file))
n = 0

'''2d coordinates'''
# column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
# pmodel = MovenetPose()

'''3d coordinates'''
column_names = [f'{j}{i}' for i in range(9) for j in 'xyz']
pmodel = MediapipePose()

output_path = os.path.join("out", f"{file}.csv")
if os.path.exists(output_path):
    os.remove(output_path)
data_writer = CSVOutput(output_path, column_names)

print(f"Estimation started for {file}")
while 1:
    # 30 fps video
    frame = cap.process()
    if frame is None:
        print("??")
        break
    n += 1
    if n%1000==0:
        print(f"{n} frames processed")
    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = pmodel.process(image)
    if landmarks is not None:
        data_writer.process(landmarks)
    else:
        print(f"No pose detected for frame <{n}>")
#this code doesn't  work on windows (under some conditions)