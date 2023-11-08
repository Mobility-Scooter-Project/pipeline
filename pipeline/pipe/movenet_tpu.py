# movenet 
# inputs: image
# output: list[(float, float)] -> 9 2d-landmarks
import os
import numpy as np
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from ..utils.resize import resize_with_pad


landmark_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12]
'''
0 - nose
1 - left eye
2 - right eye
3 - left ear
4 - right ear
5 - left shoulder
6 - right shoulder
7 - left elbow
8 - right elbow
9 - left wrist
10 - right wrist
11 - left hip
12 - right hip
13 - left knee
14 - right knee
15 - left ankle
16 - right ankle
'''
_NUM_KEYPOINTS = 17


class MovenetTPU:
    def __init__(self):
        self.interpreter = make_interpreter(os.path.join('assets', 'movenet_tpu.tflite'))
        self.interpreter.allocate_tensors()
        self.input_dim = common.input_size(self.interpreter)

    def process(self, inputs):
        input_image = resize_with_pad(inputs, *self.input_dim)
        input_image = np.array([input_image])
        common.set_input(self.interpreter, input_image)
        self.interpreter.invoke()
        landmarks = common.output_tensor(self.interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
        return self.convert(landmarks)

    def convert(self, landmarks):
        result = []
        for index in landmark_indices:
            landmark = landmarks[index]
            result.extend([landmark[0], landmark[1]])
        return result
