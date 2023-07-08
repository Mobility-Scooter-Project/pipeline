# movenet 
# inputs: image
# output: list[(float, float)] -> 9 2d-landmarks

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

import inspect

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
class MovenetPose:
    def __init__(self):
        self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        # self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        # self.input_size = 256

    def process(self, inputs):
        model = self.module.signatures['serving_default']
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(inputs, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return self.convert(keypoints_with_scores)

    def convert(self, landmarks):
        result = []
        for index in landmark_indices:
            landmark = landmarks[0][0][index]
            result.extend([landmark[0], landmark[1]])
        return result

    def __del__(self):
        pass

