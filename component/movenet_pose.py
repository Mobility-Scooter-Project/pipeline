# movenet
# inputs: image
# output: list[(float, float, float)] -> 9 landmarks

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

class MovenetPose:
    def __init__(self):
        self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192

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
        return keypoints_with_scores

    def convert(self, landmarks):
        result = []
        for index in landmark_indices:
            landmark = landmarks[index]
            result.extend([landmark.x, landmark.y, landmark.z])
        return result

    def __del__(self):
        pass

image = cv2.imread('joe.png')
m = MovenetPose()
print(m.process(image))
