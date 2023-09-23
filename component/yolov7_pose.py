# yolov7 
# inputs: image
# output: list[(float, float)] -> 9 2d-landmarks


import torch
from torchvision import transforms
import numpy as np
from pipeline.utils.datasets import letterbox
from pipeline.utils.general import non_max_suppression_kpt
from pipeline.utils.plots import output_to_keypoint

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

class Yolov7Pose:
    def __init__(self):
        device = torch.device("cpu")
        weigths = torch.load('pipeline/yolov7-w6-pose.pt', map_location=device)
        self.model = weigths['model']
        _ = self.model.float().eval()
        # if torch.cuda.is_available():
        #     self.model.half().to(device)


    def process(self, inputs):
        image = letterbox(inputs, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        with torch.no_grad():
            output, _ = self.model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        length = output.shape[0]
        if length == 0:
            return None
        # take only the first person in the list, the first 7 elements in the output[0] list is unknown
        landmarks = output[0, 7:].T
        return self.convert(landmarks)


    def convert(self, landmarks):
        result = []
        # outputs are in pixel value of 960x960 image
        for index in landmark_indices:
            result.extend([landmarks[index*3]/960, landmarks[1+index*3]/960])
        return result

    def __del__(self):
        pass
