# yolov7 
# inputs: image
# output: list[(float, float)] -> 9 2d-landmarks


import torch
from torchvision import transforms
import numpy as np
from ..utils.yolov7.datasets import letterbox
from ..utils.yolov7.general import non_max_suppression_kpt
from ..utils.yolov7.plots import output_to_keypoint

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
        if torch.cuda.is_available():
            print("The following GPU will be used: ", torch.cuda.get_device_name(0), "\n")

            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        weights = torch.load('assets/yolov7-w6-pose.pt', map_location=device)
        self.model = weights['model']
        _ = self.model.float().eval()
        self.input_size = 192
        #if torch.cuda.is_available():
             #self.model.to("cuda")


    def process(self, inputs):
        image = letterbox(inputs, self.input_size, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        #move input to gpu if available
        if torch.cuda.is_available():
            image = image.to(torch.device("cuda:0"))

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
        for index in landmark_indices:
            result.extend([landmarks[index*3]/self.input_size, landmarks[1+index*3]/self.input_size])
        return result

    def __del__(self):
        pass

    def process_batch(self, inputs):
        images = []
        for i in inputs:
            image = letterbox(i, self.input_size, stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            images.append(image.numpy())
        batch_input = torch.tensor(np.array(images))

        #move input to gpu if available
        if torch.cuda.is_available():
            batch_input = batch_input.to(torch.device("cuda:0"))

        landmarks_array = []
        with torch.no_grad():
            output, _ = self.model(batch_input)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        for i in output:
            with torch.no_grad():
                output = output_to_keypoint([i])
            length = output.shape[0]
            if length == 0:
                landmarks_array.append(None)
            else:
                # take only the first person in the list, the first 7 elements in the output[0] list is unknown
                landmarks_array.append(self.convert(output[0, 7:].T))
        return landmarks_array
