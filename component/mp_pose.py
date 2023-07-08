# mediapipe
# inputs: image
# output: list[(float, float, float)] -> 9 landmarks

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
'''
0 - nose
1 - left eye (inner)
2 - left eye
3 - left eye (outer)
4 - right eye (inner)
5 - right eye
6 - right eye (outer)
7 - left ear
8 - right ear
9 - mouth (left)
10 - mouth (right)
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
17 - left pinky
18 - right pinky
19 - left index
20 - right index
21 - left thumb
22 - right thumb
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
29 - left heel
30 - right heel
31 - left foot index
32 - right foot index
'''

class MediapipePose:
    def __init__(self):
        self.mp_pose_model = mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, image):
        result = self.mp_pose_model.process(image)
        if result.pose_landmarks is None:
            return None
        return self.convert(result.pose_world_landmarks.landmark)

    def convert(self, landmarks):
        result = []
        for index in landmark_indices:
            landmark = landmarks[index]
            result.extend([landmark.x, landmark.y, landmark.z])
        return result

    def __del__(self):
        self.mp_pose_model.close()
