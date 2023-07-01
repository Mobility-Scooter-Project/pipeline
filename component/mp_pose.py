# mediapipe
# inputs: image
# output: list[(float, float, float)] -> 33 landmarks

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class MediapipePose:
    def __init__(self):
        self.mp_pose_model = mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def predict(self, image):
        return self.mp_pose_model.predict(image)

    def __del__(self):
        self.mp_pose_model.close()
