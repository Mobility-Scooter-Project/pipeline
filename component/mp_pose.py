# mediapipe
# inputs: image
# output: list[(float, float, float)] -> 9 landmarks

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

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
