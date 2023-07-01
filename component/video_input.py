import cv2

class VideoInput:
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture(video_src)

    def process(self):
        success, frame = self.cap.read()
        if success:
            return frame
        return None
    
    def __del__(self):
        self.cap.release()
