import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class VideoOutput:
    def __init__(self, path, fps=30, dimension=(640, 480)):
        self.video_writer = cv2.VideoWriter(path, fourcc, fps, dimension)

    def process(self, inputs):
        self.video_writer.write(inputs)

    def __del__(self):
        self.video_writer.release()