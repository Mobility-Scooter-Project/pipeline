import numpy as np
import cv2
from tqdm import tqdm
from .pipe.video_input import VideoInput
from .pipe.bodypix_neck import BodypixNeck
from .pipe.mediapipe_pose import MediapipePose
from .pipe.csv_output import CSVOutput
from .utils import time_func

WINDOW_NAME = "Face Patch"
SHOW_VIDEO = False
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
POINT_COLOR = (0, 0, 0) # black
PROCESS_WIDTH, PROCESS_HEIGHT = 640, 360
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 540
HEAD_WIDTH, HEAD_HEIGHT = 180, 200


class FacePatch:
    def __init__( self, head_src, head_width, head_height, offset_x, offset_y):
        self.neck_model = BodypixNeck()
        self.prior_neck_position = 0, 0
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.head = cv2.resize(cv2.imread(head_src, cv2.IMREAD_UNCHANGED), (head_height, head_width))

    def process(self, inputs):
        frame = cv2.resize(inputs, (PROCESS_WIDTH, PROCESS_HEIGHT))
        window_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        height_diff = WINDOW_HEIGHT - PROCESS_HEIGHT
        self.blit(frame, window_image, 0, height_diff)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        neck_pos = self.neck_model.process(frame)
        if neck_pos is None:
            neck_pos = self.prior_neck_position
        self.prior_neck_position = neck_pos
        x, y = neck_pos
        self.blit_head(self.head, window_image, x-HEAD_WIDTH//2+self.offset_x, y+height_diff-HEAD_HEIGHT+self.offset_y)
        return window_image


    def blit(self, src, dest, x, y):
        dest[y:y+src.shape[0], x:x+src.shape[1]] = src

    def blit_head(self, src, dest, x, y):
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                a, b = i+y, j+x
                if src[i][j][3]>0:
                    if 0<=a<WINDOW_HEIGHT and 0<=b<WINDOW_WIDTH:
                        dest[a][b] = src[i][j][:3]

fmodel = FacePatch(
    head_src="assets/joe.png", 
    head_width=180, 
    head_height=200, 
    offset_x=-10, 
    offset_y=20
)

@time_func
def process_file(in_file, out_file):
    cap = VideoInput(in_file)
    frame = True
    failed_frames = []

    '''3d coordinates'''
    column_names = [f'{j}{i}' for i in range(9) for j in 'xyz']
    pmodel = MediapipePose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for i in tqdm(range(cap.total)):
        frame = cap.process()

        if frame is not None:
            window_image = fmodel.process(frame)
            landmarks = pmodel.process(window_image)
            if landmarks is not None:
                data_writer.process(landmarks)
            else:
                failed_frames.append(i)
            if SHOW_VIDEO:
                cv2.imshow(WINDOW_NAME, window_image)
                cv2.waitKey(1)
        else:
            failed_frames.append(i)
    print(f"Saved {cap.total-len(failed_frames)} estimations to {out_file}")
    if failed_frames:
        print(f"Estimation failed in frames: {failed_frames}")
