"""
Patch for pipeline/yolov7.py and pipeline/pipe/yolov7_pose.py.

Changes, in order of importance:
  1. Write the frame index for every frame, including frames with no detection,
     so pose rows can be placed on a time axis.
  2. Return keypoint confidences and the detection count instead of discarding them.
  3. Select the ridden subject rather than output[0] (highest NMS score), which
     switches to bystanders.  Here: the detection whose centre is closest to the
     previous accepted pose, falling back to the largest box.
  4. Undo the letterbox transform so coordinates are fractions of the original
     frame, not of the padded 192-pixel canvas.
  5. Raise input_size; 192 is well below what yolov7-w6-pose expects and is the
     main driver of collapsed skeletons.
"""
import numpy as np, torch
from torchvision import transforms
from tqdm import tqdm
from .pipe.video_input import VideoInput
from .pipe.csv_output import CSVOutput
from .utils import time_func
from .utils.yolov7.datasets import letterbox
from .utils.yolov7.general import non_max_suppression_kpt
from .utils.yolov7.plots import output_to_keypoint

LANDMARK_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12]   # nose, shoulders, elbows, wrists, hips


class Yolov7PoseFixed:
    def __init__(self, weights='assets/yolov7-w6-pose.pt', input_size=640):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = torch.load(weights, map_location=device)['model'].float().eval()
        self.input_size = input_size
        self.prev_centre = None

    def process(self, frame):
        """Returns (coords, confs, n_detections) with coords as fractions of the frame."""
        h, w = frame.shape[:2]
        img, ratio, (dw, dh) = letterbox(frame, self.input_size, stride=64, auto=True)
        t = transforms.ToTensor()(img)[None].to(self.device)
        with torch.no_grad():
            output, _ = self.model(t)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'],
                                         nkpt=self.model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        n = output.shape[0]
        if n == 0:
            return None, None, 0
        pick = self._select(output)
        lm = output[pick, 7:].T
        r = ratio[0]
        coords, confs = [], []
        for idx in LANDMARK_INDICES:
            x = (lm[idx*3] - dw) / r / w          # undo padding, scaling, then normalise
            y = (lm[1 + idx*3] - dh) / r / h
            coords.extend([x, y]); confs.append(lm[2 + idx*3])
        self.prev_centre = (np.mean(coords[0::2]), np.mean(coords[1::2]))
        return coords, confs, n

    def _select(self, output):
        """Track the rider instead of taking the top-scoring detection."""
        boxes = output[:, 2:6]                      # cx, cy, w, h in letterbox pixels
        if self.prev_centre is None:
            return int(np.argmax(boxes[:, 2] * boxes[:, 3]))     # largest subject
        d = np.hypot(boxes[:, 0]/self.input_size - self.prev_centre[0],
                     boxes[:, 1]/self.input_size - self.prev_centre[1])
        return int(np.argmin(d))


@time_func
def process_file(in_file, out_file, input_size=640):
    cap = VideoInput(in_file)
    cols = (['frame', 'detected', 'n_detections']
            + [f'{j}{i}' for i in range(9) for j in 'xy']
            + [f'conf{i}' for i in range(9)])
    pmodel = Yolov7PoseFixed(input_size=input_size)
    writer = CSVOutput(out_file, cols)

    n_detected = 0
    for i in tqdm(range(cap.total)):
        frame = cap.process()
        if frame is None:
            writer.process([i, 0, 0] + ['']*27)      # frame unreadable: row still written
            continue
        coords, confs, n = pmodel.process(frame)
        if coords is None:
            writer.process([i, 0, 0] + ['']*27)      # no detection: row still written
        else:
            writer.process([i, 1, n] + list(coords) + list(confs))
            n_detected += 1
    print(f"{n_detected}/{cap.total} frames with a detection -> {out_file}")
