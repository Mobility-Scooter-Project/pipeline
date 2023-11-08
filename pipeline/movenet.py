from tqdm import tqdm
from .pipe.video_input import VideoInput
from .pipe.movenet_pose import MovenetPose
from .pipe.csv_output import CSVOutput
from .utils import time_func

'''2d coordinates'''
column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
pmodel = MovenetPose()

@time_func
def process_file(in_file, out_file):
    cap = VideoInput(in_file)
    data_writer = CSVOutput(out_file, column_names)
    failed_frames = []

    print(f"Estimation started for {in_file}")
    for i in tqdm(range(cap.total)):
        frame = cap.process()
        if frame is not None:
            landmarks = pmodel.process(frame)
            data_writer.process(landmarks)
        else:
            failed_frames.append(i)
    print(f"Saved {cap.total} estimations to {out_file}")
    if failed_frames:
        print(f"Estimation failed in frames: {failed_frames}")
