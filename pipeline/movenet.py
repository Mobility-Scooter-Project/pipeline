from .pipe import VideoInput, MovenetPose, CSVOutput
from tqdm import tqdm
from .utils import time_func

@time_func
def process_file(in_file, out_file):
    cap = VideoInput(in_file)
    frame = True

    '''2d coordinates'''
    column_names = [f'{j}{i}' for i in range(9) for j in 'xy']
    pmodel = MovenetPose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for _ in tqdm(range(cap.total)):
        frame = cap.process()
        landmarks = pmodel.process(frame)
        data_writer.process(landmarks)
    print(f"Saved {cap.total} estimations to {out_file}")
