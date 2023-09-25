from tqdm import tqdm
from .pipe import VideoInput, MediapipePose, CSVOutput
from .utils import time_func

@time_func
def process_file(in_file, out_file):
    frame = True
    failed_frames = []

    cap = VideoInput(in_file)
    column_names = [f'{j}{i}' for i in range(9) for j in 'xyz']
    pmodel = MediapipePose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for i in tqdm(range(cap.total)):
        frame = cap.process()
        landmarks = pmodel.process(frame)
        if landmarks is not None:
            data_writer.process(landmarks)
        else:
            failed_frames.append(i)
    print(f"Saved {cap.total-len(failed_frames)} estimations to {out_file}")
    if failed_frames:
        print(f"Estimation failed in frames: {failed_frames}")
