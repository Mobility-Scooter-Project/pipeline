# Pipelines
This repo is created to run data preprocessing pipelines.
- pipe initializes a model or resource and provides destructor as `__del__` to release it.
- If a GPU is detected with CUDA while using YOLOv7, that GPU will be utilized for computations.

## setup
- install anaconda
```
conda activate
conda create -n pipeline python=3.9 -y
conda activate pipeline
pip install -r requirements.txt
```

## Usage of Pose Extraction
```
python main.py -p PIPELINE -i INPUT -o OUTPUT [-b BATCHSIZE]
```
- `-p, --pipeline`
    - Specifies the pipeline module to use for pose estimation from the pipeline directory. 
        - `movenet`
        - `mediapipe`
        - `face_patch_to_mediapipe`
        - `yolov7`
        - `yolov7_patched` (added per frame pose extraction confidence, frame id, corrected letterbox y distortion)
- `-i, --input`
    - The path to the input video file.
- `-o, --output`
    - The path to the output CSV file. 
- `-b, --batchsize` (Optional) 
    - Batch size for processing. 
    - only applicable to `yolov7`

## Usage of IMU and Pose Synchronization 
batch_preprocess does the following: IMU calibration, pose cleanup, frame-index repair, pose/IMU synchronization, merge onto IMU grid (averages pose over a local time window around each IMU sample
), QC and export.
```
python batch_preprocess.py <pose_root> <video_motion_root>  --out <output_folder> --skip-existing
```
- `<pose_root>`
    - The path to the folder that contains pose csv files, organized by session IDs (subfolders)
- `<video_motion_root>`
    - The path to the folder that contains motion IMU csv files and processed videos, organized by session IDs (subfolders)
- `--out`
    - The path to the output merged csv files that contains synchronized IMU + pose data, a QC JSON with sync and gate metrics. 
- `--skip-existing`
    - Skip the session which has been synchronized before

## models
- Download yolov7 model file to `assets/` on GitHub releases if needed~ 
- Download an untrained auth model to `assets/` to test the authentication model

## references
- https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md
- https://coral.ai/docs/accelerator/get-started/#requirements
