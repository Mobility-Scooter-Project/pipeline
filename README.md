# Pipelines
This repo is created to run and combine different machine learning pipelines with other tasks.
- pipe initializes a model or resource and provides destructor as `__del__` to release it.
- If a GPU is detected with CUDA, the program will utilize that GPU for computations.

## setup
- install anaconda
```
conda activate
conda create -n pipeline python=3.9 -y
conda activate pipeline
pip install -r requirements.txt
```

## Usage
```
python main.py -p PIPELINE -i INPUT -o OUTPUT [-b BATCHSIZE]
```
- `-p, --pipeline`
    - Specifies the pipeline module to use for pose estimation from the pipeline directory.
        - `movenet`
        - `mediapipe`
        - `face_patch_to_mediapipe`
        - `yolov7`
- `-i, --input`
    - The path to the input video file.
- `-o, --output`
    - The path to the output CSV file. 
- `-b, --batchsize` (Optional) 
    - Batch size for processing. 
    - only applicable to `yolov7`

## models
Download yolov7 model file to `assets/` on GitHub releases if needed~ 

## references
- https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md
