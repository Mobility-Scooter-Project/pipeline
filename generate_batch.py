import os

videos = os.listdir("video")
out = os.path.join("out", "Yolov7")
batches = "batches"
bf = open("run_batches.sh", 'w')
for i, video in enumerate(videos):
    batch_file = os.path.join(batches, str(i).zfill(3) + '.sbatch')
    with open(batch_file, 'w') as f:
        f.write(f'''#!/bin/bash
#SBATCH --job-name=Job
#SBATCH --output=Job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=compute
source /data03/home/ruoqihuang/anaconda3/etc/profile.d/conda.sh
conda activate pipeline
chmod +x main.py
srun main.py -i "{os.path.join("video", video)}" -o "{os.path.join(out, video+'.csv')}" -p yolov7
''')
        bf.write(f'sbatch {batch_file}\n')
    
bf.close()
