import argparse
# import pipeline.face_patch as f
# import pipeline.pose_estimation as p

parser = argparse.ArgumentParser(
    prog='Pose Estimation',
    description='Process videos with pose estimation algorithms',
)
parser.add_argument('-p', '--pipeline')
parser.add_argument('-f', '--file')
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()
print(args.pipeline, args.file, args.verbose)

