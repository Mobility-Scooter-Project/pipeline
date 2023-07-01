# Bodypix model
# inputs: image
# output: (int, int) -> neck position



import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
'''
left face : rgb(110,64,170)
right face: rgb(144,61,178)
torso     : rgb(175,240,91)
'''
left_face_color = 110, 64, 170
right_face_color = 143, 61, 178
torse_color = 175, 240, 91

class BodypixNeck:
    def __init__(self):
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))

    def predict(self, image):
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        result = self.bodypix_model.predict_single(image_array)
        return result.get_colored_part_mask(result.get_mask(threshold=0.75))

    def process(self, inputs):
        mask = self.predict(inputs)
        neck_position_xs = []
        neck_position_ys = []
        torso_position_xs = []
        torso_position_ys = []
        for i, row in enumerate(mask):
            if i > len(mask)//2: break 
            for j, item in enumerate(row):
                if tuple(item) in [left_face_color, right_face_color]:
                    neck_position_xs.append(j)
                    neck_position_ys.append(i)
                if tuple(item) == torse_color:
                    torso_position_xs.append(j)
                    torso_position_ys.append(i)
        if len(neck_position_xs)>0:
            return sum(neck_position_xs)//len(neck_position_xs), max(neck_position_ys)
        if len(torso_position_xs)>0:
            return sum(torso_position_xs)//len(torso_position_xs), min(torso_position_ys)
        return None

    def __del__(self):
        pass