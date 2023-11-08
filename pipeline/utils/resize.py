import numpy as np

def resize_with_pad(image, target_height, target_width):
    height, width, channels = image.shape
    aspect = width / height
    target_aspect = target_width / target_height

    if aspect > target_aspect:
        # Resize based on width
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        # Resize based on height
        new_height = target_height
        new_width = int(target_height * aspect)
        
    # Resize the image using NumPy
    y_scale = new_height / height
    x_scale = new_width / width
    resized_image = np.zeros((new_height, new_width, channels))
    for i in range(new_height):
        for j in range(new_width):
            x = int(j / x_scale)
            y = int(i / y_scale)
            resized_image[i, j] = image[y, x]

    # Pad the image
    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad
    padded_image = np.pad(resized_image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')

    return padded_image