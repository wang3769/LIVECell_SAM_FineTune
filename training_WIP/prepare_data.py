import json
import numpy as np
import cv2

def labelme_to_mask(json_path, image_shape):
    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros(image_shape, dtype=np.uint8)
    for shape in data["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask
