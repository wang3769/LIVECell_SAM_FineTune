import cv2
import numpy as np

def localize_box(image, box_size=100):
    edges = cv2.Canny(image, 50, 150)
    y, x = np.unravel_index(np.argmax(edges), edges.shape)
    
    h, w = image.shape[:2]
    half = box_size // 2

    x_min = max(0, x - half)
    x_max = min(w, x + half)
    y_min = max(0, y - half)
    y_max = min(h, y + half)

    return [x_min, y_min, x_max, y_max]
# the return is consistent with SAM box prompt format; left top corner (x_min, y_min), right bottom corner (x_max, y_max)