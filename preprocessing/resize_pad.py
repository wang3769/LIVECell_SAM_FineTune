import cv2
import numpy as np

def resize_and_pad(image, target=1024):
    H, W = image.shape[:2]
    scale = target / max(H, W)

    new_H, new_W = int(H * scale), int(W * scale)
    resized = cv2.resize(image, (new_W, new_H))

    pad_H = target - new_H
    pad_W = target - new_W

    padded = cv2.copyMakeBorder(
        resized,
        0, pad_H,
        0, pad_W,
        cv2.BORDER_CONSTANT,
        value=0
    )

    meta = {
        "scale": scale,
        "pad_h": pad_H,
        "pad_w": pad_W,
        "orig_shape": (H, W)
    }

    return padded, meta
