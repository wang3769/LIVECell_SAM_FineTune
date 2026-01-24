import numpy as np

def measure(mask, pixel_nm=1.0):
    coords = np.column_stack(np.where(mask))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    area = mask.sum() * pixel_nm**2
    width = (x_max - x_min) * pixel_nm
    height = (y_max - y_min) * pixel_nm

    return {
        "area_nm2": area,
        "width_nm": width,
        "height_nm": height
    }
