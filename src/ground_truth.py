import numpy as np

from data import load_labels
from transforms import rectcam_to_lidar


def gt_boxes(i, calib):
    boxes, classes, ignored = [], [], []
    for lab in load_labels(i):
        if lab["type"] == "DontCare":
            continue
        h, w, l = lab["hwl"]
        ry = lab["ry"]
        bottom = rectcam_to_lidar(calib, lab["loc"][None, :])[0]                     # centro del fondo, in LiDAR
        tip = rectcam_to_lidar(calib, (lab["loc"] + [np.cos(ry), 0, -np.sin(ry)])[None, :])[0]   # 1 m verso il muso
        psi = np.arctan2(tip[1] - bottom[1], tip[0] - bottom[0])                     # direzione = differenza fra due punti
        boxes.append([bottom[0], bottom[1], bottom[2], l, w, h, psi])
        classes.append(lab["type"])
        height_px = lab["bbox"][3] - lab["bbox"][1]                                  # altezza nella foto
        ignored.append(lab["occluded"] > 1 or lab["truncated"] > 0.3 or height_px <= 25)   # oltre "Moderate"
    return np.array(boxes).reshape(-1, 7), classes, np.array(ignored, dtype=bool)
