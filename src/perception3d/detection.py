import numpy as np

from bev import build_grid, label_points, labeling
from boxes import find_boxes, grow_car
from classify import classify
from config import IMG_W
from data import load_lidar
from floor import ransac_floor
from transforms import lidar_to_rectcam


def in_camera_fov(box, calib):
    center = np.array([[box[0], box[1], box[2] + box[5] / 2]])        # centro della scatola, in LiDAR
    p = lidar_to_rectcam(calib, center)[0]                  # stesso punto in camera rettificata
    if p[2] < 0.1:                                          # z <= 0: dietro la camera, niente divisione
        return False
    u, _, w = calib["P2"].reshape(3, 4) @ np.append(p, 1.0) # un punto solo: forma a colonna, niente trasposte
    return 0 <= u / w < IMG_W                               # colonna del pixel dentro la foto

def detect(i, calib):
    points = load_lidar(i)[:, :3]
    no_floor, n, d = ransac_floor(points)
    grid, i_r, i_theta, valid_mask = build_grid(no_floor)
    clean_labeled, n_clusters = labeling(grid)
    in_range = no_floor[valid_mask]
    point_labels = label_points(clean_labeled, i_r[valid_mask], i_theta[valid_mask])
    boxes = find_boxes(in_range, point_labels, n_clusters, n, d)

    det_boxes, det_classes = [], []
    for box in boxes:
        c = classify(box)
        if c is None or not in_camera_fov(box, calib):      # non è niente, o la camera non lo vede
            continue
        if c == "Car":
            box = grow_car(box)                             # la parte nascosta sta dalla parte lontana
        det_boxes.append(box)
        det_classes.append(c)
    return np.array(det_boxes).reshape(-1, 7), det_classes
