from pathlib import Path
from os import listdir
from os.path import isfile, join
import numpy as np


def load_calib(frame_id):
    """Legge calib/XXXXXX.txt -> dict di array numpy"""
    calibPath = Path(__file__).parents[3] / f"data/kitti/training/calib/{frame_id:06d}.txt"
    processed_lines = dict()
    with open(calibPath) as file:
        for line in file:
            if ":" in line:
                splitted_line = line.split(":", 1)
                processed_lines[splitted_line[0]] = np.array([float(v) for v in splitted_line[1].split()]) 

    return {
        "P2": processed_lines["P2"].reshape(3, 4), 
        "R0_rect": processed_lines["R0_rect"].reshape(3, 3), 
        "Tr_velo_to_cam": processed_lines["Tr_velo_to_cam"].reshape(3, 4)
    }
        
            
# def load_labels(frame_id):

# def lidar_to_rect(pts, calib):

def rect_to_lidar(pts, calib):
    R0_rect = calib["R0_rect"]
    Tr_velo_to_cam = calib["Tr_velo_to_cam"]
    rotation = Tr_velo_to_cam[:, :3]
    traslation = Tr_velo_to_cam[:, 3]   
    M = np.linalg.inv(R0_rect @ rotation)
    b = np.linalg.inv(rotation) @ traslation
    return pts @ np.linalg.matrix_transpose(M) - b

def load_velodyne(frame_id):
    veloPath = Path(__file__).parents[3] / f"data/kitti/training/velodyne/{frame_id:06d}.bin"
    return np.fromfile(veloPath, dtype=np.float32).reshape(-1, 4)