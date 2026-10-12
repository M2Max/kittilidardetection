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
        
            
def load_labels(frame_id):
    """label_2/XXXXXX.txt -> lista di dict.

    15 colonne: type, truncated, occluded, alpha, bbox2d(4), h w l, x y z, rotation_y.
    Alcuni file hanno una 16a colonna (score): la ignoro.
    Le coordinate x y z sono bottom-center nel frame camera rettificato.
    Tengo TUTTE le righe, DontCare comprese (in valutazione servono: sono le
    zone dove un falso positivo non va penalizzato).
    """
    labelPath = Path(__file__).parents[3] / f"data/kitti/training/label_2/{frame_id:06d}.txt"
    labels = []
    with open(labelPath) as f:
        for line in f:
            v = line.split()
            if len(v) < 15:
                continue
            labels.append({
                "type": v[0],
                "truncated": float(v[1]),    # in KITTI e' float (0..1)
                "occluded": int(v[2]),      # 0..3
                "alpha": float(v[3]),
                "h": float(v[8]), "w": float(v[9]), "l": float(v[10]),
                "x": float(v[11]), "y": float(v[12]), "z": float(v[13]),
                "rotation_y": float(v[14]),
            })
    return labels

def lidar_to_rect(pts, calib):
    """Inverso di rect_to_lidar: punti LiDAR -> camera rettificata (convenzione riga).
    Serva per il filtro FOV nella valutazione (step 6)."""
    R0_rect = calib["R0_rect"]
    Tr_velo_to_cam = calib["Tr_velo_to_cam"]
    rotation = Tr_velo_to_cam[:, :3]
    traslation = Tr_velo_to_cam[:, 3]
    return (pts @ rotation.T + traslation) @ R0_rect.T

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