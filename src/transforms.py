import numpy as np


def lidar_to_rectcam(calib, points):
    # Matrice 3x4 con prime 3 colonne matrice di Rotazione R della camera rispetto al LiDAR e ultima colonna spostamento t
    tr_velo_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
    # Rettifica camera vale per tutte e 4 perchè i dati delle P1-2-3 sono già traslate in P0 a cui questo si riferisce
    R0_rect = calib["R0_rect"].reshape(3, 3) 
    R, t = tr_velo_cam[:, :3], tr_velo_cam[:, 3]
    # Qui sotto traspongo perchè i miei punti sono righe in points e non colonne
    camera_raw = (points @ R.T + t) # Ruoto e sposto dal sistema di riferimento del LiDAR alla camera raw
    camera_rect = camera_raw @ R0_rect.T # Rettifico rispetto alla camera 0 per confrontare i dati
    return camera_rect

def rectcam_to_lidar(calib, points):
    # Matrice 3x4 con prime 3 colonne matrice di Rotazione R della camera rispetto al LiDAR e ultima colonna spostamento t
    tr_velo_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
    # Rettifica camera vale per tutte e 4 perchè i dati delle P1-2-3 sono già traslate in P0 a cui questo si riferisce
    R0_rect = calib["R0_rect"].reshape(3, 3) 
    R, t = tr_velo_cam[:, :3], tr_velo_cam[:, 3]
    camera_raw = points @ R0_rect
    lidar = (camera_raw - t) @ R
    return lidar
