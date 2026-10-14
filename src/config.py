from pathlib import Path

N_frames = 7480
N_R, N_THETA, R_MIN, R_MAX = 200, 1000, 2.0, 64.0
DATA = Path(__file__).resolve().parent.parent / "data/kitti/training"
CLASSES = ["Car", "Pedestrian", "Cyclist"]
CAR_L, CAR_W = 3.9, 1.6                                     # misure medie di un'auto in KITTI
IMG_W = 1242                                                # larghezza della foto (varia di pochi pixel fra frame)

IOU_THRESH = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}   # soglie ufficiali KITTI
NEIGHBOR = {"Car": "Van", "Pedestrian": "Person_sitting", "Cyclist": None}   # classi "vicine": né premio né errore
