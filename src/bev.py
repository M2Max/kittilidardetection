import numpy as np
from scipy import ndimage

from config import N_R, N_THETA, R_MAX, R_MIN


def build_grid(points, n_r=N_R, n_theta=N_THETA, r_max=R_MAX, r_min=R_MIN):
    x = points[:, 0]
    y = points[:, 1]
    r = np.hypot(x, y) # Distanza euclidea tra sensore e punto
    theta = np.arctan2(y,x) # Angolo "segno" fra verticale del sensore e punto
    grid = np.zeros((n_r, n_theta), dtype=bool) # Griglia BEV inizialmente tutta False
    i_r = np.floor(n_r * np.log(r/r_min)/np.log(r_max/r_min)).astype(int) # Array di indici distanza di caselle piene per la griglia
    i_theta = np.floor((theta + np.pi) / (2 * np.pi) * n_theta).astype(int) % n_theta # Array di indici angolo di caselle piene per la griglia normalizzato per unire punti dietro sul limite dell'angolo
    valid_mask = (r>=r_min) & (r<r_max) # Maschera che filtra i punti sotto la distanza minima e sopra la distanza massima considerate
    grid[i_r[valid_mask], i_theta[valid_mask]] = True # Applico alla griglia e coloro
    return grid, i_r, i_theta, valid_mask


def labeling(grid):
    king_adj = np.ones((3,3), dtype=bool) # [[1,1,1],[1,1,1],[1,1,1]]
    closed = ndimage.binary_closing(grid, structure=king_adj) | grid
    closed_labeled, n = ndimage.label(closed, structure=king_adj)
    clean_labeled = closed_labeled * grid
    return clean_labeled, n

def label_points(labels, i_r, i_theta):
    print(labels[i_r, i_theta])
    return labels[i_r, i_theta]
