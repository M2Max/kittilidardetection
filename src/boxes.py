import numpy as np

from config import CAR_L, CAR_W


def box_bev(xy, step=0.5, criterion="edges"):
    best_key, best = (np.inf, np.inf), None
    for deg in np.arange(0.0, 90.0, step):              # 0..90 basta
        psi = np.radians(deg)
        u = np.array([np.cos(psi), np.sin(psi)])        # direzione "lunghezza"
        v = np.array([-np.sin(psi), np.cos(psi)])       # direzione "larghezza" (u girato di 90°)
        a, b = xy @ u, xy @ v                           # proiezione ombre vettori rispetto a u e v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if criterion == "area":
            key = (area,)
        else:
            dist = np.minimum.reduce([a - a.min(), a.max() - a, b - b.min(), b.max() - b]).mean()  # distanza media dal bordo più vicino
            key = (dist, area)                          # a parità di distanza vince l'area più piccola
        if key < best_key:
            best_key, best = key, (psi, u, v, a.min(), a.max(), b.min(), b.max())
    psi, u, v, a0, a1, b0, b1 = best
    cx, cy = (a0 + a1) / 2 * u + (b0 + b1) / 2 * v     # centro: metà delle ombre, riportata nel foglio normale
    l, w = a1 - a0, b1 - b0
    if w > l:                                           # il lato lungo è lungo v: scambio e giro di 90°
        l, w, psi = w, l, psi + np.pi / 2
    return cx, cy, l, w, psi

def box_height(pts, cx, cy, n, d):
    z_bottom = -(n[0] * cx + n[1] * cy + d) / n[2]     # altezza della strada sotto il centro (piano RANSAC)
    h = pts[:, 2].max() - z_bottom                      # dal suolo al punto più alto
    return z_bottom, h

def find_boxes(points, point_labels, n_clusters, n, d, min_points=5):
    boxes = []
    for k in range(1, n_clusters + 1):                  # le etichette vanno da 1 a n (0 = vuoto)
        pts = points[point_labels == k]                 # maschera: i punti del cluster k
        if len(pts) < min_points:                       # troppo pochi punti: rumore
            continue
        cx, cy, l, w, psi = box_bev(pts[:, :2])
        z_bottom, h = box_height(pts, cx, cy, n, d)
        boxes.append([cx, cy, z_bottom, l, w, h, psi])
    return np.array(boxes).reshape(-1, 7)               # una riga per scatola (forma (0, 7) se non ce ne sono)

def grow_car(box):
    cx, cy, z, l, w, h, psi = box
    u = np.array([np.cos(psi), np.sin(psi)])                # direzione lunghezza
    v = np.array([-np.sin(psi), np.cos(psi)])               # direzione larghezza
    c = np.array([cx, cy])
    if l < CAR_L:
        away = u if c @ u > 0 else -u                       # verso di u che si allontana dal sensore
        c = c + (CAR_L - l) / 2 * away                      # il centro si sposta di metà dell'allungamento
        l = CAR_L
    if w < CAR_W:
        away = v if c @ v > 0 else -v
        c = c + (CAR_W - w) / 2 * away
        w = CAR_W
    return np.array([c[0], c[1], z, l, w, h, psi])
