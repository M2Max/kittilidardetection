import numpy as np


def plane_from_points(points):
    p1, p2, p3 = points[0], points[1], points[2]
    v1 = p2 - p1
    v2 = p3 - p1
    euclidean_norm = np.linalg.norm(np.cross(v1, v2))
    if euclidean_norm < 1e-6:
        return None
    n_normalized = np.cross(v1, v2) / euclidean_norm
    d = - n_normalized @ p1
    # Se z < 0 il verso del vettore è verso il basso quindi li nego entrambi per puntare verso l'alto
    if n_normalized[2] < 0:
        n_normalized = - n_normalized
        d = -d
    return n_normalized, d

def ransac_floor(points, tries=500, threshold=0.2):
    rng = np.random.default_rng(42)
    low = points[points[:, 2] < -1.5] 
    best_votes, best_n, best_d = 0, None, None
    for i in range(tries):
        idx = rng.choice(len(low), size=3, replace=False) 
        chosen = low[idx]
        plane = plane_from_points(chosen)
        if plane is None:                       # tre punti allineati
            continue
        n, d = plane

        # n a destra perchè i vettori punto sono "orizzontali"
        s = points @ n + d
        votes = np.count_nonzero(np.abs(s) < threshold)
        if votes > best_votes:
            best_votes, best_n, best_d = votes, n, d

    no_floor_points = points[points @ best_n + best_d >= threshold]
    return no_floor_points, best_n, best_d
