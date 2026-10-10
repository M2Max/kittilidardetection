import random
import numpy as np
from io_velodyne import load_velodyne

def fit_ground(points):
    """A tentativi (RANSAC) cerco il piano"""
    tries = 100
    threshold_z = -1.5
    threshold = 0.25

    best_inlier = 0
    best_n, best_offset = np.array(np.empty), np.array(np.empty)
    for _ in range(tries):
        filtered_points = points[points[:, 2] < threshold_z]
        point1, point2, point3 = random.sample(range(0, len(filtered_points)), 3)
        v1 = filtered_points[point2] - filtered_points[point1] 
        v2 = filtered_points[point3] - filtered_points[point1]
        n = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        d = -n @ filtered_points[point1]

        ct = int(np.count_nonzero(np.abs(filtered_points @ n + d) < threshold))
        best_n = n if ct > best_inlier else best_n
        best_offset = d if ct > best_inlier else best_offset
        best_inlier = ct if ct > best_inlier else best_inlier
    
    # Impongo normale verso l'alto
    if best_n[2] < 0:
        best_n, best_offset = -best_n, -best_offset

    distance = np.abs(points @ best_n + best_offset)
    return best_n, best_offset, distance < threshold

pts = load_velodyne(100)[:, :3]
n, d, inlier = fit_ground(pts)

print("normale:", n)              # circa [0, 0, 1], terza componente positiva
print("offset:", d)
print("tolti:", inlier.sum())     # 55000-70000 su 123183
print("altezza sotto l'auto:", d + n @ np.array([0., 0., 0.]))   # circa -1.7


random.seed(0)
for i in range(10):
    n, d, inlier = fit_ground(pts)
    assert n[2] > 0, f"run {i}: normale capovolta"
    assert 1.5 < d < 1.9, f"run {i}: offset {d:.2f} fuori range"
    print(f"run {i}: ok, offset {d:.3f}, tolto {inlier.sum()}")
