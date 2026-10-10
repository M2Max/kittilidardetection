import numpy as np
from io_velodyne import load_velodyne
from road_removal import fit_ground
import matplotlib.pyplot as plt

def build_grid(points, n_r=120, n_theta=1000, r_max=64):
    grid = np.zeros((n_r, n_theta), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_norm = (theta + np.pi) / (2 * np.pi)

    i_r = (r / r_max * n_r).astype(int)
    i_theta = (theta_norm * n_theta).astype(int)

    i_r = np.clip(i_r, 0, n_r - 1)
    i_theta = np.clip(i_theta, 0, n_theta - 1)  

    grid[i_r, i_theta] = True

    return grid

pts = load_velodyne(100)[:, :3]
n, d, inlier = fit_ground(pts)
griglia = build_grid(pts[~inlier])

print(f"shape: {griglia.shape}") # (120, 1000)
print(f"celle occupate: {griglia.sum()}") # qualche migliaio, non 60000

# la forma a campana e' il test piu' importante
print("celle per riga, righe 20-28 (dove dovrebbe esserci la strada):")
for i in range(20, 28):
    print(f"  riga {i:>3} ({i*64/120:.0f} m): {griglia[i].sum():>4} celle")

plt.figure(figsize=(16, 8))
plt.imshow(griglia.T, cmap="gray", aspect="auto")
plt.title("occupazione: bianco = punti")
plt.show()