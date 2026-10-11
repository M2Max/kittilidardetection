from pathlib import Path
import numpy as np
from io_velodyne import load_velodyne
from road_removal import fit_ground
import matplotlib.pyplot as plt
import scipy.ndimage

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

def find_clusters(grid, min_points=5, connectivity=8):
    if connectivity != 4 and connectivity != 8:
        raise ValueError("connectivity must be 4 or 8")
    if connectivity == 8:
        struct = [[1,1,1],
                  [1,1,1],
                  [1,1,1]]
    else:
        struct = [[0,1,0],
                  [1,1,1],
                  [0,1,0]]

    etichette, n = scipy.ndimage.label(grid, structure=struct)

    # size[i] = quante celle ha la componente i; size[0] e' lo sfondo
    size = np.bincount(etichette.ravel())

    tenute = [i for i in range(1, n + 1) if size[i] >= min_points]
    return [etichette == i for i in tenute]

pts = load_velodyne(100)[:, :3]
n, d, inlier = fit_ground(pts)
griglia = build_grid(pts[~inlier])

# --- diagnostica grezza: cosa fa label PRIMA del filtro ---
etichette, n_grezze = scipy.ndimage.label(griglia, structure=np.ones((3, 3), dtype=int))
size = np.bincount(etichette.ravel())   # size[i] = celle della componente i

print(f"componenti grezze (8-connessita) : {n_grezze}")
print(f"  di 1 cella                     : {(size[1:] == 1).sum()}")
print(f"  di 2-4 celle                   : {((size[1:] >= 2) & (size[1:] <= 4)).sum()}")
print(f"  di 5+ celle                    : {(size[1:] >= 5).sum()}")

top = np.argsort(size)[::-1][:10]
print("le 10 piu' grandi:")
for i in top:
    if i > 0:
        print(f"  etichetta {i:>3}: {size[i]:>4} celle")

# --- quello che usa la pipeline ---
cluster = find_clusters(griglia)
print(f"cluster tenuti (min_points=5)    : {len(cluster)}")

# # --- plot: binario a sinistra, componenti colorate a destra ---
# fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# ax[0].imshow(griglia.T, cmap="gray", aspect="auto")
# ax[0].set_title("occupazione (binaria)")
# ax[1].imshow(etichette.T, cmap="nipy_spectral", aspect="auto")
# ax[1].set_title(f"{n_grezze} componenti (8-connessita)")
# plt.tight_layout()
# OUT = Path(__file__).resolve().parents[3] / "outputs" / "figures"
# OUT.mkdir(parents=True, exist_ok=True)
# plt.savefig(OUT / "step3_cluster.png", dpi=110)
# plt.close(fig)
# plt.show()

# print(f"shape: {griglia.shape}") # (120, 1000)
# print(f"celle occupate: {griglia.sum()}") # qualche migliaio, non 60000

# # la forma a campana e' il test piu' importante
# print("celle per riga, righe 20-28 (dove dovrebbe esserci la strada):")
# for i in range(20, 28):
#     print(f"  riga {i:>3} ({i*64/120:.0f} m): {griglia[i].sum():>4} celle")

# plt.figure(figsize=(16, 8))
# plt.imshow(griglia.T, cmap="gray", aspect="auto")
# plt.title("occupazione: bianco = punti")
# plt.show()