"""Step 4: dai cluster alle box 3D.

Per ogni maschera di cluster (uscita di find_clusters) costruisco un Box3D:
    A. selezione dei punti originali del cluster (maschera + indici di cella)
    B. rettangolo di area minima della proiezione (x, y) con cv2.minAreaRect
       (convesso e rotating calipers calcolati internamente da OpenCV)
    C. statistiche di z -> bottom-center e altezza

Convenzioni (README): bottom-center, (l, w, h) in metri, yaw in radianti
(0 = asse lungo allineato a +x LiDAR). label lo riempie lo step 5.
"""

from pathlib import Path
import cv2
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from io_velodyne import load_velodyne
from road_removal import fit_ground
from bev_grid import build_grid, find_clusters

# Convenzione del README: bottom-center, (l, w, h) in metri, yaw in radianti.
Box3D = namedtuple("Box3D", ["x", "y", "z", "l", "w", "h", "yaw", "label", "score"])


def indici_cella(punti, n_r=120, n_theta=1000, r_max=64):
    """Indice di cella (i_r, i_t) di ogni punto.

    Deve essere la STESSA formula di build_grid: qui uso gli indici per
    LEGGERE dalla maschera quello che build_grid ha SCRITTO nella griglia.
    Se le due formule divergono, leggo una griglia diversa da quella dipinta.
    """
    x = punti[:, 0]
    y = punti[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_norm = (theta + np.pi) / (2 * np.pi)

    i_r = np.clip((r / r_max * n_r).astype(int), 0, n_r - 1)
    i_t = np.clip((theta_norm * n_theta).astype(int), 0, n_theta - 1)
    return i_r, i_t


def _rettangolo_min_area(xy):
    """Rettangolo di area minima che contiene i punti 2D (K, 2).

    Implementazione di produzione: cv2.minAreaRect gira il rotating calipers
    sul convesso (il convesso lo calcola lui internamente, quindi passo i
    punti grezzi).

    NON uso l'angolo che minAreaRect restituisce: la sua convenzione è
    cambiata tra versioni di OpenCV (intervallo e lato a cui si riferisce).
    Per non dipendere dalla versione ricavo tutto dai 4 vertici con
    cv2.boxPoints.

    Ritorna (cx, cy, l, w, yaw) o None se i punti sono degeneri
    (< 3 punti o tutti allineati: non esiste un'area).
    """
    if len(xy) < 3:
        return None
    rect = cv2.minAreaRect(np.ascontiguousarray(xy, dtype=np.float32))
    vertici = cv2.boxPoints(rect)                  # (4, 2): i 4 angoli del rettangolo
    e0 = vertici[1] - vertici[0]; l0 = np.linalg.norm(e0)
    e1 = vertici[2] - vertici[1]; l1 = np.linalg.norm(e1)
    if l0 >= l1:                                    # asse lungo sullo spigolo 0 o sull'1
        l, w, yaw = l0, l1, np.arctan2(e0[1], e0[0])
    else:
        l, w, yaw = l1, l0, np.arctan2(e1[1], e1[0])
    if min(l, w) < 1e-4:                           # tutti i punti allineati: no area
        return None
    cx, cy = rect[0]
    return float(cx), float(cy), float(l), float(w), float(yaw)


def box_da_punti(punti):
    """Da punti 3D (K, 3) al Box3D. None se il cluster è degenere."""
    geom = _rettangolo_min_area(punti[:, :2])
    if geom is None:
        return None
    cx, cy, l, w, yaw = geom
    z_bottom = float(punti[:, 2].min())            # punto più basso del cluster
    h = float(punti[:, 2].max() - z_bottom)        # (stimato leggermente in basso)
    return Box3D(x=cx, y=cy, z=z_bottom, l=l, w=w, h=h, yaw=yaw, label=None, score=1.0)


def find_boxes(punti_sopra, maschere):
    """Per ogni maschera di cluster -> Box3D (i degeneri vengono saltati)."""
    i_r_pts, i_t_pts = indici_cella(punti_sopra)   # una volta sola per tutti i punti

    box = []
    for maschera in maschere:
        # --- IL PUNTO A ---
        # maschera[i_r_pts, i_t_pts] significa letteralmente:
        #   [maschera[i_r_pts[j], i_t_pts[j]] for j in range(len(punti_sopra))]
        # cioè, per ogni punto j: "la cella di questo punto è nel cluster?"
        sel = maschera[i_r_pts, i_t_pts]           # array bool di N punti
        P = punti_sopra[sel]                       # le righe dove sel è True
        b = box_da_punti(P)
        if b is not None:
            box.append(b)
    return box


if __name__ == "__main__":
    # --- test sintetico: rettangolo 4x2 m ruotato di 30 gradi ---
    a = np.deg2rad(30)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    vertici = np.array([[-2., -1.], [2., -1.], [2., 1.], [-2., 1.]]) @ R.T
    P_syn = np.hstack([vertici, np.array([-1.7, -1.7, -0.2, -0.2]).reshape(4, 1)])
    b = box_da_punti(P_syn)
    print("test sintetico (atteso: l=4, w=2, yaw=30 o -150, centro=(0,0), z=-1.7, h=1.5):")
    print(f"  l={b.l:.3f}  w={b.w:.3f}  yaw={np.degrees(b.yaw):8.2f} deg  "
          f"centro=({b.x:.3f}, {b.y:.3f})  z={b.z:.2f}  h={b.h:.2f}")

    # --- frame 100: pipeline completa ---
    pts = load_velodyne(100)[:, :3]
    _, _, inlier = fit_ground(pts)
    sopra = pts[~inlier]
    griglia = build_grid(sopra)
    maschere = find_clusters(griglia)
    box = find_boxes(sopra, maschere)

    B = np.array([[bx.x, bx.y, bx.z, bx.l, bx.w, bx.h, bx.yaw] for bx in box])
    print(f"\ncluster: {len(maschere)}  ->  box validi: {len(box)}  "
          f"saltati (degeneri): {len(maschere) - len(box)}")
    for nome, col in [("l", 3), ("w", 4), ("h", 5)]:
        v = B[:, col]
        print(f"  {nome}: min {v.min():.2f}   mediana {np.median(v):.2f}   max {v.max():.2f}")

    auto = B[(B[:, 3] > 3) & (B[:, 4] > 1.4) & (B[:, 4] < 2.6)
             & (B[:, 5] > 1.2) & (B[:, 5] < 2.0)]
    print(f"\nbox 'da auto' (l>3, w 1.4-2.6, h 1.2-2.0): {len(auto)}")
    for r in sorted(auto, key=lambda r: -r[0]):
        print(f"  centro ({r[0]:6.1f}, {r[1]:6.1f})  l={r[3]:.1f} w={r[4]:.1f} h={r[5]:.1f}")

    # overlay: punti + rettangoli (vista dall'alto)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(sopra[:, 0], sopra[:, 1], s=0.3, c="lightgray", rasterized=True)
    for bx in box:
        u = np.array([np.cos(bx.yaw), np.sin(bx.yaw)])
        v = np.array([-np.sin(bx.yaw), np.cos(bx.yaw)])
        c = np.array([bx.x, bx.y])
        quad = np.array([c + bx.l / 2 * u + bx.w / 2 * v,
                         c - bx.l / 2 * u + bx.w / 2 * v,
                         c - bx.l / 2 * u - bx.w / 2 * v,
                         c + bx.l / 2 * u - bx.w / 2 * v])
        ax.add_patch(Polygon(quad, fill=False, ec="red" if bx.l > 3 else "cyan", lw=1.2))
    ax.set_xlim(-5, 45)
    ax.set_ylim(20, -20)      # y invertito: +y (sinistra auto) in basso
    ax.set_aspect("equal")
    ax.set_title("box step 4 (vista dall'alto): rosso = l>3m, ciano = resto")
    plt.tight_layout()
    OUT = Path(__file__).resolve().parents[3] / "outputs" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / "step4_boxes.png", dpi=110)
    plt.close(fig)
    print(f"\noverlay salvato in {OUT / 'step4_boxes.png'}")
