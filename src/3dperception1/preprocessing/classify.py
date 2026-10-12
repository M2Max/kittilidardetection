import cv2
import numpy as np

from io_velodyne import load_velodyne
from road_removal import fit_ground
from bev_grid import build_grid, find_clusters
from find_boxes import Box3D, box_da_punti, indici_cella

CLASSI = ["Car", "Pedestrian", "Cyclist"]


def vertici_box(x, y, l, w, yaw):
    return cv2.boxPoints(((x, y), (l, w), float(np.degrees(yaw))))


def area_poligono(p):
    return cv2.contourArea(p.astype(np.float32).reshape(-1, 1, 2))


def clip_convessi(a, b):
    area, out = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    if area <= 0.0:
        return np.empty((0, 2))
    return out.reshape(-1, 2)


def iou_bev(a, b):
    va = vertici_box(a.x, a.y, a.l, a.w, a.yaw)
    vb = vertici_box(b.x, b.y, b.l, b.w, b.yaw)
    inter = clip_convessi(va, vb)
    if len(inter) < 3:
        return 0.0
    ai = area_poligono(inter)
    au = area_poligono(va) + area_poligono(vb) - ai
    return float(ai / au) if au > 0 else 0.0


def nms(box, iou_thresh=0.1):
    box = sorted(box, key=lambda b: b.l * b.w, reverse=True)
    tenute = []
    for b in box:
        if all(iou_bev(b, t) < iou_thresh for t in tenute):
            tenute.append(b)
    return tenute


def classify(box, n_points):
    l, w, h = box.l, box.w, box.h
    if l >= 3.0 and 1.4 <= w <= 3.0 and 1.2 <= h <= 2.6:
        return "Car"
    if 1.2 <= l < 3.0 and 0.4 <= w < 1.2 and 1.3 <= h <= 2.6:
        return "Cyclist"
    if 0.35 <= l < 1.2 and 0.35 <= w < 1.2 and 1.2 <= h <= 2.6:
        return "Pedestrian"
    return "DontCare"


def sopra_piano(b, n, d):
    z_p = float(-(n[0] * b.x + n[1] * b.y + d) / n[2])
    top = b.z + b.h
    h = max(0.0, top - z_p)
    return Box3D(b.x, b.y, z_p, b.l, b.w, h, b.yaw, label=None, score=None)


def _etichetta_tutti(frame_id, n_r=120, n_theta=1000, r_max=64, min_points=5):
    pts = load_velodyne(frame_id)[:, :3]
    n, d, inlier = fit_ground(pts)
    sopra = pts[~inlier]
    griglia = build_grid(sopra, n_r, n_theta, r_max)
    maschere = find_clusters(griglia, min_points)
    i_r_pts, i_t_pts = indici_cella(sopra)

    out = []
    for m in maschere:
        sel = m[i_r_pts, i_t_pts]
        P = sopra[sel]
        b = box_da_punti(P)
        if b is None:
            continue
        b = sopra_piano(b, n, d)
        lab = classify(b, len(P))
        out.append((Box3D(*b[:7], label=lab, score=1.0), len(P)))
    return out


def detect(frame_id, n_r=120, n_theta=1000, r_max=64, min_points=5):
    tutti = _etichetta_tutti(frame_id, n_r, n_theta, r_max, min_points)
    tenute = [(b, n) for b, n in tutti if b.label != "DontCare"]
    box_nms = nms([b for b, _ in tenute], iou_thresh=0.1)
    cont = {id(b): n for b, n in tenute}
    return [(b, cont[id(b)]) for b in box_nms]


if __name__ == "__main__":
    FRAME = 100
    tutti = _etichetta_tutti(FRAME)
    det = detect(FRAME)
    n_dontcare = sum(1 for b, _ in tutti if b.label == "DontCare")
    print(f"frame {FRAME}: {len(tutti)} box -> {n_dontcare} DontCare scartate -> {len(det)} dopo NMS")
    print("\nconteggio per classe (dopo NMS):")
    for c in CLASSI:
        print(f"  {c:<12}: {sum(1 for b, _ in det if b.label == c)}")
    print("\nle box (ordinate per numero di punti del cluster):")
    for b, n in sorted(det, key=lambda t: -t[1])[:15]:
        print(f"  {b.label:<12} centro=({b.x:6.1f}, {b.y:6.1f})  l={b.l:5.2f} w={b.w:4.2f} h={b.h:4.2f}  punti={n:4d}")
