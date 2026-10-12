from pathlib import Path
import numpy as np

from io_velodyne import load_calib, load_labels, lidar_to_rect, rect_to_lidar
from classify import CLASSI, detect, area_poligono, clip_convessi, vertici_box
from find_boxes import Box3D

IOU_THRESH = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}

LARGHEZZA_IMG = 1242
ALTEZZA_IMG = 375


def dentro_fov(box, calib):
    p_lid = np.array([[box.x, box.y, box.z + box.h / 2]])
    p_rect = lidar_to_rect(p_lid, calib)[0]
    h = p_rect @ calib["P2"][:3, :3].T + calib["P2"][:3, 3]
    if h[2] < 0.1:
        return False
    u, v = h[0] / h[2], h[1] / h[2]
    return 0.0 <= u < LARGHEZZA_IMG and 0.0 <= v < ALTEZZA_IMG


def iou_3d(a, b):
    dx, dy = b.x - a.x, b.y - a.y
    ca, sa = np.cos(a.yaw), np.sin(a.yaw)
    cx = dx * ca + dy * sa
    cy = -dx * sa + dy * ca
    va = vertici_box(0., 0., a.l, a.w, 0.)
    vb = vertici_box(cx, cy, b.l, b.w, b.yaw - a.yaw)
    inter = clip_convessi(va, vb)
    area_i = area_poligono(inter) if len(inter) >= 3 else 0.0
    z_ov = max(0.0, min(a.z + a.h, b.z + b.h) - max(a.z, b.z))
    vol_i = area_i * z_ov
    vol_u = a.l * a.w * a.h + b.l * b.w * b.h - vol_i
    return float(vol_i / vol_u) if vol_u > 0 else 0.0


def _match(dets, gts, iou_thresh):
    if not dets or not gts:
        return ([], list(range(len(dets))), list(range(len(gts))))
    ious = np.zeros((len(dets), len(gts)))
    for i in range(len(dets)):
        for j in range(len(gts)):
            ious[i, j] = iou_3d(dets[i], gts[j])
    used_d, used_g = set(), set()
    pairs = []
    for k in np.argsort(ious.ravel(), kind="stable")[::-1]:
        i, j = divmod(int(k), len(gts))
        if ious[i, j] < iou_thresh:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((i, j))
    fp = [i for i in range(len(dets)) if i not in used_d]
    fn = [j for j in range(len(gts)) if j not in used_g]
    return pairs, fp, fn


def match(dets, gts, iou_thresh):
    pairs, fp, fn = _match(dets, gts, iou_thresh)
    return [i for i, _ in pairs], fp, fn


def gt_lidar(frame_id):
    calib = load_calib(frame_id)
    R2L = (calib["R0_rect"].T @ calib["Tr_velo_to_cam"][:, :3]).T
    out = []
    for lab in load_labels(frame_id):
        if lab["type"] not in CLASSI:
            continue
        p = rect_to_lidar(np.array([[lab["x"], lab["y"], lab["z"]]]), calib)[0]
        fwd_cam = np.array([np.cos(lab["rotation_y"]), 0., -np.sin(lab["rotation_y"])])
        fwd_lid = R2L @ fwd_cam
        out.append(Box3D(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                         l=lab["l"], w=lab["w"], h=lab["h"],
                         yaw=float(np.arctan2(fwd_lid[1], fwd_lid[0])),
                         label=lab["type"], score=1.0))
    return out


_BINI = [(0, 10, "0-10"), (10, 20, "10-20"), (20, 40, "20-40"), (40, float("inf"), "40+")]


def _bin_di(r):
    return _BINI[np.digitize(r, [b[1] for b in _BINI[:-1]])][2]


def _nuovo_res(n_frames, fov_only):
    return {
        "n_frames": n_frames,
        "fov_only": fov_only,
        "TP": {c: 0 for c in CLASSI},
        "FP": {c: 0 for c in CLASSI},
        "FN": {c: 0 for c in CLASSI},
        "conf": {g: {d: 0 for d in CLASSI} for g in CLASSI},
        "bin": {nome: {"TP": {c: 0 for c in CLASSI},
                       "FP": {c: 0 for c in CLASSI},
                       "FN": {c: 0 for c in CLASSI}} for _, _, nome in _BINI},
    }


def _accuma(res, dets, gts, calib, fov_only):
    if fov_only:
        dets = [b for b in dets if dentro_fov(b, calib)]
        gts = [g for g in gts if dentro_fov(g, calib)]
    for c in CLASSI:
        dc = [d for d in dets if d.label == c]
        gc = [g for g in gts if g.label == c]
        pairs, fp, fn = _match(dc, gc, IOU_THRESH[c])
        res["TP"][c] += len(pairs)
        res["FP"][c] += len(fp)
        res["FN"][c] += len(fn)
        for (i, j) in pairs:
            res["conf"][gc[j].label][dc[i].label] += 1
            res["bin"][_bin_di(float(np.hypot(gc[j].x, gc[j].y)))]["TP"][c] += 1
        for i in fp:
            res["bin"][_bin_di(float(np.hypot(dc[i].x, dc[i].y)))]["FP"][c] += 1
        for j in fn:
            res["bin"][_bin_di(float(np.hypot(gc[j].x, gc[j].y)))]["FN"][c] += 1


def evaluate_frames(frame_ids, fov_only=False):
    res = _nuovo_res(len(frame_ids), fov_only)
    for fid in frame_ids:
        det = detect(fid)
        dets = [b for b, _ in det]
        gts = gt_lidar(fid)
        calib = load_calib(fid) if fov_only else None
        _accuma(res, dets, gts, calib, fov_only)
    return res


def stampa(res):
    prot = "solo FOV camera" if res["fov_only"] else "tutto il campo LiDAR (grezzo)"
    print(f"\n=== valutazione su {res['n_frames']} frame — {prot} — soglie {IOU_THRESH} ===")
    print(f"{'classe':<12}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>12}{'recall':>9}{'F1':>7}")
    for c in CLASSI:
        tp, fp, fn = res["TP"][c], res["FP"][c], res["FN"][c]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{c:<12}{tp:>5}{fp:>5}{fn:>5}{p:>11.3f}{r:>9.3f}{f1:>7.3f}")

    print("\nconfusion matrix (riga = GT, colonna = predetto):")
    print(f"{'':>14}" + "".join(f"{c:>11}" for c in CLASSI) + f"{'FN':>6}")
    for g in CLASSI:
        print(f"{g:>14}" + "".join(f"{res['conf'][g][d]:>11d}" for d in CLASSI) + f"{res['FN'][g]:>6d}")
    print(f"{'(nessun GT)':>14}" + "".join(f"{res['FP'][d]:>11d}" for d in CLASSI))

    print("\nper distanza (P/R per classe):")
    print(f"{'bin':>7}" + "".join(f"  {c:>12}" for c in CLASSI))
    for _, _, nome in _BINI:
        riga = f"{nome:>7}"
        for c in CLASSI:
            t = res["bin"][nome]
            p = t["TP"][c] / (t["TP"][c] + t["FP"][c]) if t["TP"][c] + t["FP"][c] else float("nan")
            r = t["TP"][c] / (t["TP"][c] + t["FN"][c]) if t["TP"][c] + t["FN"][c] else float("nan")
            riga += f"  {p:5.2f}/{r:5.2f}"
        print(riga)


if __name__ == "__main__":
    FRAME = 100
    det = detect(FRAME)
    dets = [b for b, _ in det]
    gts = gt_lidar(FRAME)
    print(f"frame {FRAME}: {len(dets)} det, {len(gts)} GT")
    for c in CLASSI:
        dc = [d for d in dets if d.label == c]
        gc = [g for g in gts if g.label == c]
        pairs, fp, fn = _match(dc, gc, IOU_THRESH[c])
        for (i, j) in pairs:
            print(f"  TP {c:<11}: det({dc[i].x:6.1f}, {dc[i].y:6.1f}) ~ gt({gc[j].x:6.1f}, {gc[j].y:6.1f})  IoU3D={iou_3d(dc[i], gc[j]):.2f}")
        for j in fn:
            print(f"  FN {c:<11}: gt({gc[j].x:6.1f}, {gc[j].y:6.1f})  l={gc[j].l:.1f} w={gc[j].w:.1f} h={gc[j].h:.1f}")
        for i in fp:
            print(f"  FP {c:<11}: det({dc[i].x:6.1f}, {dc[i].y:6.1f})  l={dc[i].l:.1f} w={dc[i].w:.1f} h={dc[i].h:.1f}")

    with open(Path(__file__).parents[3] / "data/kitti/ImageSets/train.txt") as f:
        frame_ids = [int(line) for line in f if line.strip()][:20]
    print(f"\nsmoke test su frame {frame_ids}")

    res_grezzo = _nuovo_res(len(frame_ids), False)
    res_fov = _nuovo_res(len(frame_ids), True)
    for fid in frame_ids:
        det = detect(fid)
        dets = [b for b, _ in det]
        gts = gt_lidar(fid)
        calib = load_calib(fid)
        _accuma(res_grezzo, dets, gts, calib, False)
        _accuma(res_fov, dets, gts, calib, True)
    stampa(res_grezzo)
    stampa(res_fov)
