"""Step 6: valutazione delle detection contro il GT (KITTI label_2).

- iou_3d: IoU ESATTO tra due box 3D (niente approssimazione a voxel)
- match: matching greedy, ogni det e ogni GT usati al massimo una volta
- evaluate_frames: P/R/F1 per classe + confusion matrix + breakdown per distanza

Soglie del protocollo KITTI: Car 0.70, Pedestrian 0.50, Cyclist 0.50
(NON 0.5 per tutte: è l'errore piu' comune).

Questo modulo è per lo sviluppo e il debug. Il numero finale confrontabile
con la letteratura lo dà il devkit ufficiale (devkit/cpp/evaluate_object).
"""

from pathlib import Path
import numpy as np

from io_velodyne import load_calib, load_labels, lidar_to_rect, rect_to_lidar
from classify import CLASSI, detect, area_poligono, clip_convessi, vertici_box
from find_boxes import Box3D

# Soglie del protocollo KITTI (README)
IOU_THRESH = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}

# Dimensioni dell'immagine KITTI (per il filtro FOV)
LARGHEZZA_IMG = 1242
ALTEZZA_IMG = 375


def dentro_fov(box, calib):
    """True se il CENTRO della box proietta dentro l'immagine camera.

    Approssimazione: test solo il centro (una box a cavalcamento del bordo
    conta come dentro solo se il centro lo e'). Serve per il protocollo
    'solo FOV': i label KITTI esistono solo dove vede la camera, mentre il
    LiDAR vede 360 gradi, quindi ogni det fuori FOV e' una FP scontata.
    """
    p_lid = np.array([[box.x, box.y, box.z + box.h / 2]])
    p_rect = lidar_to_rect(p_lid, calib)[0]
    if p_rect[2] < 0.1:
        return False                      # dietro la camera
    u = calib["P2"][0, 0] * p_rect[0] / p_rect[2] + calib["P2"][0, 2]
    v = calib["P2"][1, 1] * p_rect[1] / p_rect[2] + calib["P2"][1, 2]
    return 0.0 <= u < LARGHEZZA_IMG and 0.0 <= v < ALTEZZA_IMG


def iou_3d(a, b):
    """IoU esatto di due box 3D (bottom-center, l, w, h, yaw in radianti).

    Trucco: porto b nel frame locale di a. La rotazione è solo attorno a z,
    quindi z non cambia: in quel frame a è asse-allineata (l x w al centro
    dell'origine) e b è ruotata di (b.yaw - a.yaw). L'intersezione XY si
    calcola con lo stesso Sutherland-Hodgman dell'IoU 2D; l'asse z è la
    sovrapposizione di due intervalli. Volume intersezione = area_2D * z_ov.
    """
    dx, dy = b.x - a.x, b.y - a.y
    ca, sa = np.cos(a.yaw), np.sin(a.yaw)
    cx = dx * ca + dy * sa                  # centro di b nel frame di a
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
    """Matching greedy: ordino TUTTE le coppie (det, gt) per IoU 3D decrescente
    e abbino, usando ogni det e ogni gt al massimo una volta.

    Ritorna (pairs, fp, fn): pairs = [(i_det, j_gt), ...] delle coppie abbinate,
    fp = indici det non abbinate, fn = indici GT non abbinate.
    """
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
            break                            # sotto soglia: anche tutto il resto
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((i, j))
    fp = [i for i in range(len(dets)) if i not in used_d]
    fn = [j for j in range(len(gts)) if j not in used_g]
    return pairs, fp, fn


def match(dets, gts, iou_thresh):
    """Contratto: (tp, fp, fn) come liste di indici (tp = le det abbinate)."""
    pairs, fp, fn = _match(dets, gts, iou_thresh)
    return [i for i, _ in pairs], fp, fn


def gt_lidar(frame_id):
    """label_2 -> lista di Box3D nel frame LiDAR.

    Posizione: la location KITTI è bottom-center nel frame camera rettificato
    (step 1), trasformata con rect_to_lidar.
    Heading: trasformo il VETTORE (fwd_cam = [cos ry, 0, -sin ry]), non l'angolo
    con arccotangente — vedi step 1 (scartare il vettore dà yaw sbagliato di pi/2).
    """
    calib = load_calib(frame_id)
    R2L = (calib["R0_rect"].T @ calib["Tr_velo_to_cam"][:, :3]).T
    out = []
    for lab in load_labels(frame_id):
        if lab["type"] not in CLASSI:
            continue                          # DontCare e altre classi: non li matcho in questa v1
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
    for lo, hi, nome in _BINI:
        if lo <= r < hi:
            return nome
    return "40+"


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
    """Aggrega i match di un frame (det e GT gia' calcolati) nel dict res.
    Se fov_only=True, prima filtra a cio' che proietta dentro l'immagine."""
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
    """Detections vs GT su un insieme di frame.
    Ritorna un dict con TP/FP/FN per classe, confusion matrix e breakdown per distanza.

    fov_only: se True, valuta solo gli oggetti (det e GT) il cui centro proietta
    dentro l'immagine camera. E' il protocollo usato dai lavori LiDAR per
    rendere i numeri confrontabili: senza, ogni auto parcheggiata fuori
    dal campo visiva della camera e' una FP inspiegabile (i label KITTI
    vengono annotati sull'immagine, quindi non coprono mai il resto)."""
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
    # --- single frame: vedere esattamente cosa si e' abbinato a cosa ---
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

    # --- smoke test: i primi 20 frame della train split (val solo alla fine) ---
    with open(Path(__file__).parents[3] / "data/kitti/ImageSets/train.txt") as f:
        frame_ids = [int(line) for line in f if line.strip()][:20]
    print(f"\nsmoke test su frame {frame_ids}")

    # le detection si calcolano UNA volta sola e si valutano con i due
    # protocolli a parita' di det (altrimenti RANSAC stocastico le sposterebbe)
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
