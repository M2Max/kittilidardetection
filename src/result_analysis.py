"""Analisi dei risultati di una run: tabelle, grafici e un video "dentro al sensore".

Uso dal main, a fine run:

    from result_analysis import analyze_run, make_sensor_video
    analyze_run(records, iou_3d, load_points=load_lidar)    # tabelle + grafici in results/
    make_sensor_video(records, load_lidar)                  # video a 24 fps in results/

records e' una lista con un dizionario per frame:

    {"frame": 7,                                            # numero del frame KITTI
     "det_boxes": (N, 7) array, "det_classes": [str, ...],  # le tue scatole
     "gt_boxes":  (M, 7) array, "gt_classes":  [str, ...],  # le scatole vere (senza DontCare)
     "gt_ignored": (M,) bool array,                         # True = fuori da "Moderate"
     "time_s": 0.42}                                        # opzionale: secondi spesi sul frame

Ogni scatola e' [cx, cy, z_bottom, l, w, h, psi] in coordinate LiDAR, come in main.
"""
import csv
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                                       # niente finestre: i grafici vanno solo su file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

CLASSES = ["Car", "Pedestrian", "Cyclist"]
IOU_THRESH = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}             # soglie ufficiali KITTI
NEIGHBOR = {"Car": "Van", "Pedestrian": "Person_sitting", "Cyclist": None}  # classi vicine: ne' premio ne' errore
DIST_BINS = [(0, 10), (10, 20), (20, 40), (40, np.inf)]                     # fasce di distanza dal sensore (m)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)                      # soglie IoU per le curve
COLORS = {"Car": "#4C72B0", "Pedestrian": "#DD8452", "Cyclist": "#55A868"}  # palette adatta ai daltonici
CAMERA_HALF_FOV = np.radians(40.7)                          # meta' campo visivo della camera 2 (621 px / f 721 px)


# ================================================================ entrypoint

def analyze_run(records, iou_fn, out_dir="results", load_points=None, n_bev=6):
    """Tabelle e grafici di una run. iou_fn e' iou_3d del main (passata come parametro)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = collect(records, iou_fn)                         # tabelle di IoU, calcolate una volta sola
    summary = summary_table(data)
    counts = per_frame_counts(data)
    print_summary(summary, len(records))
    save_summary(summary, len(records), out)
    save_per_frame(records, counts, out)
    plot_scores(summary, out)
    plot_counts(summary, out)
    plot_by_distance(data, out)
    plot_threshold_curves(data, out)
    plot_best_iou(data, out)
    plot_per_frame(records, counts, out)
    for rec in records[:n_bev]:                             # i primi frame visti dall'alto
        pts = load_points(rec["frame"])[:, :3] if load_points else None
        plot_bev(rec, pts, out / "bev" / f"frame_{rec['frame']:06d}.png")
    print(f"\nRisultati salvati in {out.resolve()}")
    return summary


# ================================================================ salvare e ricaricare una run

def save_run(records, path="results/run.pkl"):
    """Salva i record di una run: grafici e video si rifanno senza rilanciare il detector."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f)                             # array numpy, liste e stringhe: pickle le salva tutte
    print(f"Run salvata in {path.resolve()} ({len(records)} frame)")

def load_run(path="results/run.pkl"):
    # Solo file salvati da te: pickle esegue cio' che trova nel file
    with open(path, "rb") as f:
        return pickle.load(f)


# ================================================================ matching (C6 + C7)

def iou_table(a_boxes, b_boxes, iou_fn):
    # Tabella: riga = scatola di a, colonna = scatola di b
    return np.array([[iou_fn(a, b) for b in b_boxes] for a in a_boxes]).reshape(len(a_boxes), len(b_boxes))

def greedy_match(ious, thresh):
    # C6: coppie dalla piu' sovrapposta in giu', ognuno abbinato al massimo una volta
    pairs, used_d, used_g = [], set(), set()
    for k in np.argsort(ious, axis=None)[::-1]:             # indici della tabella, IoU decrescente
        i, j = np.unravel_index(k, ious.shape)
        if ious[i, j] < thresh:                             # da qui in poi tutte sotto soglia
            break
        if i in used_d or j in used_g:                      # uno dei due e' gia' abbinato
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((int(i), int(j)))
    return pairs

def class_split(rec, c):
    # Le det della classe c, le GT valide di c, e le GT ignorate (c fuori Moderate, o classe vicina)
    det_sel = np.array([k == c for k in rec["det_classes"]], dtype=bool)
    is_c = np.array([k == c for k in rec["gt_classes"]], dtype=bool)
    near = np.array([k == NEIGHBOR[c] for k in rec["gt_classes"]], dtype=bool)
    ign = np.asarray(rec["gt_ignored"], dtype=bool)
    return rec["det_boxes"][det_sel], rec["gt_boxes"][is_c & ~ign], rec["gt_boxes"][(is_c & ign) | near]

def collect(records, iou_fn):
    # Per ogni classe e ogni frame: le scatole e le tabelle di IoU
    data = {c: [] for c in CLASSES}
    for rec in records:
        for c in CLASSES:
            dets, valid, ignored = class_split(rec, c)
            data[c].append({"frame": rec["frame"], "dets": dets, "valid": valid,
                            "iou_v": iou_table(dets, valid, iou_fn),        # det contro GT valide
                            "iou_i": iou_table(dets, ignored, iou_fn)})     # det contro GT ignorate
    return data

def score_frame(f, thresh):
    # TP (coppie), FP (indici det), FN (indici GT) di un frame a una certa soglia
    pairs = greedy_match(f["iou_v"], thresh)
    used_d = {i for i, _ in pairs}
    used_g = {j for _, j in pairs}
    if f["iou_i"].size:
        covers_ignored = f["iou_i"].max(axis=1) >= thresh   # C7: sopra una GT ignorata non e' un errore
    else:
        covers_ignored = np.zeros(len(f["dets"]), dtype=bool)
    fp = [i for i in range(len(f["dets"])) if i not in used_d and not covers_ignored[i]]
    fn = [j for j in range(len(f["valid"])) if j not in used_g]
    return pairs, fp, fn

def totals(frames, thresh):
    tp = fp = fn = 0
    for f in frames:
        pairs, fps, fns = score_frame(f, thresh)
        tp, fp, fn = tp + len(pairs), fp + len(fps), fn + len(fns)
    return tp, fp, fn

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# ================================================================ tabelle

def summary_table(data):
    rows = []
    for c in CLASSES:
        tp, fp, fn = totals(data[c], IOU_THRESH[c])
        p, r, f1 = prf(tp, fp, fn)
        rows.append({"class": c, "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "F1": f1})
    return rows

def per_frame_counts(data):
    # Per classe: una riga per frame con (TP, FP, FN)
    out = {}
    for c in CLASSES:
        rows = []
        for f in data[c]:
            pairs, fp, fn = score_frame(f, IOU_THRESH[c])
            rows.append((len(pairs), len(fp), len(fn)))
        out[c] = np.array(rows, dtype=int).reshape(-1, 3)
    return out

def summary_lines(summary, n_frames):
    lines = [f"Run su {n_frames} frame - IoU 3D, difficolta' Moderate",
             "", "| classe | TP | FP | FN | precision | recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['class']} | {r['TP']} | {r['FP']} | {r['FN']} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['F1']:.3f} |")
    return lines

def print_summary(summary, n_frames):
    print("\n" + "\n".join(summary_lines(summary, n_frames)))

def save_summary(summary, n_frames, out):
    (out / "summary.md").write_text("\n".join(summary_lines(summary, n_frames)) + "\n")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

def save_per_frame(records, counts, out):
    fields = ["frame", "time_s", "n_det", "n_gt_valid"] + [f"{c}_{k}" for c in CLASSES for k in ("TP", "FP", "FN")]
    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx, rec in enumerate(records):
            row = {"frame": rec["frame"], "time_s": rec.get("time_s", ""), "n_det": len(rec["det_boxes"]),
                   "n_gt_valid": n_valid_gt(rec)}
            for c in CLASSES:
                row[f"{c}_TP"], row[f"{c}_FP"], row[f"{c}_FN"] = counts[c][idx]
            w.writerow(row)

def n_valid_gt(rec):
    return sum(1 for k, ign in zip(rec["gt_classes"], rec["gt_ignored"]) if k in CLASSES and not ign)


# ================================================================ grafici

def setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:                                         # matplotlib vecchio: lo stile ha un altro nome
        plt.style.use("ggplot")
    plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold", "axes.grid.axis": "y",
                         "axes.spines.top": False, "axes.spines.right": False})

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)                                          # libera la memoria: con tanti grafici serve

def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(b.get_height()),
                ha="center", va="bottom", fontsize=9)

def plot_scores(summary, out):
    # Precision, recall e F1 affiancate per ogni classe
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES))
    for k, (key, hatch) in enumerate([("precision", ""), ("recall", "//"), ("F1", "..")]):
        vals = [r[key] for r in summary]
        bars = ax.bar(x + (k - 1) * 0.27, vals, width=0.27, label=key, hatch=hatch,
                      color=["#4C72B0", "#DD8452", "#8172B3"][k], edgecolor="white")
        bar_labels(ax, bars, lambda v: f"{v:.2f}")
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("valore (0-1)")
    ax.set_title("Precision, recall e F1 per classe (IoU 3D, Moderate)")
    ax.legend(loc="upper right")
    save_fig(fig, out / "scores_by_class.png")

def plot_counts(summary, out):
    # TP, FP, FN affiancati: quante giuste, quante inventate, quante perse
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES))
    for k, (key, color, hatch) in enumerate([("TP", "#55A868", ""), ("FP", "#DD8452", "//"), ("FN", "#4C72B0", "..")]):
        bars = ax.bar(x + (k - 1) * 0.27, [r[key] for r in summary], width=0.27, label=key,
                      color=color, hatch=hatch, edgecolor="white")
        bar_labels(ax, bars, lambda v: f"{int(v)}")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("numero di scatole")
    ax.set_title("Giuste (TP), inventate (FP) e perse (FN) per classe")
    ax.legend(loc="upper right")
    save_fig(fig, out / "tp_fp_fn_by_class.png")

def bin_label(b):
    return f"{b[0]}-{b[1]} m" if np.isfinite(b[1]) else f"{b[0]}+ m"

def dist_bin(box):
    r = np.hypot(box[0], box[1])
    return next(b for b in DIST_BINS if b[0] <= r < b[1])

def plot_by_distance(data, out):
    # Recall e precision per fascia di distanza: le GT decidono la fascia di TP e FN, le det quella degli FP
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(14, 4.5), sharey=True)
    x = np.arange(len(DIST_BINS))
    for ax, c in zip(axes, CLASSES):
        cnt = {b: [0, 0, 0] for b in DIST_BINS}             # TP, FP, FN per fascia
        for f in data[c]:
            pairs, fps, fns = score_frame(f, IOU_THRESH[c])
            for _, j in pairs:
                cnt[dist_bin(f["valid"][j])][0] += 1
            for i in fps:
                cnt[dist_bin(f["dets"][i])][1] += 1
            for j in fns:
                cnt[dist_bin(f["valid"][j])][2] += 1
        rec_ = [prf(*cnt[b])[1] for b in DIST_BINS]
        prec = [prf(*cnt[b])[0] for b in DIST_BINS]
        ax.bar(x - 0.2, rec_, width=0.4, label="recall", color=COLORS[c], edgecolor="white")
        ax.bar(x + 0.2, prec, width=0.4, label="precision", color=COLORS[c], alpha=0.45, hatch="//", edgecolor="white")
        ax.set_xticks(x, [f"{bin_label(b)}\n{cnt[b][0] + cnt[b][2]} GT" for b in DIST_BINS])   # quante GT per fascia
        ax.set_ylim(0, 1.15)
        ax.set_title(c)
        ax.legend(loc="upper center", ncol=2, fontsize=8)
    axes[0].set_ylabel("valore (0-1)")
    fig.suptitle("Come cambiano recall e precision con la distanza dal sensore", fontweight="bold")
    save_fig(fig, out / "by_distance.png")

def plot_threshold_curves(data, out):
    # Recall e precision al variare della soglia IoU: quanto ci si perde chiedendo scatole piu' precise
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for c, ls, mk in zip(CLASSES, ["-", "--", ":"], ["o", "s", "^"]):
        res = [prf(*totals(data[c], t)) for t in THRESHOLDS]
        ax_r.plot(THRESHOLDS, [r[1] for r in res], ls=ls, marker=mk, ms=4, color=COLORS[c], label=c)
        ax_p.plot(THRESHOLDS, [r[0] for r in res], ls=ls, marker=mk, ms=4, color=COLORS[c], label=c)
    for ax in (ax_r, ax_p):                                 # soglie ufficiali: 0.5 pedoni e ciclisti, 0.7 auto
        ax.axvline(0.5, color="#555555", ls="--", lw=1)
        ax.axvline(0.7, color="#555555", ls="--", lw=1)
        ax.text(0.49, 1.0, "Ped/Cyc", ha="right", va="top", fontsize=8, color="#555555")
        ax.text(0.71, 1.0, "Car", ha="left", va="top", fontsize=8, color="#555555")
    for ax, name in ((ax_r, "Recall"), (ax_p, "Precision")):
        ax.set_xlabel("soglia IoU 3D")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name} al variare della soglia")
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CLASSES), bbox_to_anchor=(0.5, -0.06))
    ax_r.set_ylabel("valore (0-1)")
    save_fig(fig, out / "iou_threshold_curves.png")

def best_iou_per_gt(frames):
    # Per ogni GT valida: l'IoU della det piu' sovrapposta (0 se non c'e' nessuna det)
    vals = [f["iou_v"].max(axis=0) if len(f["dets"]) else np.zeros(len(f["valid"])) for f in frames]
    return np.concatenate(vals) if vals else np.zeros(0)

def plot_best_iou(data, out):
    # Istogramma: quanto si avvicina la scatola migliore a ogni oggetto vero
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(14, 4.5))
    for ax, c in zip(axes, CLASSES):
        vals = best_iou_per_gt(data[c])
        if len(vals) == 0:
            ax.text(0.5, 0.5, "nessuna GT valida", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(c)
        else:
            ax.hist(vals, bins=np.linspace(0, 1, 21), color=COLORS[c], edgecolor="white")
            ax.axvline(IOU_THRESH[c], color="#333333", ls="--", lw=1.2, label=f"soglia {IOU_THRESH[c]}")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{c}: {len(vals)} GT, {np.sum(vals == 0)} senza det sopra", fontsize=11)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))     # si contano oggetti: niente 2.5
        ax.set_xlim(0, 1)
        ax.set_xlabel("IoU 3D della det migliore")
    axes[0].set_ylabel("numero di GT")
    fig.suptitle("Quanto e' vicina la scatola migliore a ogni oggetto vero", fontweight="bold")
    save_fig(fig, out / "best_iou_per_gt.png")

def plot_per_frame(records, counts, out):
    # Frame per frame: quante det, quante GT valide, quante giuste; sotto, il tempo di calcolo
    idx = np.arange(len(records))
    has_time = all("time_s" in r for r in records)
    fig, axes = plt.subplots(2 if has_time else 1, 1, figsize=(11, 6.5 if has_time else 4),
                             sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(idx, [len(r["det_boxes"]) for r in records], color="#DD8452", lw=1.5, label="rilevate")
    ax.plot(idx, [n_valid_gt(r) for r in records], color="#4C72B0", lw=1.5, ls="--", label="vere (valide)")
    ax.plot(idx, sum(counts[c][:, 0] for c in CLASSES), color="#55A868", lw=1.5, ls=":", label="giuste (TP)")
    ax.set_ylabel("scatole nel frame")
    ax.set_title("Scatole frame per frame")
    ax.legend(loc="upper right")
    if has_time:
        t = np.array([r["time_s"] for r in records])
        axes[1, 0].plot(idx, t, color="#8172B3", lw=1.2)
        axes[1, 0].axhline(t.mean(), color="#333333", ls="--", lw=1, label=f"media {t.mean():.2f} s")
        axes[1, 0].set_ylabel("secondi")
        axes[1, 0].set_title("Tempo di calcolo per frame")
        axes[1, 0].legend(loc="upper right")
    step = max(1, len(records) // 12)                       # al massimo una dozzina di etichette
    axes[-1, 0].set_xticks(idx[::step], [f"{records[k]['frame']}" for k in idx[::step]])
    axes[-1, 0].set_xlabel("frame")
    save_fig(fig, out / "per_frame.png")


# ================================================================ vista dall'alto (BEV)

def box_corners_bev(box):
    cx, cy, _, l, w, _, psi = box
    u = np.array([np.cos(psi), np.sin(psi)])
    v = np.array([-np.sin(psi), np.cos(psi)])
    c = np.array([cx, cy])
    return np.array([c + l/2*u + w/2*v, c + l/2*u - w/2*v, c - l/2*u - w/2*v, c - l/2*u + w/2*v])

def draw_bev_box(ax, box, **style):
    # Nel grafico: avanti in alto, sinistra a sinistra (x grafico = -y LiDAR, y grafico = x LiDAR)
    c = box_corners_bev(box)
    ring = np.vstack([c, c[:1]])                            # richiudo il rettangolo
    ax.plot(-ring[:, 1], ring[:, 0], **style)
    front = (c[0] + c[1]) / 2                               # meta' del muso: segno la direzione
    ax.plot([-box[1], -front[1]], [box[0], front[0]], **style)

def plot_bev(rec, points, path, x_range=(-5, 70), half_width=35):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    if points is not None:
        keep = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & (np.abs(points[:, 1]) < half_width)
        pts = points[keep][:: max(1, int(keep.sum()) // 40000)]   # al massimo ~40k punti, per velocita'
        ax.scatter(-pts[:, 1], pts[:, 0], s=0.2, c="#9A9A9A", linewidths=0)
    for s in (1, -1):                                       # bordi del campo visivo della camera
        ax.plot([0, -s * x_range[1] * np.tan(CAMERA_HALF_FOV)], [0, x_range[1]], color="#BBBBBB", lw=0.8, ls="-.")
    for box, c, ign in zip(rec["gt_boxes"], rec["gt_classes"], rec["gt_ignored"]):
        ls = ":" if ign or c not in CLASSES else "-"
        draw_bev_box(ax, box, color=COLORS.get(c, "#7F7F7F"), lw=2.0, ls=ls)
    for box, c in zip(rec["det_boxes"], rec["det_classes"]):
        draw_bev_box(ax, box, color=COLORS.get(c, "#7F7F7F"), lw=1.4, ls="--")
    ax.plot(0, 0, marker="^", color="black", ms=8)          # il sensore
    legend = [Line2D([], [], color="#333333", ls="-", lw=2, label="vera"),
              Line2D([], [], color="#333333", ls="--", lw=1.4, label="rilevata"),
              Line2D([], [], color="#333333", ls=":", lw=2, label="vera ignorata / altra classe"),
              Line2D([], [], color="#BBBBBB", ls="-.", label="campo della camera")]
    legend += [Line2D([], [], color=COLORS[c], lw=3, label=c) for c in CLASSES]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(*x_range)
    ax.set_aspect("equal")
    ax.grid(True, axis="both", alpha=0.5)                  # qui serve anche la griglia verticale
    ax.set_xlabel("sinistra  <-  laterale (m)  ->  destra")
    ax.set_ylabel("avanti (m)")
    ax.set_title(f"Frame {rec['frame']:06d} visto dall'alto")
    save_fig(fig, path)


# ================================================================ video "dentro al sensore"

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

def make_sensor_video(records, load_points, out_path="results/sensor_view.mp4", fps=1,
                      size=(1280, 720), hfov_deg=90.0, max_range=70.0, show_boxes=True):
    """Video in prima persona: gli occhi stanno nel sensore e guardano avanti (asse x del LiDAR).

    records puo' essere la lista dei record della run (disegna anche le scatole)
    oppure una semplice lista di numeri di frame (solo i punti).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = size
    lut = depth_lut()
    writer = open_writer(out_path, fps, size)
    for r in records:
        rec = r if isinstance(r, dict) else {"frame": int(r)}
        xyz = load_points(rec["frame"])[:, :3]
        img = render_points(xyz, W, H, hfov_deg, max_range, lut)
        if show_boxes:
            for box in rec.get("gt_boxes", []):
                draw_box(img, box, (255, 255, 255), 1, W, H, hfov_deg)       # vere: bianche, sottili
            for box, c in zip(rec.get("det_boxes", []), rec.get("det_classes", [])):
                draw_box(img, box, hex_to_bgr(COLORS.get(c, "#BBBBBB")), 2, W, H, hfov_deg)   # tue: spesse
        draw_hud(img, rec, len(xyz))
        writer.write(img)
    writer.release()
    print(f"Video salvato in {out_path.resolve()} ({len(records)} frame a {fps} fps)")

def project_to_eye(xyz, W, H, hfov_deg):
    # Pinhole come nella Parte A, con gli assi del LiDAR: avanti = x, destra = -y, giu' = -z
    f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)          # focale in pixel, dal campo visivo voluto
    depth = xyz[:, 0]
    ok = depth > 0.5                                        # solo cio' che sta davanti agli occhi
    u = W / 2 + f * (-xyz[ok, 1]) / depth[ok]
    v = H / 2 + f * (-xyz[ok, 2]) / depth[ok]
    return u, v, depth[ok], ok

def depth_lut():
    # 256 colori BGR: vicino = caldo (rosso), lontano = freddo (blu)
    try:
        cmap = matplotlib.colormaps["turbo"]
    except AttributeError:                                  # matplotlib < 3.5
        cmap = plt.get_cmap("turbo")
    rgb = (cmap(np.linspace(1, 0, 256))[:, :3] * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()                              # OpenCV vuole BGR, non RGB

def render_points(xyz, W, H, hfov_deg, max_range, lut):
    img = np.zeros((H, W, 3), dtype=np.uint8)               # cielo nero
    u, v, depth, _ = project_to_eye(xyz, W, H, hfov_deg)
    ui, vi = u.astype(int), v.astype(int)
    keep = (ui >= 0) & (ui < W - 1) & (vi >= 0) & (vi < H - 1) & (depth < max_range)
    ui, vi, depth = ui[keep], vi[keep], depth[keep]
    order = np.argsort(-depth)                              # prima i lontani, poi i vicini disegnati sopra
    ui, vi, depth = ui[order], vi[order], depth[order]
    col = lut[np.clip((depth / max_range * 255).astype(int), 0, 255)]   # colore dalla distanza
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):         # ogni punto diventa 2 x 2 pixel
        img[vi + dv, ui + du] = col
    return img

def box_corners_3d(box):
    bev = box_corners_bev(box)
    z0, z1 = box[2], box[2] + box[5]
    return np.vstack([np.c_[bev, np.full(4, z0)], np.c_[bev, np.full(4, z1)]])   # 4 sotto, poi 4 sopra

def draw_box(img, box, color, thickness, W, H, hfov_deg):
    u, v, _, ok = project_to_eye(box_corners_3d(box), W, H, hfov_deg)
    if not ok.all():                                        # qualche spigolo e' dietro agli occhi: salto
        return
    pts = np.stack([u, v], axis=1).round().astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])),
                 color, thickness, cv2.LINE_AA)

def hex_to_bgr(hexstr):
    h = hexstr.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)

def draw_hud(img, rec, n_points):
    # Solo caratteri ASCII: i font di OpenCV non hanno le lettere accentate
    text = f"frame {rec['frame']:06d}   punti {n_points}"
    if "gt_boxes" in rec:
        text += f"   GT {len(rec['gt_boxes'])}   det {len(rec['det_boxes'])}"
        cv2.putText(img, "bianco = vere   colore = tue (blu auto, arancio pedoni, verde ciclisti)",
                    (16, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

def open_writer(path, fps, size):
    for code in ("avc1", "mp4v"):                           # H.264 se disponibile, altrimenti MPEG-4
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError("Impossibile aprire il VideoWriter: prova out_path con estensione .avi")