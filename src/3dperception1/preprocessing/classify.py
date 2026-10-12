"""Step 5: classificazione rule-based Car / Pedestrian / Cyclist.

Volutamente stupida (v1): sole soglie su (l, w, h) del Box3D, una tabella
di 3 righe. Il punto non è indovinare le soglie a sentimento: è guardare la
confusion matrix (step 6) e correggere la tabella in base a dove sbaglia.

Una box che non rientra in nessuna riga -> "DontCare": la detection viene
scartata (infrastrutture, muri, rumore). È anche il principale filtro di FP.

z/h corrette rispetto al piano del suolo: il minimo dei punti superstiti
NON è il basso dell'oggetto (i punti sotto la soglia RANSAC sono stati
tolti), quindi riappoggiamo la box sul piano e ricalcoliamo h. Senza la
correzione h risulta sottostimato di ~0.2-0.3 m e l'intervallo z della box
non coincide con quello del GT.

Contiene:
- geometria 2D di box orientate (vertici, area, clip Sutherland-Hodgman)
- iou_bev, nms
- sopra_piano, classify, e il driver detect() che gira gli step 1-5 su un frame
"""

import numpy as np

from io_velodyne import load_velodyne
from road_removal import fit_ground
from bev_grid import build_grid, find_clusters
from find_boxes import Box3D, box_da_punti, indici_cella

CLASSI = ["Car", "Pedestrian", "Cyclist"]


# ------------------------------------------------------------------- geometria 2D

def vertici_box(x, y, l, w, yaw):
    """I 4 vertici (4, 2) della box in (x, y).
    U = asse lungo (direzione yaw), V = perpendicolare, centro (x, y).
    Stessa convenzione dell'overlay dello step 4."""
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.array([x, y])
    return np.array([c + l / 2 * u + w / 2 * v,
                     c - l / 2 * u + w / 2 * v,
                     c - l / 2 * u - w / 2 * v,
                     c + l / 2 * u - w / 2 * v])


def area_poligono(p):
    """Formula di Gauss (shoelace): area del poligono, vertici (K, 2) in ordine perimetrale."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross2(a, b):
    """'Prodotto vettoriale' 2D: la componente z di a x b (a, b con 2 componenti)."""
    return a[0] * b[1] - a[1] * b[0]


def clip_convessi(a, b):
    """Intersezione a ∩ b di due poligoni convessi (Sutherland-Hodgman).

    Si clippa `a` con i semipiani di `b`, uno per volta: per ogni spigolo
    A->B di b si tiene la parte del lato p->q che sta dal lato INTERNO
    dell'spigolo. Interno = il lato dove sta il centroide di b (per un
    poligono convesso è lo stesso lato per tutti gli spigoli; il segno
    dipende solo dal verso di percorrenza, che viene rilevato e non è
    rilevante).

    Ritorna i vertici (K, 2) dell'intersezione, in ordine perimetrale,
    oppure un array vuoto se i due poligoni non si toccano.
    """
    s = 1.0 if _cross2(b[1] - b[0], b.mean(axis=0) - b[0]) >= 0 else -1.0
    out = a
    for i in range(len(b)):
        A, B = b[i], b[(i + 1) % len(b)]
        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            fp = s * _cross2(B - A, p - A)
            fq = s * _cross2(B - A, q - A)
            if fp >= 0:                       # p dentro
                new.append(p)
                if fq < 0:                    # il lato p->q esce: aggiungo l'incrocio
                    new.append(p + (fp / (fp - fq)) * (q - p))
            elif fq >= 0:                     # il lato p->q entra: aggiungo l'incrocio
                new.append(p + (fp / (fp - fq)) * (q - p))
        out = np.array(new) if new else np.empty((0, 2))
        if out.shape[0] == 0:
            return out
    return out


def iou_bev(a, b):
    """IoU 2D (BEV) di due box (x, y, l, w, yaw). 0 se non si intersecano."""
    va = vertici_box(a.x, a.y, a.l, a.w, a.yaw)
    vb = vertici_box(b.x, b.y, b.l, b.w, b.yaw)
    inter = clip_convessi(va, vb)
    if len(inter) < 3:
        return 0.0
    ai = area_poligono(inter)
    au = area_poligono(va) + area_poligono(vb) - ai
    return float(ai / au) if au > 0 else 0.0


def nms(box, iou_thresh=0.1):
    """NMS 2D (IoU BEV), stile KITTI.

    Tutti i score sono 1.0, quindi per l'ordine uso l'area (le piu' grandi
    prima): una box viene scartata se il suo IoU con una box già tenuta
    supera la soglia. Serve sui cluster che si sovrappongono parzialmente
    (es. due auto a contatto che hanno fatto due cluster vicini).
    """
    box = sorted(box, key=lambda b: b.l * b.w, reverse=True)
    tenute = []
    for b in box:
        if all(iou_bev(b, t) < iou_thresh for t in tenute):
            tenute.append(b)
    return tenute


# ------------------------------------------------------------------- regole

def classify(box, n_points):
    """Tabella di regole (v1, volutamente stupida).

    L'ordine delle righe conta:
      Car        grande e alto:  l >= 3, w 1.4-3, altezza nella fascia auto
      Cyclist    umano allungato: l 1.2-3 (la bici), w 0.4-1.2, alto
      Pedestrian umano compatto: l, w 0.35-1.2, alto

    I bound INFERIORI non sono estetici: escludono i "bastoncini" (pali,
    bordi di muri visti di profilo, min(l,w) < 0.3-0.4 m) che senza questo
    filtro formano la meta' dei falsi positivi. Misurato su 20 frame:
    i pedoni veri sono sempre >= 0.39 m in entrambe le dimensioni, mentre
    142/314 FP pedona hanno min(l,w) < 0.3.
    Nessuna riga -> "DontCare": la detection viene scartata.
    """
    l, w, h = box.l, box.w, box.h
    if l >= 3.0 and 1.4 <= w <= 3.0 and 1.2 <= h <= 2.6:
        return "Car"
    if 1.2 <= l < 3.0 and 0.4 <= w < 1.2 and 1.3 <= h <= 2.6:
        return "Cyclist"
    if 0.35 <= l < 1.2 and 0.35 <= w < 1.2 and 1.2 <= h <= 2.6:
        return "Pedestrian"
    return "DontCare"


# ------------------------------------------------------------------- driver

def sopra_piano(b, n, d):
    """Riappoggia la box sul piano del suolo (n, d) e ricalcola h.

    Il piano ha equazione n . p + d = 0 con n[2] > 0 (normale verso l'alto,
    imposto da fit_ground). La quota del piano nel punto (b.x, b.y) è
    z_p = -(n_x b.x + n_y b.y + d) / n_z.

    Il minimo dei punti NON è il basso dell'oggetto: i punti a meno di
    0.25 m dal piano sono stati rimossi da RANSAC. Percio' il basso vero
    della box è il piano (valutato nella posizione della box) e l'altezza
    è top - z_p.
    """
    z_p = float(-(n[0] * b.x + n[1] * b.y + d) / n[2])
    top = b.z + b.h
    h = max(0.0, top - z_p)
    return Box3D(b.x, b.y, z_p, b.l, b.w, h, b.yaw, label=None, score=None)


def _etichetta_tutti(frame_id, n_r=120, n_theta=1000, r_max=64, min_points=5):
    """Step 1-4 + classificazione, su TUTTI i cluster (anche DontCare, senza NMS).
    Ritorna lista di (Box3D con label, n_points).
    z e h sono corrette rispetto al piano del suolo (vedi sopra_piano)."""
    pts = load_velodyne(frame_id)[:, :3]
    n, d, inlier = fit_ground(pts)
    sopra = pts[~inlier]
    griglia = build_grid(sopra, n_r, n_theta, r_max)
    maschere = find_clusters(griglia, min_points)
    i_r_pts, i_t_pts = indici_cella(sopra)

    out = []
    for m in maschere:
        sel = m[i_r_pts, i_t_pts]             # stessa selezione di find_boxes
        P = sopra[sel]
        b = box_da_punti(P)
        if b is None:
            continue
        b = sopra_piano(b, n, d)
        lab = classify(b, len(P))
        out.append((Box3D(*b[:7], label=lab, score=1.0), len(P)))
    return out


def detect(frame_id, n_r=120, n_theta=1000, r_max=64, min_points=5):
    """Step 1-5 completi: .bin -> box con label, togliendo DontCare e applicando NMS.
    Ritorna lista di (Box3D, n_points)."""
    tutti = _etichetta_tutti(frame_id, n_r, n_theta, r_max, min_points)
    tenute = [(b, n) for b, n in tutti if b.label != "DontCare"]
    box_nms = nms([b for b, _ in tenute], iou_thresh=0.1)
    cont = {id(b): n for b, n in tenute}      # rimappo i n_points dopo l'NMS
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
