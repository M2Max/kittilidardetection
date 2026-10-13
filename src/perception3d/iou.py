import numpy as np


def box_corners_bev(box):
    cx, cy, _, l, w, _, psi = box
    u = np.array([np.cos(psi), np.sin(psi)])
    v = np.array([-np.sin(psi), np.cos(psi)])
    c = np.array([cx, cy])
    # in ordine, girando attorno al rettangolo (serve per la formula del laccio)
    return np.array([c + l/2*u + w/2*v, c + l/2*u - w/2*v, c - l/2*u - w/2*v, c - l/2*u + w/2*v])

def clip_polygon(poly, inside):
    # tiene solo la parte di poly dove inside(punto) >= 0
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]      # lato da p a q (l'ultimo si richiude sul primo)
        fp, fq = inside(p), inside(q)
        if fp >= 0:
            out.append(p)                               # p sta dalla parte giusta: lo tengo
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)                          # il lato attraversa il bordo: a che punto del lato?
            out.append(p + t * (q - p))                 # aggiungo il punto d'incrocio
    return out

def polygon_area(poly):
    # Formula del laccio: vertici in ordine attorno al poligono
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

def bev_intersection(a, b):
    # Trucco C3: giro il foglio in modo che a sia dritto e centrato nello zero
    u_a = np.array([np.cos(a[6]), np.sin(a[6])])
    v_a = np.array([-np.sin(a[6]), np.cos(a[6])])
    rel = box_corners_bev(b) - a[:2]                    # angoli di b, visti dal centro di a
    poly = list(np.stack([rel @ u_a, rel @ v_a], axis=1))   # coordinate nel foglio girato (le ombre)
    hl, hw = a[3] / 2, a[4] / 2
    # Taglio b con i 4 lati di a, ora dritti: x <= hl, x >= -hl, y <= hw, y >= -hw
    for inside in (lambda p: hl - p[0], lambda p: p[0] + hl, lambda p: hw - p[1], lambda p: p[1] + hw):
        poly = clip_polygon(poly, inside)
        if not poly:                                    # non resta niente: nessuna sovrapposizione
            return 0.0
    return polygon_area(poly)

def iou_bev(a, b):
    inter = bev_intersection(a, b)
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0

def iou_3d(a, b):
    inter = bev_intersection(a, b)
    z_ov = max(0.0, min(a[2] + a[5], b[2] + b[5]) - max(a[2], b[2]))   # sovrapposizione in altezza
    vol_inter = inter * z_ov
    union = a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - vol_inter
    return vol_inter / union if union > 0 else 0.0

def match(dets, gts, thresh):
    pairs = sorted(((iou_3d(d, g), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)), reverse=True)
    used_d, used_g = set(), set()
    for iou, i, j in pairs:                                 # dalla coppia più sovrapposta in giù
        if iou < thresh:                                    # da qui in poi sono tutte sotto soglia
            break
        if i in used_d or j in used_g:                      # uno dei due è già abbinato
            continue
        used_d.add(i)
        used_g.add(j)
    fp_idx = [i for i in range(len(dets)) if i not in used_d]
    return len(used_d), fp_idx, len(gts) - len(used_g)     # TP, indici delle det rimaste, FN
