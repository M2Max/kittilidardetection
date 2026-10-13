from config import CLASSES, IOU_THRESH, NEIGHBOR
from iou import iou_3d, match


def evaluate_frame(det_boxes, det_classes, gt_b, gt_c, gt_ign, counts):
    for c in CLASSES:
        dets = [b for b, k in zip(det_boxes, det_classes) if k == c]
        valid = [b for b, k, ign in zip(gt_b, gt_c, gt_ign) if k == c and not ign]
        ignored = [b for b, k, ign in zip(gt_b, gt_c, gt_ign) if (k == c and ign) or k == NEIGHBOR[c]]
        tp, fp_idx, fn = match(dets, valid, IOU_THRESH[c])
        # C7: una det che copre una GT ignorata non è un errore
        fp = [i for i in fp_idx if not any(iou_3d(dets[i], g) >= IOU_THRESH[c] for g in ignored)]
        counts[c]["TP"] += tp
        counts[c]["FP"] += len(fp)
        counts[c]["FN"] += fn

def print_scores(counts):
    print(f"{'classe':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>11}{'recall':>8}{'F1':>7}")
    for c in CLASSES:
        tp, fp, fn = counts[c]["TP"], counts[c]["FP"], counts[c]["FN"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{c:<12}{tp:>6}{fp:>6}{fn:>6}{p:>11.3f}{r:>8.3f}{f1:>7.3f}")
