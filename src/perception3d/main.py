from config import CLASSES
from data import load_calib
import time
from detection import detect
from evaluation import evaluate_frame, print_scores
from ground_truth import gt_boxes
from result_analysis import analyze_run, make_sensor_video, CLASSES, IOU_THRESH, NEIGHBOR, load_run, save_run
from iou import iou_3d
from data import load_lidar
from pathlib import Path

RUN_FILE = Path("results/run.pkl")
RECOMPUTE = False

if __name__ == "__main__":
    records = []
    if RUN_FILE.exists() and not RECOMPUTE:
        records = load_run(RUN_FILE)                # seconds instead of the whole run
    else:
        for i in range(7480):
            calib = load_calib(i)
            t0 = time.perf_counter()
            det_boxes, det_classes = detect(i, calib)
            elapsed = time.perf_counter() - t0                  # tempo del solo detector
            gt_b, gt_c, gt_ign = gt_boxes(i, calib)
            records.append({"frame": i, "det_boxes": det_boxes, "det_classes": det_classes,
                            "gt_boxes": gt_b, "gt_classes": gt_c, "gt_ignored": gt_ign,
                            "time_s": elapsed})
        save_run(records, RUN_FILE)

    analyze_run(records, iou_3d, load_points=load_lidar)    # tabelle + grafici in results/
    make_sensor_video(records, load_lidar, fps=1)          # results/sensor_view.mp4