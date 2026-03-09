"""
Honeypot Scoring System
Usage: python scorer.py <tasks_dir>

Expects <tasks_dir> to contain paired zip files:
  <task_name>.zip
  gt_<task_name>.zip

Extracts them, scores the honeypot frames, prints a report, and saves JSON.
"""

import sys
import json
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1 — Zip discovery and extraction
# ---------------------------------------------------------------------------

def find_task_pairs(tasks_dir: Path) -> list[tuple[str, Path, Path]]:
    zips = {p.stem: p for p in tasks_dir.glob("*.zip")}
    pairs = []
    for stem, path in zips.items():
        if stem.startswith("gt_"):
            task_name = stem[3:]
            if task_name in zips:
                pairs.append((task_name, path, zips[task_name]))
    if not pairs:
        sys.exit(f"[error] No paired zip files found in {tasks_dir}.\n"
                 f"  Expected: <task>.zip + gt_<task>.zip")
    return pairs


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"  extracted {zip_path.name} → {dest_dir}")


# ---------------------------------------------------------------------------
# Annotation file discovery
# ---------------------------------------------------------------------------

SKIP_NAMES = {"train.txt", "obj.data", "obj.names"}


def build_filename_map(root: Path) -> dict[str, Path]:
    mapping = {}
    for p in root.rglob("*.txt"):
        if p.name in SKIP_NAMES:
            continue
        mapping[p.name] = p
    return mapping


# ---------------------------------------------------------------------------
# Parse YOLO annotation
# ---------------------------------------------------------------------------

def parse_annotation(path: Path):
    boxes = []
    text = path.read_text().strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cls, xc, yc, w, h))
    return boxes


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(b1, b2):
    _, x1, y1, w1, h1 = b1
    _, x2, y2, w2, h2 = b2

    ax1, ay1, ax2, ay2 = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
    bx1, by1, bx2, by2 = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

IOU_MATCH = 0.6
IOU_DUP = 0.8


def score_frame(gt_boxes, pred_boxes):

    candidates = []

    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            if gb[0] == pb[0]:
                v = iou(gb, pb)
                if v >= IOU_MATCH:
                    candidates.append((v, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for v, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)

    tp = len(matched_gt)
    fn = len(gt_boxes) - tp

    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    fp = len(unmatched_pred)

    unmatched_gt_idx = [i for i in range(len(gt_boxes)) if i not in matched_gt]

    class_errors = 0

    for pi in unmatched_pred:
        pb = pred_boxes[pi]

        for gi in unmatched_gt_idx:
            gb = gt_boxes[gi]

            if pb[0] != gb[0] and iou(pb, gb) >= IOU_MATCH:
                class_errors += 1
                break

    dup_penalties = 0

    for i in range(len(pred_boxes)):
        for j in range(i + 1, len(pred_boxes)):

            pb_i, pb_j = pred_boxes[i], pred_boxes[j]

            if pb_i[0] != pb_j[0]:
                continue

            if iou(pb_i, pb_j) <= IOU_DUP:
                continue

            extra_idx = None

            if i in matched_pred and j not in matched_pred:
                extra_idx = j
            elif j in matched_pred and i not in matched_pred:
                extra_idx = i
            elif i not in matched_pred and j not in matched_pred:
                extra_idx = j
            else:
                continue

            extra_box = pred_boxes[extra_idx]

            same_class_gt_nearby = any(
                gb[0] == extra_box[0] and iou(extra_box, gb) > IOU_MATCH
                for gb in gt_boxes
            )

            if not same_class_gt_nearby:
                dup_penalties += 1

    return {
        "gt": len(gt_boxes),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "class_errors": class_errors,
        "dup_penalties": dup_penalties,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(frame_results):

    totals = {k: sum(r[k] for r in frame_results)
              for k in ("gt", "tp", "fn", "fp", "class_errors", "dup_penalties")}

    gt = totals["gt"]

    if gt == 0:
        sys.exit("[error] GT has zero boxes across all honeypot frames.")

    fn_rate = totals["fn"] / gt
    fp_rate = totals["fp"] / gt
    ce_rate = totals["class_errors"] / gt
    dup_rate = totals["dup_penalties"] / gt

    # Decision without quality score
    if fn_rate > 0.08:
        decision = "Reject (FN rate > 8%)"
    elif dup_rate > 0.10:
        decision = "Reject (Duplicate rate > 10%)"
    else:
        decision = "Accept"

    return {
        "totals": totals,
        "fn_rate": round(fn_rate, 4),
        "fp_rate": round(fp_rate, 4),
        "ce_rate": round(ce_rate, 4),
        "dup_rate": round(dup_rate, 4),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_frame_table(frame_results, filenames):

    header = f"{'Frame':<55} {'GT':>4} {'TP':>4} {'FN':>4} {'FP':>4} {'CE':>4} {'DP':>4}"

    print(header)
    print("-" * len(header))

    for fname, r in zip(filenames, frame_results):

        short = fname if len(fname) <= 55 else "…" + fname[-54:]

        print(
            f"{short:<55} {r['gt']:>4} {r['tp']:>4} {r['fn']:>4} "
            f"{r['fp']:>4} {r['class_errors']:>4} {r['dup_penalties']:>4}"
        )


def print_aggregate(agg):

    t = agg["totals"]

    print("\n=== Aggregate ===")

    print(f"  GT boxes       : {t['gt']}")
    print(f"  TP             : {t['tp']}")
    print(f"  FN             : {t['fn']}  (rate: {agg['fn_rate']:.1%})")
    print(f"  FP             : {t['fp']}  (rate: {agg['fp_rate']:.1%})")
    print(f"  Class errors   : {t['class_errors']}  (rate: {agg['ce_rate']:.1%})")
    print(f"  Dup penalties  : {t['dup_penalties']}  (rate: {agg['dup_rate']:.1%})")

    print(f"\n  Decision       : {agg['decision']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_task(task_name, gt_zip, pred_zip, tasks_dir, reports_dir):

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    task_dir = tasks_dir / task_name

    extract_zip(gt_zip, task_dir / f"gt_{task_name}")
    extract_zip(pred_zip, task_dir / task_name)

    gt_map = build_filename_map(task_dir / f"gt_{task_name}")
    pred_map = build_filename_map(task_dir / task_name)

    honeypot_frames = sorted(set(gt_map) & set(pred_map))

    print(f"\nHoneypot frames found: {len(honeypot_frames)}\n")

    frame_results = []

    for fname in honeypot_frames:

        gt_boxes = parse_annotation(gt_map[fname])
        pred_boxes = parse_annotation(pred_map[fname])

        frame_results.append(score_frame(gt_boxes, pred_boxes))

    print_frame_table(frame_results, honeypot_frames)

    agg = aggregate(frame_results)

    print_aggregate(agg)

    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"{task_name}_score_report.json"

    report = {
        "task": task_name,
        "frames": [{"filename": f, **r} for f, r in zip(honeypot_frames, frame_results)],
        "aggregate": agg,
    }

    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n  Report saved → {report_path}")


def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python scorer.py <tasks_dir>")

    tasks_dir = Path(sys.argv[1]).resolve()
    reports_dir = tasks_dir.parent / "reports"

    pairs = find_task_pairs(tasks_dir)

    print(f"Found {len(pairs)} task(s) to score.")

    for task_name, gt_zip, pred_zip in pairs:
        run_task(task_name, gt_zip, pred_zip, tasks_dir, reports_dir)


if __name__ == "__main__":
    main()