# Honeypot Scoring System — Plan

## Input
Two folders dropped into the working directory:
- `gt_<task_name>/` — ground truth (honeypot frames)
- `<task_name>/` — annotator's full exported batch

---

## Step 1 — Discover Honeypot Frames
- Walk both folder trees, extract just filenames (ignoring deep nested paths)
- Build `{filename → full_path}` maps for GT and pred separately
- Intersection of filenames = honeypot frames to score
- Skip `train.txt`, `obj.data`, `obj.names`

---

## Step 2 — Per-Frame Processing (for each honeypot frame)

### 2a. Parse annotations
- Read YOLO format `class_id x_center y_center w h` (normalized)
- Convert to absolute `[x1, y1, x2, y2]` using a standard image size (or keep normalized — IoU works either way since both GT and pred are on the same image)

### 2b. Compute IoU matrix
- Full matrix: every GT box vs every pred box (class-agnostic at this stage)

### 2c. 1-to-1 matching → TP / FN / FP
- Filter to pairs where: same class AND IoU ≥ 0.6
- Sort by IoU descending, greedily assign (each GT and each pred used at most once)
- Matched pairs → **TP**
- Unmatched GT boxes → **FN**
- Unmatched pred boxes → **FP** (base bucket — all go here initially)

### 2d. Class error detection (out of the FP pool)
- For each unmatched pred box: check IoU ≥ 0.6 against any unmatched GT box (now ignoring class)
- If a match is found with a different class → tag this pred box as **Class Error** (still in FP, additionally counted in class_error count)

### 2e. Duplicate detection (out of the FP pool)
- Among all pred boxes (matched or not), find pairs with: same class AND IoU > 0.8 between pred boxes
- The "extra" box in each duplicate pair = the one that did NOT get matched to a GT box
- For each extra/duplicate pred box:
  - Check if a GT box exists at that location (IoU > 0.6) with the **same class**
  - **Same class as GT** → sloppy but correct, counts only as **FP** (no extra penalty)
  - **No GT nearby OR GT has different class** → counts as **FP + Duplicate Penalty** (tagged in duplicate_penalty count)

---

## Step 3 — Per-Frame Offense Counts
Each frame records:

| Metric | Description |
|---|---|
| GT boxes | Total GT boxes in this frame |
| TP | Correctly matched boxes |
| FN | Missed GT boxes |
| FP | All unmatched pred boxes |
| Class errors | Subset of FP: correct location, wrong class |
| Duplicate penalties | Subset of FP: duplicate pred box with class mismatch to GT |

---

## Step 4 — Aggregate Across All Honeypot Frames
Sum all counts: `GT_total, TP_total, FN_total, FP_total, class_error_total, dup_penalty_total`

---

## Step 5 — Compute Rates
```
FN_rate           = FN_total / GT_total
FP_rate           = FP_total / GT_total
Class_error_rate  = class_error_total / GT_total
Duplicate_rate    = dup_penalty_total / GT_total
```

---

## Step 6 — Quality Score
```
Quality = 100 - (2×FN_rate + 1×FP_rate + 1×Duplicate_rate + 1×Class_error_rate) × 100
```

---

## Step 7 — Decision

| Condition | Decision |
|---|---|
| FN_rate > 8% | Reject |
| Duplicate_rate > 10% | Reject |
| Quality ≥ 92 | Accept |
| 88 ≤ Quality < 92 | Rework |
| Quality < 88 | Reject |

---

## Step 8 — Output
- **Console**: per-frame table showing all offense counts per frame
- **Aggregate block**: total counts, rates, quality score, decision
- **JSON file**: full report saved as `<task_name>_score_report.json`
