# Honeypot Scoring System

Scores annotator batches against ground truth honeypot frames. Outputs per-frame offense counts, a quality score, and an Accept / Rework / Reject decision.

---

## Setup

No dependencies beyond the Python standard library.

```
honeypot/
├── scorer.py
├── tasks/      ← drop zip files here
└── reports/    ← reports are saved here
```

---

## Usage

### 1. Download the two zip files for a task

| File | Contents |
|---|---|
| `<task_name>.zip` | Annotator's full exported batch |
| `gt_<task_name>.zip` | Ground truth honeypot frames |

### 2. Drop both zips into `tasks/`

```
tasks/
├── a2b_ekkaduthangal_2_20260218_4.zip
└── gt_a2b_ekkaduthangal_2_20260218_4.zip
```

### 3. Run

```bash
python scorer.py tasks/
```

That's it. The scorer extracts, scores, prints results, and saves the report. Multiple task pairs can be dropped at once — all are scored in a single run.

---

## Output

**Console — per-frame table:**
```
Frame                                                      GT   TP   FN   FP   CE   DP
---------------------------------------------------------------------------------------
ekkaduthangal_2_channel1002_20260215T090132_22.txt          3    2    1    4    1    0
...
```

**Console — aggregate block:**
```
=== Aggregate ===
  GT boxes       : 3
  FN             : 1  (rate: 33.3%)
  FP             : 78  (rate: 2600.0%)
  Class errors   : 1  (rate: 33.3%)
  Dup penalties  : 0  (rate: 0.0%)

  Quality score  : -2600.00
  Decision       : Reject (FN rate > 8%)
```

**JSON report** saved to `reports/<task_name>_score_report.json`.

---

## Scoring Logic

### Metrics per frame

| Column | Meaning |
|---|---|
| GT | Total ground truth boxes |
| TP | Correctly matched boxes (same class, IoU ≥ 0.6) |
| FN | Missed GT boxes |
| FP | All unmatched pred boxes |
| CE | Class errors — correct location, wrong class (subset of FP) |
| DP | Duplicate penalties — duplicate pred box with class mismatch to GT (subset of FP) |

### Quality score

```
FN_rate  = FN / GT
FP_rate  = FP / GT
CE_rate  = class_errors / GT
DP_rate  = dup_penalties / GT

Quality = 100 − (2×FN_rate + FP_rate + DP_rate + CE_rate) × 100
```

FN is weighted 2× because missing an object is worse than a false alarm.

### Decision

| Condition | Decision |
|---|---|
| FN_rate > 8% | Reject |
| DP_rate > 10% | Reject |
| Quality ≥ 92 | Accept |
| 88 ≤ Quality < 92 | Rework |
| Quality < 88 | Reject |
