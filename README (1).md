Honeypot Annotation Scoring System
==================================

Example Task Used
-----------------

This README demonstrates the expected format using the following task
names:

a2b\_ekkaduthangal\_2\_20260218.zip
gt\_a2b\_ekkaduthangal\_2\_20260218.zip

Where:

-   **a2b\_ekkaduthangal\_2\_20260218.zip** → Prediction annotations
-   **gt\_a2b\_ekkaduthangal\_2\_20260218.zip** → Ground truth
    annotations

Both files must exist inside the same tasks directory.

------------------------------------------------------------------------

Project Overview
================

The Honeypot Scoring System evaluates object detection annotations by
comparing **predicted bounding boxes** with **ground truth bounding
boxes** in YOLO format.

The system:

1.  Extracts zip files
2.  Matches annotation frames
3.  Compares bounding boxes
4.  Calculates detection metrics
5.  Generates a JSON evaluation report

------------------------------------------------------------------------

Expected Folder Structure
=========================

Example:

tasks/ │ ├── a2b\_ekkaduthangal\_2\_20260218.zip ├──
gt\_a2b\_ekkaduthangal\_2\_20260218.zip

After extraction the structure will be:

tasks/ │ ├── a2b\_ekkaduthangal\_2\_20260218/ │ ├── frame1.txt │ └──
frame2.txt │ ├── gt\_a2b\_ekkaduthangal\_2\_20260218/ │ ├── frame1.txt │
└── frame2.txt

------------------------------------------------------------------------

Annotation Format (YOLO)
========================

Each annotation file must follow YOLO format:

class\_id x\_center y\_center width height

Example:

0 0.45 0.62 0.30 0.40 1 0.33 0.50 0.10 0.20

All coordinates must be normalized between **0 and 1**.

------------------------------------------------------------------------

Evaluation Metrics
==================

GT --- Ground Truth
-------------------

Total number of real objects in the image.

TP --- True Positive
--------------------

Correct detection where:

-   class is correct
-   IoU ≥ 0.6

FN --- False Negative
---------------------

Object exists in GT but **not detected**.

FP --- False Positive
---------------------

Prediction exists but **no object in GT**.

CE --- Class Error
------------------

Bounding box overlaps but **class label is wrong**.

DP --- Duplicate Penalty
------------------------

Multiple predicted boxes detect the **same object**.

------------------------------------------------------------------------

IoU Thresholds
==============

IOU\_MATCH = 0.6 IOU\_DUP = 0.8

-   0.6 → minimum overlap for correct detection
-   0.8 → threshold for duplicate detection

------------------------------------------------------------------------

Running the Scoring Script
==========================

Run the script using:

python scorer.py tasks

Example:

python scorer.py tasks

The script will automatically detect:

a2b\_ekkaduthangal\_2\_20260218.zip
gt\_a2b\_ekkaduthangal\_2\_20260218.zip

------------------------------------------------------------------------

Output Example
==============

Frame GT TP FN FP CE DP
-----------------------

frame\_001.txt 3 2 1 0 0 0 frame\_002.txt 2 2 0 1 0 0

------------------------------------------------------------------------

Aggregate Result Example
========================

GT boxes : 5 TP : 4 FN : 1 FP : 1 Class errors : 0 Dup penalties : 0

Decision : Accept

------------------------------------------------------------------------

Generated Report
================

The system automatically generates a JSON report:

reports/ └── a2b\_ekkaduthangal\_2\_20260218\_score\_report.json

The JSON file contains:

-   frame level results
-   aggregated metrics
-   final decision

------------------------------------------------------------------------

Use Cases
=========

This tool is useful for:

-   Dataset annotation validation
-   Object detection quality checking
-   Labeling team performance evaluation
-   Honeypot annotation verification
