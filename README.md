# InfraScan Sentinel 🏢🏛️



<img width="674" height="721" alt="image" src="https://github.com/user-attachments/assets/1f5d56b6-4f79-4d04-a7c7-5b405d0f1ecb" />
# InfraScan Sentinel

## Overview

InfraScan Sentinel is a computer vision project aimed at detecting seismic gaps between adjacent buildings using a single image.

The goal is to estimate whether the gap between buildings is safe or poses a collision risk during earthquakes.

---

## Approach

The current system combines:

* **MiDaS depth estimation** → to understand relative distance
* **Depth gradient** → to detect candidate gap boundaries
* **Hough Transform** → to refine structural edges
* **Fusion logic** → combines depth and geometric edges

---

## Key Idea

A valid seismic gap should satisfy:

> The region between two building edges should have higher depth (farther away) than the buildings themselves.

This is used as a validation step instead of relying only on edge detection.

---

## Current Status

* Prototype pipeline implemented
* Tested on real images
* Edge detection partially working
* Depth-based validation implemented (under testing)

---

## Current Issues

* Coordinate mismatch between depth map and image space
* Inconsistent gap boundary detection
* Hardcoded parameters (distance, building height)
* No ground truth calibration yet

---

## Next Steps

* Fix coordinate system (depth vs image space)
* Validate depth-based gap condition
* Collect real-world measurements for calibration
* Improve robustness across different scenes

---

## Note

This is an ongoing project under active development.

