# Tracer

Computer vision application for detecting and tracing tool contours using ArUco markers for scale reference.

## Features

- Real-time camera feed with undistortion correction
- ArUco marker detection for automatic scaling
- Contour detection using brightness thresholding or Canny edge detection
- Tolerance adjustment for manufacturing specifications
- SVG export functionality
- Camera parameter calibration interface

## Requirements

- Python 3.8+
- OpenCV camera (USB or built-in)
- ArUco markers for scale reference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/contour-tracer.git
cd contour-tracer
