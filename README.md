# Vector Perception Demo — Humanoid Vision Stack

An independent browser demo of warehouse perception primitives: real-time object
detection, multi-object tracking, human pose estimation, and simulated depth
visualization. Built as a portfolio project aligned with an intralogistics
humanoid-robotics focus.

## What it does

- Webcam feed (or a labeled animated simulated scene when no camera is available)
- COCO-SSD object detection: bounding boxes, labels, confidence scores
- MoveNet pose estimation: skeleton overlay on people
- IoU-based track IDs that persist across frames
- Simulated depth panel with a near-to-far legend
- Telemetry: FPS, inference latency (ms), active detection count
- A "how it works" section explaining the pipeline

## Run it

Serve the folder with any static server:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`. (`file://` works too, but the browser
falls back to the simulated scene because camera access needs a secure context.)

A live deployment of this exact code runs on GitHub Pages (see the repo's
deployments).

## Stack

TensorFlow.js (COCO-SSD and MoveNet models), canvas rendering, zero backend,
zero API keys. UI colors sampled from Vector Robotics' production CSS
(dark slate `#0f172a`, electric cyan `#06b6d4`).

## Framing

This is an independent portfolio project demonstrating perception primitives in
the browser. It is not affiliated with Vector Robotics and does not run on, or
replicate, their robot stack.
