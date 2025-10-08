
# Drone and Jet Detection using Deep Learning

This project focuses on detecting drones (and in the future, jets) using a deep learning model. Users can upload an image containing a drone, set a confidence level, and generate detection results.

---

## How to Use

1. Visit the link provided in the **About** section.
2. Select the model file you want to use:

   * `best.pt` — Recommended PyTorch model
   * `best.onnx` — For ONNX runtime
   * `best.engine` — Currently not supported due to GPU limitations
3. Upload an image containing a drone.
4. Set the desired confidence threshold.
5. Click on **Generate** to view detection results.

---

## Future Scope

* Implement real-time detection for drones and jets.
* Enhance precision and accuracy to support defense and surveillance systems.

---

## Tech Stack

* Python
* PyTorch / ONNX
* Computer Vision
* Deep Learning

---

## Note

If the `best.engine` model fails to run, use either `best.pt` or `best.onnx` due to current GPU limitations.

---
