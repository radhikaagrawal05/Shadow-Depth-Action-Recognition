# Shadow Depth Action Recognition

Physics-based vision system that estimates hand-to-face distance using shadow occlusion
and inverse square modeling.

## Method
- Face detection using MediaPipe
- Shadow segmentation using grayscale thresholding
- Shadow Area computation
- Depth estimation using:
  Z = K / sqrt(Shadow_Area)
- Heatmap visualization of intensity loss
- Action classification: Touching / Not Touching
