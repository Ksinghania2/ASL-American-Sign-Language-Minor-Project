# Real-Time American Sign Language (ASL) Translation Engine

A high-performance computer vision and deep learning translation engine engineered in Python to parse live camera video streams and translate American Sign Language (ASL) gestures into real-time alphanumeric text and speech output.

The architecture integrates high-speed matrix pre-processing pipelines with convolutional neural network ingestion models, using asynchronous multi-threaded audio execution to eliminate frame freezing during real-time inference.

## Architectural Framework

1. Video Frame Ingestion: Leverages OpenCV to manage native hardware camera streams, capturing dynamic matrices at high frame rates for active spatial analysis.
2. Color-Space and Matrix Normalization: Maps incoming frame matrices from native OpenCV `BGR` patterns into standard `RGB` arrays to match training requirements. Pixel values are downscaled to floating-point values between `0.0` and `1.0`.
3. Region of Interest (ROI) Isolation: Restricts matrix evaluations to a dedicated bounding region to minimize environmental noise and decrease computational inference overhead.
4. Asynchronous Speech Synthesis: Deploys background worker threads to execute text-to-speech rendering, ensuring the visual processing stream never suffers from processing lag.

## Deployment and Verification

### 1. Environment Setup
Install the necessary computer vision and deep learning dependencies:
```bash
pip3 install -r requirements.txt