# Usage Guide

## Downloading the Input Video
To download the input video, use the following command:

```bash
!gdown https://drive.google.com/uc?id=1RkcIzaHbH3hNJKAfyvMouUdGnd56nrsW
```

## Running the Python Preprocessing Script

```bash
python vsr_preprocess.py
```

## Preprocessing Tracks

1. **CLAHE**
   - Parameters: `clipLimit=2.0`, `tileGridSize=(8,8)`
   - Process the video in grayscale and resize to `224x224`.

2. **Procrustes Alignment**
   - Method: Use MediaPipe Face Mesh for `468` landmarks.
   - Procedure: Align faces such that the eyes are horizontal and the mouth center is at `(112,160)` using `cv2.getSimilarityTransform` or `cv2.estimateAffinePartial2D`.

3. **Partition-Time Masking (PTM)**
   - Configuration: Implement a `4x4` grid with `alpha=0.2`.
   - Masking: Mask `2` of `16` partitions in masked frames to the mean or `0`.

## Data Links
| Description | Link |
|-------------|------|
| Baseline    | [https://drive.google.com/file/d/1RkcIzaHbH3hNJKAfyvMouUdGnd56nrsW/view?usp=sharing](https://drive.google.com/file/d/1RkcIzaHbH3hNJKAfyvMouUdGnd56nrsW/view?usp=sharing) |
| CLAHE       | [https://drive.google.com/file/d/1DHVsoaw3zIQgeg19mWdjJaIrLIXf9XSF/view?usp=sharing](https://drive.google.com/file/d/1DHVsoaw3zIQgeg19mWdjJaIrLIXf9XSF/view?usp=sharing) |
| Procrustes   | [https://drive.google.com/file/d/1plVmyTE4cVB7vC3nMmmaQhUqG2uOhFUm/view?usp=sharing](https://drive.google.com/file/d/1plVmyTE4cVB7vC3nMmmaQhUqG2uOhFUm/view?usp=sharing) |
| PTM         | [https://drive.google.com/file/d/1-BW1xb9Osidt_Nzt9zc91SdFFUjZX5bQ/view?usp=sharing](https://drive.google.com/file/d/1-BW1xb9Osidt_Nzt9zc91SdFFUjZX5bQ/view?usp=sharing) |

---

Follow these steps to ensure reproducibility of the preprocessing pipeline.
