# Video Action Recognition: 3D CNN vs. VideoMAE

A hands-on project exploring how deep learning models understand video—starting from a manual 3D convolution in NumPy and Farneback Optical Flow, to training a custom 3D CNN from scratch and comparing it against a fine-tuned VideoMAE Transformer.

## Dataset
I recorded a custom dataset of **82 videos** across 4 action classes (`wave`, `walk`, `sitting`, `standing`) under varying lighting and camera angles:
- **Train split:** 66 clips (16 frames sampled evenly per clip)
- **Test split:** 16 clips (4 per class)

## Results: 3D CNN vs. VideoMAE

| Metric | Custom 3D CNN (From Scratch) | VideoMAE (Fine-Tuned) |
| :--- | :--- | :--- |
| **Parameters** | 3.28M | 86.23M |
| **Input Shape** | `16 x 112 x 112 x 3` | `16 x 3 x 224 x 224` |
| **Epochs** | 50 | 15 (converged at Epoch 1) |
| **Train Accuracy** | 100.0% (66/66) | 100.0% (66/66) |
| **Test Accuracy** | **81.2%** (13/16, peak 93.8%) | **100.0%** (16/16) |

### 3D CNN Test Confusion Matrix
Rows = Actual, Columns = Predicted (`[Wave, Walk, Sitting, Standing]`):
```text
Wave:     [0, 0, 3, 1]  (0/4)
Walk:     [0, 3, 0, 1]  (3/4)
Sitting:  [0, 0, 4, 0]  (4/4)
Standing: [0, 0, 0, 4]  (4/4)
