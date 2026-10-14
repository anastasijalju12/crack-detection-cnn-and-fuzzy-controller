# Concrete Crack Detection (CNN) and Fuzzy PI Controller

Neural Networks course project, Group 9 — Milica Pavlović, Anastasija Ljubičić.

## 1. Concrete crack detection

Binary image classification (Positive = crack, Negative = no crack) with a convolutional neural network built in TensorFlow/Keras.

**Dataset:** [Surface Crack Detection (Kaggle)](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) — 40,000 RGB images (227 × 227), 20,000 per class. Download it and extract it so that the folder looks like this:

```
archive/
├── Positive/
└── Negative/
```

**Run:**

```bash
pip install -r requirements.txt
python crack_cnn.py --data-dir archive --output-dir results
```

All figures, the trained model and the metrics are written to `results/`.

**Pipeline**

- Stratified split: 70% train / 15% validation / 15% test (28,000 / 6,000 / 6,000)
- Images resized to 224 × 224 and rescaled to [0, 1]
- Augmentation (training set only): rotation 10°, zoom 0.1, horizontal flip
- Model: 3 × (Conv2D 3×3 + ReLU + MaxPooling) with 32/64/128 filters → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid); 11,169,089 parameters
- Binary cross-entropy, Adam, 5 epochs

**Results (test set, 6,000 images)**

| Metric | Value |
|---|---|
| Accuracy | 99.05% |
| Precision (Positive) | 98.64% |
| Recall (Positive) | 99.47% |
| F1 score | 99.05% |

## 2. Fuzzy PI controller

Mamdani fuzzy PI controller (incremental form) for the plant

G(s) = e^(-4s) / (s⁴ + 26s³ + 193s² + 510s + 90)

with reference r ∈ [-0.4, 0.4] and control signal u ∈ [-100, 100], tested with and without an input disturbance d ∈ [-50, 50].

- `fuzzy-controller/mojFIS.fis` — fuzzy inference system (inputs E and dE, output dU, 5 membership functions each, 25 rules)
- `fuzzy-controller/zad2.slx` — Simulink model (MATLAB R2025b, Fuzzy Logic Toolbox)

Open `zad2.slx` from the `fuzzy-controller` folder and press **Run**; results are shown in the Scope blocks. The `.fis` file must stay in the same folder as the model, since the Fuzzy Logic Controller blocks load it by name.

## Report

`Report_EN.docx` contains the full description, figures and discussion of both parts.
