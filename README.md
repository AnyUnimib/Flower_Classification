# Flower Image Classification — CNN vs. Transfer Learning

**Any Das**

*Originally developed as a group project (3 contributors); this repository reflects my individual exploration and contribution to the modeling pipeline.*

A comparative deep learning project classifying flower images into 5 categories (Daisy, Dandelion, Roses, Sunflowers, Tulips), built to answer a practical question: on a small dataset, how much does transfer learning actually buy you over a CNN trained from scratch — and where do the returns from further tuning a custom CNN start to flatten out?

## Why this project

With only ~3,670 images spread across 5 classes, this is a textbook case for transfer learning — but rather than assume that and move on, the project builds out **4 models in a deliberate progression** (baseline CNN → pretrained backbone → regularized CNN → tuned CNN) to actually measure where the performance gap comes from and how far a from-scratch model can be pushed to close it.

## Dataset

- **Source:** TensorFlow's `flower_photos` dataset (3,670 images, 5 classes, varying sizes/orientations)
- **Split:** 70% train / 15% validation / 15% test
- **Preprocessing:** resize to 160×160, normalize to [0,1], augment with random flip / rotation / zoom / contrast
- **Pipeline:** built with `tf.data` — cached, shuffled, prefetched (`AUTOTUNE`) for training throughput

## Models — a deliberate progression, not just a leaderboard

### Model 1 — Custom CNN (baseline)
`Conv2D → MaxPooling` ×3 → `Flatten` → `Dense` → `Softmax`, ~6.6M parameters, dropout for regularization.
**Test accuracy: 0.59 · Test loss: 1.05**

Built first to establish a from-scratch baseline — most of the model's capacity (and overfitting risk) sits in the dense layer after flattening, which is the first thing the later iterations address.

### Model 2 — MobileNetV2 (transfer learning)
Frozen MobileNetV2 backbone + custom head (`GlobalAveragePooling2D → Dropout → Dense(128, ReLU, L2) → Dropout → Dense(5, softmax)`), Adam (lr=0.001), `EarlyStopping` / `ModelCheckpoint` / `ReduceLROnPlateau`, 10 epochs.
**Test accuracy: 0.87 · Test loss: 0.37**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Daisy | 0.91 | 0.83 | 0.87 |
| Dandelion | 0.84 | 0.95 | 0.89 |
| Roses | 0.78 | 0.93 | 0.85 |
| Sunflowers | 0.92 | 0.84 | 0.87 |
| Tulips | 0.90 | 0.78 | 0.84 |
| **Accuracy** | | | **0.87** |

A ~28-point accuracy jump over Model 1 from pretrained features alone, with no architecture change on the classification side beyond a small head.

### Model 3 — Improved custom CNN
Same base as Model 1, rebuilt with `Conv2D → BatchNorm → Activation` ordering (instead of activation-first), `same` padding, L2 regularization, and `GlobalAveragePooling2D` replacing `Flatten` — the goal being to close the gap to Model 2 by fixing architectural weaknesses rather than borrowing pretrained weights.

### Model 4 — Tuned Model 3
Learning rate `0.001 → 0.0001`, dropout `0.3 → 0.4`, L2 factor `0.0001 → 0.001`, plus one additional convolution block — testing whether tuning regularization strength and adding depth could push the custom architecture further, at the cost of added training complexity.

## Results Summary

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Model 1 — Custom CNN | 0.59 | 1.05 |
| **Model 2 — MobileNetV2** | **0.87** | **0.37** |
| Model 3 — Regularized CNN | see notebook | see notebook |
| Model 4 — Tuned CNN | see notebook | see notebook |

MobileNetV2 reached ~94% training / ~88% validation accuracy vs. ~59%/~53% for the original custom CNN, with a consistently lower loss curve across all 10 epochs.

## What stood out

- **Transfer learning wasn't just better — it was better with less effort.** A frozen backbone and a lightweight head outperformed a purpose-built CNN by ~28 points of test accuracy, without any architecture search on the CNN side yet.
- **Class-level errors were informative, not just noise.** MobileNetV2's confusion matrix showed most of its errors were **Roses ↔ Tulips**, which tracks — both share overlapping color palettes and petal structure at 160×160 resolution, a genuinely hard visual distinction rather than a modeling failure.
- **Architecture fixes (Model 3) matter, but have a ceiling.** Adding batch norm, proper layer ordering, and regularization measurably improves a from-scratch CNN, but on ~3,670 images it's unlikely to close the full gap to a backbone pretrained on millions of images — the data volume is the real constraint, not just the architecture.
- **Regularization tuning (Model 4) is a trade-off, not a free win.** Lower learning rate and stronger L2/dropout reduce overfitting risk but slow convergence — worth it only if training budget allows more epochs.

## Repository Structure

```
├── flower_classification.ipynb    # Full notebook: data pipeline, all 4 models, training, evaluation
├── README.md
```

*(Dataset is downloaded automatically by the notebook from TensorFlow's servers — not stored in this repo.)*

## Running it

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
jupyter notebook flower_classification.ipynb
```

Running top-to-bottom downloads the dataset, builds the `tf.data` pipeline, trains all 4 models, and generates evaluation plots and confusion matrices.

## What I'd improve next

- **Fine-tune MobileNetV2** (unfreeze the top layers) instead of keeping the backbone fully frozen — likely the single highest-leverage next step given the current gap to Model 1.
- **Increase input resolution** from 160×160 to 224×224 to see if the Roses/Tulips confusion improves with finer detail.
- **Stronger augmentation** (zoom, rotation) beyond the current flip/brightness/contrast set.
- **Ensemble MobileNetV2 with the tuned custom CNN** to see if their error patterns are complementary enough to boost accuracy further.

## Tech Stack
Python · TensorFlow / Keras · scikit-learn · Matplotlib · Seaborn
