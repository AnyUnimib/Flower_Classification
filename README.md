# Flower Image Classification using Deep Learning

Classifying flower images into 5 categories — **Daisy, Dandelion, Roses, Sunflowers, Tulips** — using a custom CNN and a transfer-learning (MobileNetV2) approach, built and compared in TensorFlow/Keras.

## Authors
- Any Das
- Semen Sazanov
- Hadis Forghani

## Dataset
We use the **`flower_photos`** dataset released by TensorFlow (3,670 images across 5 classes, varying sizes/orientations).

- Source: `http://download.tensorflow.org/example_images/flower_photos.tgz`
- The dataset is **downloaded automatically** by the notebook (no need to manually download or upload it to this repo).
- Split: 70% train / 15% validation / 15% test.

## Project Structure
```
├── Flower_Classification.ipynb     # Main notebook: data loading, preprocessing, models, training, evaluation
├── Flower_Image_Classification.pdf # Project presentation slides
└── README.md
```

## Methodology
1. Dataset import & loading
2. Data preprocessing (normalization, augmentation, class labeling, prefetching)
3. Model building
4. Model compilation
5. Model training
6. Model evaluation
7. Performance comparison

### Preprocessing
- Images resized to `160x160`
- Pixel normalization (`/255.0`)
- Data augmentation: random horizontal flip, rotation, zoom, contrast
- `tf.data` pipeline: caching, shuffling, prefetching (`AUTOTUNE`)

## Models

### Model 1 — Custom CNN (from scratch)
`Conv2D → MaxPooling2D` ×3 → `Flatten` → `Dense` → `Output(softmax)`
- ~6.6M parameters
- Dropout for regularization
- **Test accuracy: 0.59**, Test loss: 1.05

### Model 2 — MobileNetV2 (transfer learning)
`Input → MobileNetV2 (frozen) → GlobalAveragePooling2D → Dropout → Dense(128, ReLU, L2) → Dropout → Dense(5, softmax)`
- Adam optimizer (lr=0.001), sparse categorical crossentropy
- Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
- 10 epochs
- **Test accuracy: 0.87**, Test loss: 0.37
- Per-class F1: Daisy 87, Dandelion 89, Roses 85, Sunflowers 87, Tulips 84 (macro avg 86, accuracy 87)

### Model 3 — Improved Custom CNN
Base of Model 1, with:
- Batch Normalization (`Conv2D → BN → Activation` order)
- L2 regularization
- `same` padding
- `GlobalAveragePooling2D` instead of `Flatten`

### Model 4 — Tuned Model 3
- Learning rate: `0.001 → 0.0001`
- Dropout: `0.3 → 0.4`
- L2 factor: `0.0001 → 0.001`
- One extra convolution block added

## Results Summary

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Model 1 — Custom CNN | 0.59 | 1.05 |
| Model 2 — MobileNetV2 | **0.87** | **0.37** |
| Model 3 | see notebook | see notebook |
| Model 4 (tuned Model 3) | see notebook | see notebook |

MobileNetV2 (Model 2) clearly outperformed the custom CNN, reaching ~94% training / ~88% validation accuracy vs. ~59%/~53% for the custom CNN, confirming the benefit of transfer learning on a limited dataset.

## Conclusion
The best results were obtained with **MobileNetV2** (88% validation accuracy, 87% test accuracy). The custom CNN, while less accurate, still meaningfully improved on a plain MLP baseline by leveraging spatial structure in the images.

## Future Improvements
- Fine-tune MobileNetV2 instead of keeping it fully frozen
- Increase input resolution (e.g., 224×224 instead of 160×160)
- Try stronger augmentation (zoom, rotation) beyond flip/brightness/contrast
- Ensemble MobileNetV2 with the tuned custom CNN

## How to Run
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
jupyter notebook Flower_Classification.ipynb
```
Running the notebook top-to-bottom will download the dataset, preprocess it, train all four models, and generate the evaluation plots/confusion matrices.
