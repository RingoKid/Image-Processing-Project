# eurosat-bigearthnet-transfer

Transfer learning from **BigEarthNet v2.0 (reBEN)** pretrained models onto the multi-spectral **EuroSAT** satellite image dataset.

## Core Pipeline & Technical Approach

1. **Dataset (`eurosat/all`)**: Loaded via TensorFlow Datasets (`tensorflow_datasets`), consisting of $64 \times 64$ Sentinel-2 satellite images across 13 spectral bands and 10 land-use/land-cover classes.
2. **Preprocessing & Alignment**:
   - **Band Selection**: Filters EuroSAT's 13 Sentinel-2 bands down to the 10 bands expected by BigEarthNet v2.0 models (dropping bands B01, B09, and B10).
   - **Spatial Resizing**: Upsamples images from $64 \times 64$ to $120 \times 120$.
   - **Normalization**: Applies z-score normalization using precomputed BigEarthNet v2.0 band means and standard deviations from `configilm` (`band_combi_to_mean_std`).
   - **Framework Bridge**: Converts the preprocessed `tf.data.Dataset` into PyTorch `Dataset` / `DataLoader` splits (70% train, 15% validation, 15% test).
3. **Transfer Learning (`BIFOLD-BigEarthNetv2-0`)**:
   - Uses pretrained Sentinel-2 backbones from Hugging Face / `configilm` (`BigEarthNetv2_0_ImageClassifier`), including **ResNet-50**, **ResNet-101**, **Vision Transformer (ViT)**, and **MLP-Mixer**.
   - Replaces the classification head for 10-class classification and freezes backbone layers (selectively unfreezing the final blocks/layers for fine-tuning).
   - Evaluates models using multi-class metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrices) and explores class-weighted cross-entropy, Focal Loss, and ViT attention map visualization.
