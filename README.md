# Machine Learning Notebook

## Overview
This project implements and compares various machine learning pipelines for aerial scene classification. Techniques include Local Binary Patterns (LBP), Scale-Invariant Feature Transform with Bag-of-Visual-Words (SIFT + BoVW), and Principal Component Analysis (PCA), followed by classical classifiers like K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).

## Features
- **Image Preprocessing** with OpenCV
- **Feature Extraction** using:
  - Local Binary Patterns (LBP)
  - PCA for dimensionality reduction
- **Classification Models**:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Model Evaluation** using:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix
  - ROC Curve and AUC
- **Visualization** using Matplotlib

## Used Libraries
- | Library           | Version  |
  | ----------------- | -------- |
  | `cv2` (OpenCV)    | 4.11.0   |
  | `numpy`           | 2.1.3    |
  | `scikit-learn`    | 1.6.1    |
  | `matplotlib`      | 3.10.1   |
  | `skimage`         | 0.25.2   |
  | `os` / `warnings` | Built-in |

## Experimental Setups

### 1. Basic LBP + PCA Models

| Method                 | Classifier | Accuracy | F1 Score | Precision | Recall |
| ---------------------- | ---------- | -------- | -------- | --------- | ------ |
| LBP (r=1, p=8, PCA=30) | KNN        | 57.7%    | 0.579    | 0.597     | 0.577  |
| LBP (r=1, p=8, PCA=30) | SVM        | 64.5%    | 0.644    | 0.650     | 0.645  |

---

### 2. SIFT + BoVW (k=60) + PCA Models

| Method              | Classifier | Accuracy | F1 Score | Precision | Recall |
| ------------------- | ---------- | -------- | -------- | --------- | ------ |
| SIFT (k=60, PCA=30) | KNN        | 40.8%    | 0.402    | 0.426     | 0.408  |
| SIFT (k=60, PCA=30) | SVM        | 50.3%    | 0.497    | 0.502     | 0.503  |

---

### 3. LBP + SIFT Fusion Features

| Method         | Classifier | Accuracy | F1 Score | Precision | Recall |
| -------------- | ---------- | -------- | -------- | --------- | ------ |
| LBP+SIFT (PCA) | KNN        | 63.3%    | 0.635    | 0.647     | 0.633  |
| LBP+SIFT (PCA) | SVM        | 72.5%    | 0.723    | 0.725     | 0.725  |

---

### 4. Ablation Study

This section explores the impact of different LBP configurations (`r=1,p=8` vs. `r=2,p=16`) and SIFT codebook sizes (`k=30` vs. `k=60`) with PCA dimensions (`20` or `30`) and two classifiers (`KNN`, `SVM`).

| Feature Type | PCA  | Classifier | Accuracy | F1 Score | Precision | Recall |
| ------------ | ---- | ---------- | -------- | -------- | --------- | ------ |
| LBP r1p8     | 20   | KNN        | 57.8%    | 0.580    | 0.595     | 0.578  |
| LBP r1p8     | 20   | SVM        | 64.2%    | 0.643    | 0.648     | 0.642  |
| LBP r1p8     | 30   | KNN        | 57.7%    | 0.579    | 0.597     | 0.577  |
| LBP r1p8     | 30   | SVM        | 64.5%    | 0.644    | 0.650     | 0.645  |
| LBP r2p16    | 20   | KNN        | 32.2%    | 0.319    | 0.334     | 0.322  |
| LBP r2p16    | 20   | SVM        | 42.4%    | 0.419    | 0.425     | 0.424  |
| LBP r2p16    | 30   | KNN        | 31.0%    | 0.304    | 0.323     | 0.310  |
| LBP r2p16    | 30   | SVM        | 41.0%    | 0.406    | 0.413     | 0.410  |
| SIFT k30     | 20   | KNN        | 39.0%    | 0.383    | 0.398     | 0.390  |
| SIFT k30     | 20   | SVM        | 46.0%    | 0.451    | 0.451     | 0.460  |
| SIFT k30     | 30   | KNN        | 36.8%    | 0.362    | 0.382     | 0.368  |
| SIFT k30     | 30   | SVM        | 47.0%    | 0.461    | 0.462     | 0.470  |
| SIFT k60     | 20   | KNN        | 40.9%    | 0.403    | 0.415     | 0.409  |
| SIFT k60     | 20   | SVM        | 50.0%    | 0.492    | 0.496     | 0.500  |
| SIFT k60     | 30   | KNN        | 40.8%    | 0.402    | 0.426     | 0.408  |
| SIFT k60     | 30   | SVM        | 50.3%    | 0.497    | 0.502     | 0.503  |

---

### 5. Unbalanced Data Experiment

| Setting          | Accuracy | F1 Score | Precision | Recall |
| ---------------- | -------- | -------- | --------- | ------ |
| SVM (unweighted) | 68.1%    | 0.577    | 0.643     | 0.570  |
| SVM (weighted)   | 65.4%    | 0.600    | 0.597     | 0.625  |

Weighted SVM improved both F1 and Recall, showing better performance on underrepresented classes.

---

## Usage

1. Make sure you have Python 3.11 installed.
1. Install the required packages using pip:

```bash
pip install opencv-python numpy scikit-learn matplotlib scikit-image
```

## How to Run

1. Place aerial images into a structured directory.
2. Adjust paths and parameters in `Machine Learning.ipynb`.
3. Run all cells in Jupyter Notebook or via PyCharm.

---

## Notes

- **LBP descriptors** are sensitive to radius (`r`) and number of sampling points (`p`). Proper tuning of `r` and `p` greatly impacts classification performance.
- **Combining SIFT and LBP** enriches the feature representation by capturing both texture and local keypoints.
- **PCA** not only reduces dimensionality, but also improves generalization and speeds up training.
- Use **class-weighted SVM** to improve results on **imbalanced datasets**.
# Deep Learning Notebook
In this repo, we are exploring the classification of [Aerial Landscape Images](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset/data) using traditional machine learning and deep learning methods.
## Requirements
- `tensorflow` (Tested with 2.x)  
  *For model training, prediction, and preprocessing*
- `keras` (Integrated in TensorFlow 2.x)  
  *For model architecture and training*
- `keras-cv`  
  *For MixUp and CutMix data augmentation*
- `tensorflow-hub`  
  *For loading pretrained models*
- `classification-models` (`classification_models.tfkeras`)  
  *For ResNet18 and other pretrained backbones*
- `opencv-python` (`cv2`)  
  *For image processing and saliency map visualization*
- `scikit-learn`  
  *For metrics: accuracy, F1 score, recall, precision, confusion matrix*
- `matplotlib`  
  *For plotting training curves and visualizing results*
- `pandas`  
  *For data tabulation, label statistics, and result saving*
- `numpy`  
  *For numerical operations and image array handling*
- `os`, `shutil`, `random`, `collections.Counter`  
  *For file operations, random sampling, and class distribution counting*
## Model Description
There are two deep learning models:
- ResNet18
- VGG16

The performance of deep learning base models are shown below:

| Model     |  Accuracy | Recall | Precision| F1 score
|-----------|---------------------------------------------|----------|----------------|------------|------------|
| ResNet Baseline        |    0.93125   | 0.93125 | 0.93150     | 0.93124 |
| ResNet All MixUp        |   0.94542  |  0.94542|  0.94547    | 0.94526|
| ResNet Half MixUp        |  0.94458  | 0.94458| 0.94470     | 0.94452 |
| ResNet All CutMix        |   0.94667   | 0.94667 | 0.94692   | 0.94655 |
| ResNet Half CutMix        |    0.94375     | 0.94375 |0.94431 | 0.94381 |
| ResNet Half-Half         |    0.94750     |  0.94750|  0.94795    | 0.94748 |
| VGG Baseline     |  0.89958  | 0.89958       |0.90131     |0.89980|
| VGG All MixUp        |    0.91958             | 0.91958 |   0.92162   |0.91946  |
| VGG Half MixUp        |   0.90958  | 0.90958 |   0.92162   | 0.90950 |
| VGG All CutMix        |   0.90417   | 0.90417 |  0.91181    | 0.90359 |
| VGG Half CutMix        |     0.91000    | 0.91000 |   0.91291   | 0.90979 |
| VGG Half-Half         |    0.91292      | 0.91292 |  0.91378    |  0.91277|
| Imbalanced ResNet Baseline       |   0.91917   |    0.91917   | 0.92171     |0.91821 |
|ResNet18 (reweighted)| 0.91958| 0.91958| 0.92143|0.91911 |
| Imbalanced VGG Baseline     | 0.82917   | 0.82917     | 0.85109   | 0.82100|
| VGG16 (reweighted)|  0.86250|  0.86250 | 0.87013|  0.86161| 

## How to run
This contains two model file in separate Jupyter Notebooks:

- `resnet18.ipynb`: Implements the ResNet18-based classifier.
- `vgg16.ipynb`: Implements the VGG16-based classifier.
  
There are a few points to note：
- Each notebook is self-contained and includes a setup block that installs necessary dependencies via `pip`.
- Both models use the same segmented dataset. There is a cell in the vgg16.ipynb file that splits the dataset, and a cell in resnet18.ipynb that splits the imbalanced dataset.
- The save model method is used to save the trained model after each training and load the model later for subsequent operations.
- Since the code is run on colab, the path needs to be changed at runtime.

  
