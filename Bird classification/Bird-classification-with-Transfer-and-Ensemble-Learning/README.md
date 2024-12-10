# Bird Species Classification using Transfer and Ensemble Learning Models

This repository contains the implementation and analysis of various machine learning strategies for bird species classification. The project is part of **COS30082 - Applied Machine Learning** at Swinburne University of Technology. It uses the **Caltech UCSD-200 Birds dataset (CUB-200)**, focusing on mitigating overfitting through custom hybrid learning strategies and ensemble learning techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Processing](#data-processing)
  - [Models Implemented](#models-implemented)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Links to Notebooks](#links-to-notebooks)

## Project Overview

Bird species classification is a challenging task due to the limited size of the CUB-200 dataset. This project leverages advanced neural network architectures like **ResNet50**, **MobileNetV2**, and **EfficientNetB0**, combined with data augmentation and ensemble techniques, to improve accuracy and reduce overfitting.

### Objectives

1. Mitigate overfitting using data augmentation and regularization techniques.
2. Implement a custom hybrid learning strategy using transfer learning.
3. Develop and evaluate ensemble learning models to combine the strengths of different architectures.
4. Compare the performance of different strategies and analyze their effectiveness.

## Dataset

The project uses the **Caltech UCSD-200 Birds Dataset (CUB-200)**:
- **Training images**: 4829
- **Testing images**: 1204
- **Classes**: 200 (each with ~20-30 images)

Dataset link: [CUB-200 on Kaggle](https://www.kaggle.com/datasets/shmaker/ucsd-200)

## Methodology

### Data Processing

- **Augmentation**:
  - Rescaling (1/255 normalization)
  - Shear Transformation (0.2 range)
  - Zoom (0.2 range)
  - Horizontal Flip
  - Rotation (up to 20°)
  - Width and Height Shifts (up to 20%)
- Images resized to 224x224 pixels for compatibility with pre-trained models.
- Data generators created using `flow_from_dataframe`.

### Models Implemented

#### 1. Custom Hybrid Learning Strategy
- Base model: **ResNet50** with pre-trained ImageNet weights.
- Fine-tuned with additional layers:
  - Global Average Pooling
  - Dropout (0.25 and later increased to 0.4)
  - Dense layers with L2 regularization.
- Optimizer: Adam with learning rate scheduling (`ReduceLROnPlateau`).
- Constraints: Gradient clipping and label smoothing.

#### 2. Ensemble Learning
- Models used: ResNet50, MobileNetV2, EfficientNetB0.
- Strategies:
  - **Concatenate-Flatten**: Combine feature maps, followed by Dense layers.
  - **Concatenate-GlobalAveragePooling2D**: Apply pooling before concatenation.
  - **Soft Voting**: Average predictions of individual models.

## Results

### Custom Hybrid Learning
- Improved convergence through controlled learning rates.
- Mitigated overfitting using Dropout and regularization.

### Ensemble Learning
- Achieved better robustness but required significant computational resources.
- `Concatenate-GlobalAveragePooling2D` achieved the best validation loss.

## Acknowledgements

- [Limteckping45's GitHub Repository](https://github.com/Limteckping45/Cos30082_Assignment/blob/main/Bird_Classifier_Training_Model.ipynb)
- Dataset courtesy of [Shmaker on Kaggle](https://www.kaggle.com/datasets/shmaker/ucsd-200).

## Links to Notebooks

- [Custom Hybrid Learning Strategy](https://www.kaggle.com/code/nguyentuanducfswhn/noob-custom-hybrid-learning-for-cub-200)
- [Ensemble Flatten](https://www.kaggle.com/code/binhswinburnehn/asm-1-ensemble-flatten)
- [Ensemble Global Average Pooling 2D](https://www.kaggle.com/code/nguyentuanducfswhn/asm-1-ensemble-global-average-pooling-2d)
- [Soft Voting](https://www.kaggle.com/code/sntrnmnh/asm-1-soft-voting-incomplete)
