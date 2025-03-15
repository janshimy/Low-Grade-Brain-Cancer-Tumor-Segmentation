# Low-Grade Glioma Segmentation

*An end-to-end deep learning project for segmenting low-grade glioma in brain MRI images*

---

## Table of Contents

- [Overview](#overview)
- [Business Understanding](#business-understanding)
  - [What is Low-Grade Glioma?](#what-is-low-grade-glioma)
  - [Why Use Deep Learning?](#why-use-deep-learning)
- [Data Understanding & Preparation](#data-understanding--preparation)
- [Visualization](#visualization)
  - [Data Distribution](#data-distribution)
  - [MRI Image Visualization](#mri-image-visualization)
- [Data Augmentation](#data-augmentation)
- [Modeling](#modeling)
  - [Vanilla U-Net Architecture](#vanilla-u-net-architecture)
  - [Feature Pyramid Network (FPN)](#feature-pyramid-network-fpn)
  - [U-Net with ResNeXt Backbone](#u-net-with-resnext-backbone)
  - [Segmentation Metrics & Loss](#segmentation-metrics--loss)
- [Evaluation](#evaluation)
- [References](#references)

---

## Overview

This passion project focuses on developing an automated segmentation framework for low-grade glioma using deep learning. By leveraging advanced neural network architectures and extensive data augmentation techniques, the project aims to accurately delineate tumor regions in brain MRI scans to support early diagnosis and treatment planning.

---

## Business Understanding

### What is Low-Grade Glioma?

Low-grade gliomas are brain tumors (WHO grade II/III) originating from glial cells. Although generally benign, they can cause significant neurological symptoms as they grow. Treatment options include surgery, radiation, or careful monitoring, given the risk of progression to higher-grade tumors.

### Why Use Deep Learning?

Traditional segmentation methods are manual, time-consuming, and subject to observer variability. Deep learning provides:
- **Precision:** Accurate identification of tumor boundaries.
- **Efficiency:** Faster, automated segmentation.
- **Consistency:** Reduced subjectivity in diagnoses.
- **Adaptability:** Robustness even for infiltrative and ambiguous tumor regions.

Explore further details in the [Jupyter Notebook](https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/low-grade-glioma-brain-tumor-segmentation.ipynb).

---

## Data Understanding & Preparation

### Data Overview

The dataset is sourced from The Cancer Imaging Archive (TCIA) and includes:
- **MRI Images:** Brain MRI scans with manual FLAIR abnormality segmentation masks.
- **Patient Data:** A table with 110 rows and 18 columns covering demographics, tumor characteristics, and genomic data.

### Sample Images

Below are sample images comparing cases without and with segmentation:

<table>
  <tr>
    <th>Low Grade Glioma Without Segmentation</th>
    <th>Low Grade Glioma With Segmentation</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/low-grade-glioma-grade-1-without-segmentation.png?raw=true" alt="Without Segmentation" width="300">
      <br><em>This image belongs to a 24-year-old man who developed focal seizures affecting the left side of his body.</em>
    </td>
    <td align="center">
      <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/low-grade-glioma-grade-1-with-segmentation.png?raw=true" alt="With Segmentation" width="300">
      <br><em>This image belongs to a 24-year-old man who developed focal seizures affecting the left side of his body. The red area indicates low-grade glioma.</em>
    </td>
  </tr>
</table>

### Data Preparation Images

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/Information%20about%20csv%20dataset.png?raw=true" alt="CSV Dataset Information" width="600">
  <br><em>Figure 3 – Information about CSV dataset</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/the%20head%20of%20the%20csv%20dataset.png?raw=true" alt="CSV Head" width="600">
  <br><em>Figure 4 – Head of the CSV dataset</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/the%20final%20dataframe%20to%20be%20used%20in%20the%20visualization%20and%20modeling%20part.png?raw=true" alt="Final Dataframe" width="600">
  <br><em>Figure 5 – Final dataframe for visualization and modeling</em>
</p>

---

## Visualization

### Data Distribution

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/Distribution%20of%20data%20grouped%20by%20diagnosis.png?raw=true" alt="Distribution by Diagnosis" width="600">
  <br><em>Figure 6 – Distribution of data grouped by diagnosis</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/Distribution%20of%20data%20grouped%20by%20patient%20and%20diagnosis.png?raw=true" alt="Distribution by Patient and Diagnosis" width="600">
  <br><em>Figure 7 – Distribution of data grouped by patient and diagnosis</em>
</p>

### MRI Image Visualization

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/Low%20Grade%20Glioma%20Detection%20on%20Brain%20MRI%20Images%20with%20original%20color%20and%20hot%20colormap.png?raw=true" alt="MRI Visualization" width="600">
  <br><em>Figure 8 – Low Grade Glioma Detection on Brain MRI Images using original color and hot colormap</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/Tumor%20location%20is%20show%20as%20segmented%20on%20one%20Brain%20MRI.png?raw=true" alt="Tumor Segmentation" width="600">
  <br><em>Figure 9 – Tumor location as segmented on a Brain MRI image</em>
</p>

---

## Data Augmentation

Using the Albumentations library, three augmentation techniques are applied (elastic deformation, grid distortion, optical distortion) to increase dataset diversity.

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/data_augmentation_mri_images.png?raw=true" alt="Augmented MRI Images" width="600">
  <br><em>Figure 10 – Augmented Brain MRI Images</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/data_augmentation_mask_images.png?raw=true" alt="Augmented Mask Images" width="600">
  <br><em>Figure 11 – Augmented Mask Images</em>
</p>

---

## Modeling

The project explores several architectures for segmentation:

### Vanilla U-Net Architecture

A classic encoder-decoder structure with skip connections is implemented (U-Net-35 with 15 convolutional layers, 14 ReLU activations, 3 max-pooling, and 3 upsampling layers).

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/u-net-architecture.png?raw=true" alt="U-Net Architecture" width="600">
  <br><em>Figure 12 – U-Net Architecture from “U-Net: Convolutional Networks for Biomedical Image Segmentation”</em>
</p>

### Feature Pyramid Network (FPN)

FPN combines multi-scale features via bottom-up, top-down, and lateral connections.

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/feature_pyramid_network.png?raw=true" alt="FPN Architecture" width="600">
  <br><em>Figure 13 – Feature Pyramid Network Architecture</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/feature_pyramid_network2.png?raw=true" alt="FPN Layers Explanation" width="600">
  <br><em>Figure 14 – Explanation of Bottom-Up, Top-Down, and Lateral Layers</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/feature_pyramid_network3.png?raw=true" alt="FPN Merging with U-Net" width="600">
  <br><em>Figure 15 – Merging the FPN Last Layers with U-Net Architecture</em>
</p>

### U-Net with ResNeXt Backbone

This variant leverages a pre-trained ResNeXt50 encoder (ResNext50-32x4d) combined with additional down-sampling and up-sampling layers for improved feature extraction.  
For a detailed explanation, see the [ResNeXt-50 Backbone (Turkish)](https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/resnext50_t%C3%BCrk%C3%A7e.ipynb).

### Segmentation Metrics & Loss

- **Dice Coefficient:**  
  $$\text{Dice} = \frac{2 \times |\text{Prediction} \cap \text{Ground Truth}|}{|\text{Prediction}| + |\text{Ground Truth}|}$$  
  A value of 1 indicates perfect overlap.
- **Loss Functions:**  
  Dice Loss combined with Binary Cross-Entropy (BCE) loss is used for training.

---

## Evaluation

The models were evaluated on training, validation, and test data:

- **U-Net:** Mean DICE ~83% on test data.
- **FPN:** Mean DICE ~78% on test data.
- **U-Net with ResNeXt50 Backbone:** Mean DICE ~89% on test data.

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/vanilla%20unet%20model%20history.png?raw=true" alt="U-Net Training History" width="600">
  <br><em>Figure 16 – Epoch vs. DICE with U-Net Architecture</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/fpn%20moodel%20history.png?raw=true" alt="FPN Training History" width="600">
  <br><em>Figure 17 – Epoch vs. DICE with FPN Architecture</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/resnext50%20model%20history.png?raw=true" alt="ResNeXt50 Training History" width="600">
  <br><em>Figure 18 – Epoch vs. DICE with U-Net ResNeXt50 Architecture</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/prediction%20mask%20image%20without%20threshold%20and%20with%20threshold.png?raw=true" alt="Prediction Masks" width="600">
  <br><em>Figure 19 – Prediction Mask Image without Threshold and with Threshold</em>
</p>

<p align="center">
  <img src="https://github.com/janshimy/Low-Grade-Glioma-Brain-Tumor-Segmentation/blob/main/Pictures/prediction%20of%20resnext50.gif?raw=true" alt="Prediction vs. Ground Truth" width="600">
  <br><em>Figure 20 – Prediction and Ground Truth Masks on Brain MRI Images with ResNeXt50 Backbone</em>
</p>

---

## References

### Websites
1. [TCIA - LGG-1p19qDeletion](https://wiki.cancerimagingarchive.net/display/Public/LGG-1p19qDeletion)
2. [Deep Learning based Brain Segmentation (Mazurowski)](https://github.com/MaciejMazurowski/brain-segmentation)
3. [Stack Overflow – Change Image Color Using a Mask](https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask)
4. [Kaggle – Brain MRI Data Visualization (Bonhart)](https://www.kaggle.com/code/bonhart/brain-mri-data-visualization-unet-fpn)
5. [Kaggle – Brain MRI Segmentation using ResUNet](https://www.kaggle.com/code/anantgupt/brain-mri-detection-segmentation-resunet)

### Blog Posts & Slides
1. [Review: Feature Pyramid Networks](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)
2. [Instance and Semantic Segmentation Architecture](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
3. [Understanding FPN for Object Detection](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)

### Articles
1. Khan, M. B., et al. “Automatic Segmentation and Shape, Texture-based Analysis of Glioma
