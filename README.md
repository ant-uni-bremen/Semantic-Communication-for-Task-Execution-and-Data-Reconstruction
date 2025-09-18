# Full Data Recovery after Task-specific Semantic Communication

This repository implements a hybrid semantic communication framework that enables both semantic classification and high-fidelity image reconstruction in distributed multi-agent systems using neural network-based encoders and decoders.

## 📚 Overview

Traditional communication systems aim to deliver raw data with high fidelity, which is often inefficient in bandwidth-constrained environments. Semantic communication (SemCom) shifts focus from data transmission to meaning transmission. This project explores a dual-task SemCom system that enables:

- **Task-specific semantic classification**
- **Accurate reconstruction of original data from semantic features**

The system is implemented and evaluated using encoder-decoder models built on:
- **CNN**
- **ResNet14 / ResNet20**
- **Vision Transformers (ViT)**

## 🧠 Key Concepts

- **Variational Autoencoders (VAEs)**: Used to encode data into compact latent representations.
- **Semantic Communication**: Transmits only task-relevant features to save bandwidth while retaining semantic meaning.
- **Infomax Principle**: Optimizes the mutual information between received representations and original inputs/labels.
- **Weighted Loss Functions**: Combines MSE, SSIM, and CE to balance classification and reconstruction performance.

## 🛠️ Features

- End-to-end encoder-decoder training for classification and reconstruction.
- Supports distributed image segmentation and multi-agent communication.
- Flexible architecture for CNN, ResNet, and ViT backbones.
- Incorporates perceptual loss using Structural Similarity Index Measure (SSIM).
- Simulates noisy transmission using AWGN channel models.

## 🏗️ Architecture

The model operates in a distributed setting where:
![Screenshot from 2025-06-03 19-57-27](https://github.com/user-attachments/assets/4b1b7fb0-d685-4b0c-976a-dd0a44d8d081)

1. An image is split into `N_split` segments.
2. Each segment is processed by an independent encoder (CNN/ResNet/ViT).
3. Encoded features are transmitted over a noisy channel.
4. A decoder performs:
   - **Classification** using a softmax layer
   - **Reconstruction** using transposed convolutions or residual decoders

## 🧪 Evaluation

### Reconstruction Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- MSE (Mean Squared Error)

### Classification Metrics:
- Accuracy

## 🔧 Training Details

- **Dataset**: CIFAR-10 (32×32×3 images)
- **Adjustable hyperparameters**:
  - `α`: Weight between reconstruction and classification loss
  - `β`: SSIM scaling factor
  - Encoder channel size (`N_Tx`)
  - Number of agents/splits (`N_splits`)

## 📄 Thesis Reference

This repository is based on the Master’s Thesis:

**<h3>Full Data Recovery after Task-specific Semantic Communication</h3>**  
**Author**: Avinash Kankari  
**Supervisor**: Maximilian Tillmann, M.Sc.  
**Institution**: Department of Communications Engineering, University of Bremen  
**Date**: June 3, 2025

## 📦 Python Dependencies
- Used Python version 3.10
- Packages List : 
  ```bash
    pip install -r requirements.txt
    ```
    

## Code Execution
- config.json is used handle parameters and selection of models
- main.py is first python executable file to run program based on configurations updated in config.json
