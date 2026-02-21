# 🚀 Lightweight CNN for Image Classification (CIFAR-10)

This repository contains the solution for an image classification task using the CIFAR-10 dataset. The primary challenge was to design a highly efficient Convolutional Neural Network (CNN) with **fewer than 400,000 parameters** while achieving maximum possible accuracy.

## 🎯 Task Objectives
- **Dataset:** CIFAR-10 (60,000 images, 32x32 pixels, 10 classes).
- **Data Split:** 45,000 training, 5,000 validation (strictly the last 5,000 of the training set), and 10,000 testing images.
- **Constraint:** Network architecture must have `< 400k parameters`.
- **Framework:** TensorFlow / Keras (Google Colab).

---

## 🛠️ Step-by-Step Approach & Methodology

### 1. Initial Attempt: A Standard CNN
Initially, a standard CNN architecture with standard `Conv2D` layers, `MaxPooling`, and a large `Dense` layer at the end was implemented.
*   **Result:** It achieved around **81% accuracy**. However, the training loss graph showed severe fluctuations (dissonance) indicating instability in the learning process, and the parameter count was close to the limit.

### 2. Solving Instability & Overfitting
To address the highly fluctuating validation loss, two major techniques were introduced:
*   **Data Augmentation:** Added random rotations, width/height shifts, and horizontal flips to prevent the model from memorizing the training data.
*   **Learning Rate Scheduler (`ReduceLROnPlateau`):** Implemented a callback to dynamically reduce the learning rate whenever the validation loss stopped improving. This immediately smoothed out the loss curve.
*   **Result:** Accuracy dropped slightly to **~79.8%**, but the model became highly stable (eliminated overfitting). The model was "underfitting" due to heavy regularization.

### 3. Final Architecture: The "Compact SE-ResNet" (The Breakthrough)
To push the accuracy past the 90% boundary while strictly adhering to the `< 400k parameter` constraint, advanced techniques inspired by modern architectures (ResNet, MobileNet, EfficientNet) were integrated:

1.  **Separable Convolutions:** Replaced standard `Conv2D` with `SeparableConv2D`. This drastically reduced the number of parameters (by almost 10x per layer), allowing the network to become much deeper without breaking the 400k limit.
2.  **Residual Connections (Skip Connections):** Added ResNet-style shortcuts to prevent the vanishing gradient problem in the deeper network.
3.  **Squeeze-and-Excitation (SE) Blocks:** Added attention mechanisms (SE-Net) to help the network focus on the most important features of an image while adding negligible parameters.
4.  **Swish Activation:** Replaced standard ReLU with the Swish activation function for smoother gradient flow.
5.  **Global Average Pooling:** Used `GlobalAveragePooling2D` instead of `Flatten` before the final output layer to completely eliminate the parameter explosion typical in fully connected layers.

---

## 📊 Final Results

Our final `Compact SE-ResNet` achieved exceptional performance on the completely unseen test set:

- **Total Parameters:** ~392,000 (Successfully under the 400k limit)
- **Final Test Accuracy:** **90.33%**
- **Final Test Loss:** ~0.33

This result is highly competitive for a lightweight model trained from scratch on CIFAR-10.

## 💻 How to Run
The complete code is available in the `CIFAR10_Lightweight_CNN.ipynb` notebook. It is designed to be run seamlessly on Google Colab.

1. Open the notebook in Google Colab.
2. Ensure the runtime type is set to GPU.
3. Run all cells sequentially. The dataset will be downloaded, the model will be constructed, and training will begin automatically.
