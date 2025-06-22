# 🧠 ResNet-50 from Scratch with TensorFlow/Keras

This project demonstrates how to build the **ResNet-50 architecture** entirely from scratch using TensorFlow 2.x and Keras. It covers every major component, including residual blocks, architecture design, training routines, and evaluation strategies.

---

## 📌 Overview

- **Framework**: TensorFlow 2.x with Keras
- **Model**: ResNet-50 (50-layer Residual Network)
- **Architecture Type**: Deep Residual CNN with Identity & Convolutional Blocks
- **Dataset**: Custom image dataset with 6 classes
- **Input Shape**: (64, 64, 3)
- **Output**: Softmax over 6 classes

---

## 🧱 Model Architecture

The ResNet-50 architecture contains the following key components:

- **Initial Layers**:
  - Zero-padding
  - Convolution (7×7), BatchNorm, ReLU
  - MaxPooling

- **4 Main Stages**:
  - **Stage 1**: 1 convolutional block + 2 identity blocks
  - **Stage 2**: 1 convolutional block + 3 identity blocks
  - **Stage 3**: 1 convolutional block + 5 identity blocks
  - **Stage 4**: 1 convolutional block + 2 identity blocks

- **Final Layers**:
  - AveragePooling
  - Fully Connected Dense Layer (Softmax)

Each block uses skip connections to allow gradient flow during training.

---

## 📂 Project Structure

```
├── resnets_utils.py         # Data preprocessing & helper functions
├── test_utils.py            # Summary and tensor comparison tools
├── public_tests.py          # Unit tests for identity & conv blocks
├── outputs.py               # Stored reference tensors
├── images/                  # Custom images for prediction
└── main_script.py / notebook.ipynb
```

---

## 🧪 Training Setup

| Parameter         | Value                    |
|------------------|--------------------------|
| Optimizer        | Adam                     |
| Loss Function    | Categorical Crossentropy |
| Epochs           | 10                       |
| Batch Size       | 32                       |
| Metric           | Accuracy                 |

Training is performed on normalized data (divided by 255), with labels one-hot encoded.

---

## 📈 Evaluation Strategy

- Performance is evaluated using `model.evaluate()`
- The model can also be tested using a pre-trained `.h5` file
- Results are printed with loss and accuracy

---

## 🖼️ Image Prediction Flow

To make predictions on new images:

1. Load the image and resize to 64x64
2. Convert to array and normalize
3. Use the trained model to predict the class
4. Output the class index and softmax vector

---

## ✅ Testing & Validation

- Unit tests are provided for both identity and convolutional blocks
- The model’s architecture is validated using shape summaries
- The final model structure is compared with reference outputs

---

## 🔍 Visualization

Use Keras utilities to generate architecture plots:

- Visualize the model with `plot_model()`

## 📚 References
- [Deep Residual Learning for Image Recognition – He et al.](https://arxiv.org/abs/1512.03385)
