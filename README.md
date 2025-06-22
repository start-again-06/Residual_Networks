🧠 ResNet-50 from Scratch with TensorFlow/Keras
This repository demonstrates how to build the ResNet-50 architecture from the ground up using TensorFlow and Keras Functional API. It walks through every component, from custom residual blocks to end-to-end model training, validation, and inference.

🚀 What You'll Learn
🔨 How to construct residual blocks:

Identity Block

Convolutional Block

🧱 How to stack these blocks to build a full ResNet-50 architecture

🧪 How to verify correctness using public test cases

🧼 How to preprocess image data and perform one-hot encoding

📊 How to train, evaluate, and predict using the model

🧬 Architecture Breakdown
The network follows a simplified ResNet-50 layout:

Input: 64x64 RGB images

Initial Stem: Convolution → BatchNorm → ReLU → MaxPooling

Residual Stages:

Stage 1: 1 conv block + 2 identity blocks

Stage 2: 1 conv block + 3 identity blocks

Stage 3: 1 conv block + 5 identity blocks

Stage 4: 1 conv block + 2 identity blocks

Output: Average Pooling → Flatten → Dense → Softmax

📁 Repository Structure
resnets_utils.py – Utilities for data loading and one-hot encoding

test_utils.py / public_tests.py – Unit tests for building blocks

outputs.py – Reference outputs for test case comparison

main.ipynb or script – Full model definition and training pipeline

📦 Dataset Overview
6-class custom image classification dataset

Normalized input images ([0, 1] range)

Labels converted to one-hot vectors

Split into training and testing subsets

🏋️‍♂️ Training Configuration
Parameter	Value
Optimizer	Adam
Loss Function	Categorical Crossentropy
Batch Size	32
Epochs	10
Metrics	Accuracy

🔍 Model Evaluation
After training, the model is evaluated on a held-out test set to measure:

✅ Loss

✅ Accuracy

A pre-trained model (.h5 file) can be loaded to skip training and go directly to evaluation.

🖼️ Predict on Custom Images
Easily test your trained model on your own image files:

Resize to 64x64

Normalize pixel values

Run prediction using the .predict() method

Output is a 6-dimensional softmax vector

🧪 Testing & Validation
To ensure correctness, the model passes:

✅ Public unit tests for residual blocks

✅ Shape and value checks against known outputs

✅ Architecture comparison via layer summaries

📊 Visualization Tools
🖼️ Visualize training curves using Matplotlib

📋 View network summary with model.summary()

🔄 Graph architecture with:

plot_model()

model_to_dot() (SVG support)

📚 References
He et al., Deep Residual Learning for Image Recognition
