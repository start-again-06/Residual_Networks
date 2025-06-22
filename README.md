# 🧠 ResNet-50 from Scratch with TensorFlow/Keras

This repository implements a simplified ResNet-50 architecture using the TensorFlow/Keras Functional API, built and trained on a 6-class custom image dataset. It covers both the theoretical and practical aspects of deep residual learning.

📌 Key Highlights

✔️ Custom implementation of Residual Blocks:

Identity Block

Convolutional Block

✔️ Construction of the full ResNet-50 architecture

✔️ Includes:

Data preprocessing

Model training

Evaluation

Inference on custom images

✔️ Validated with predefined public test cases

📁 Project Structure

resnets_utils.py: Helper functions for dataset loading, preprocessing, and one-hot encoding.

test_utils.py / public_tests.py: Used to validate the correctness of identity/convolution blocks and ResNet-50 structure.

main notebook/script: Implements the ResNet-50 model using custom building blocks and runs training, evaluation, and prediction.

🧠 ResNet-50 Architecture Overview

Input image size: 64x64x3

Conv and max-pooling layers as initial stem

4 stages with bottleneck residual blocks:

Stage 1: 1 conv block + 2 identity blocks

Stage 2: 1 conv block + 3 identity blocks

Stage 3: 1 conv block + 5 identity blocks

Stage 4: 1 conv block + 2 identity blocks

Final classification via average pooling and fully connected softmax layer

📊 Dataset Information

6-class image classification problem

Dataset is split into training and testing sets

Images are normalized and labels one-hot encoded before training

🏋️ Model Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 32

Epochs: 10

Metrics: Accuracy

📈 Evaluation

After training, the model is evaluated on the test dataset to report:

Loss

Accuracy

You can also load a pre-trained .h5 model and evaluate it directly on the same test set.

🖼️ Predict Custom Image

The trained model supports predicting classes for custom input images. Input images must be resized to 64x64, preprocessed, and passed into the model for prediction.

🧪 Testing & Validation

Residual blocks and network architecture are verified using public test cases.

Structural comparisons are done via string summaries of the layer configurations.

🖼️ Visualization

Training loss and accuracy can be plotted using Matplotlib.

Model architecture visualization available using tools like:

model.summary()

plot_model() (optional)

model_to_dot() (optional)

📚 References

Deep Residual Learning for Image Recognition – He et al.

📜 License

This project is licensed under the MIT License and is intended for educational use.
