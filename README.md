🧠 ResNet-50 from Scratch with TensorFlow/Keras
This project demonstrates how to build the ResNet-50 architecture entirely from scratch using TensorFlow 2.x and Keras. It covers every major component, including residual blocks, architecture design, training routines, and evaluation strategies.

📌 Overview
Framework: TensorFlow 2.x with Keras

Model: ResNet-50 (50-layer Residual Network)

Architecture Type: Deep Residual CNN with Identity & Convolutional Blocks

Dataset: Custom image dataset with 6 classes

Input Shape: (64, 64, 3)

Output: Softmax over 6 classes

🧱 Model Architecture
The ResNet-50 architecture contains the following key components:

Initial Layers
Zero-padding

Convolution (7×7), BatchNorm, ReLU

MaxPooling

Residual Stages
Stage 1: 1 convolutional block + 2 identity blocks

Stage 2: 1 convolutional block + 3 identity blocks

Stage 3: 1 convolutional block + 5 identity blocks

Stage 4: 1 convolutional block + 2 identity blocks

Final Layers
Average Pooling

Fully Connected Dense Layer with Softmax activation

Each block uses skip connections to preserve gradient flow and combat vanishing gradients.

📂 Project Structure
bash
Copy
Edit
├── resnets_utils.py         # Data preprocessing & helper functions
├── test_utils.py            # Summary and tensor comparison tools
├── public_tests.py          # Unit tests for identity & conv blocks
├── outputs.py               # Stored reference tensors
├── images/                  # Custom images for prediction
└── main_script.py / notebook.ipynb
🧪 Training Setup
Parameter	Value
Optimizer	Adam
Loss Function	Categorical Crossentropy
Epochs	10
Batch Size	32
Metric	Accuracy

Training is performed on normalized images (pixel values divided by 255).

Labels are one-hot encoded for multiclass classification.

📈 Evaluation Strategy
Use model.evaluate() to compute loss and accuracy.

Optionally, load a pre-trained .h5 model for inference.

Results include loss value and test accuracy.

🖼️ Image Prediction Flow
To classify new images:

Load and resize image to 64x64

Convert to NumPy array and normalize

Predict using the trained model

Output class probabilities and predicted label

✅ Testing & Validation
Identity and convolutional blocks are validated using public_tests.py

Output tensors are compared with reference values

Full model structure is verified with model.summary()

🔍 Visualization
Generate architectural diagrams with plot_model()

View individual layers and shapes using model.summary()

📚 References
He, K., Zhang, X., Ren, S., & Sun, J. (2015).
Deep Residual Learning for Image Recognition
