# ResNet-50 from Scratch with TensorFlow/Keras

A comprehensive deep learning project that implements the **ResNet-50 architecture entirely from scratch** using TensorFlow 2.x and Keras. The project emphasizes residual learning by explicitly building identity and convolutional blocks, assembling the full network, and training it on a custom multi-class image dataset.

---

## Features
- Complete ResNet-50 implementation from first principles  
- Identity and convolutional residual blocks with skip connections  
- Deep residual CNN using Keras Functional API  
- End-to-end pipeline: preprocessing, training, evaluation, inference  
- Model architecture visualization using `plot_model()`  
- Support for custom multi-class image datasets  

---

## Dataset
**Custom Image Dataset (6 Classes)**

**Description**  
A labeled RGB image dataset designed for supervised multi-class classification.

**Input Shape**  
64 × 64 × 3

**Output Labels**  
Six classes represented using one-hot encoding

**Usage**  
Images are normalized to the range [0, 1] and labels are converted to categorical format before training.

---

## System Design
The system follows a modular deep learning pipeline aligned with the original ResNet-50 design. Each architectural component is implemented as a reusable function to improve clarity, extensibility, and reproducibility.

### High-Level Architecture

| Stage | Description |
|------|------------|
| Input Images | 64×64 RGB images |
| Preprocessing | Normalization and one-hot encoding |
| Initial Layers | Conv (7×7), BatchNorm, ReLU, MaxPool |
| Residual Stages | Identity & Convolutional Blocks |
| Global Pooling | Average Pooling |
| Classifier | Fully connected Softmax layer |
| Output Layer | 6-class probability distribution |
| Visualization | Model plots and summaries |

---

## Model Architecture
**Input Layer:** (64, 64, 3)

### Initial Block
- Zero Padding  
- 7×7 Convolution  
- Batch Normalization  
- ReLU Activation  
- Max Pooling  

### Residual Stages
- Stage 1: 1 convolutional block + 2 identity blocks  
- Stage 2: 1 convolutional block + 3 identity blocks  
- Stage 3: 1 convolutional block + 5 identity blocks  
- Stage 4: 1 convolutional block + 2 identity blocks  

### Final Layers
- Average Pooling  
- Fully Connected Dense Layer (Softmax)

**Framework:** TensorFlow 2.x with Keras  
**Loss Function:** Categorical Crossentropy  
**Task:** Multi-class Image Classification  

The model is implemented using the **Keras Functional API** with explicit skip connections to preserve gradient flow in deep networks.

---

## Dataset Preprocessing
- Images resized to 64×64  
- Pixel values normalized to range [0, 1]  
- Labels converted to one-hot encoded vectors  
- Data cast to `float32` for TensorFlow compatibility  

---

## Training Pipeline
- Load and preprocess the image dataset  
- Construct ResNet-50 architecture from scratch  
- Compile the model with Adam optimizer  
- Train for 10 epochs with batch size 32  
- Track categorical accuracy during training  
- Evaluate performance on validation/test data  

---

## Evaluation Strategy
- Performance evaluated using `model.evaluate()`  
- Loss and accuracy printed after training  
- Supports testing using a saved `.h5` model file  
- Architecture validated using shape summaries  

---

## Inference on Custom Images
- Place images in the `images/` directory  
- Resize images to 64×64  
- Normalize pixel values  
- Run model prediction  
- Output predicted class index and softmax probabilities  

---

## Testing & Validation
- Unit tests for identity and convolutional blocks  
- Tensor output comparisons with reference values  
- Model architecture validation using summaries  
- Verification of residual connections  

---


---

## Design Principles
- Faithful reproduction of the original ResNet-50 architecture  
- Explicit residual learning for interpretability  
- Modular and reusable block-based design  
- Educational focus on deep CNN fundamentals  

---

## Dependencies
- Python 3.7+  
- TensorFlow 2.x / Keras  
- NumPy  
- Matplotlib  
- pydot  
- graphviz  

---

## Model Visualization
- Architecture plot generated using `plot_model()`  

---

## References
- Deep Residual Learning for Image Recognition – He et al.

---

## License
This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.
