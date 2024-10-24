# Multi-Layer Perceptron from Scratch for Digit Classification

This project implements a **Multi-Layer Perceptron (MLP)** from scratch to classify digits using the **MNIST** dataset. It explores various optimization techniques, including **SGD (Stochastic Gradient Descent)**, **Momentum**, **Adam**, **RMSProp**, and **Adagrad**, to improve training and test performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Optimization Techniques](#optimization-techniques)
- [Results](#results)
- [Installation](#installation)
- [Contributing](#contributing)

## Project Overview
The objective of this project is to build a **Multi-Layer Perceptron (MLP)** neural network to classify handwritten digits (0-9) from the **MNIST** dataset. The network is implemented from scratch using Python, without high-level deep learning libraries like TensorFlow or PyTorch. Different optimization algorithms are tested to improve convergence speed and accuracy.

## Dataset
The **MNIST** dataset consists of 70,000 grayscale images of handwritten digits, with:
- **Training set**: 60,000 samples
- **Test set**: 10,000 samples
Each image is **28x28 pixels**, flattened to a vector of **784 features**.

## Model Architecture
The MLP model has the following architecture:

- **Input layer**: 784 neurons (one for each pixel)
- **Hidden layer**: A single hidden layer with configurable size and **ReLU** activation
- **Output layer**: 10 neurons (one for each digit) with **Softmax** activation

### Layers:
1. **Input layer**: 784 neurons
2. **Hidden layer**: Variable size, **ReLU** activation
3. **Output layer**: 10 neurons, **Softmax** activation

### Key Techniques:
- **Weight Initialization**: Xavier initialization for better convergence
- **Activation Function**: **ReLU** for the hidden layer, **Softmax** for the output layer
- **Loss Function**: Categorical Cross-Entropy
- **Backpropagation**: Implemented to update weights during training

## Optimization Techniques
Optimization plays a crucial role in improving the model's convergence and performance. In this project, the following optimization algorithms are explored:

1. **Stochastic Gradient Descent (SGD)**:  
   - A simple yet effective optimization algorithm where the model's parameters are updated by calculating gradients on mini-batches. While it can be slow due to noisy gradient estimates, it's often effective with appropriate learning rate tuning.
   - **Test Accuracy**: 92.0%

2. **SGD with Momentum**:  
   - Introduces a momentum term that adds inertia to the parameter updates, which helps the optimizer overcome small local minima and converge faster. Momentum helps accelerate gradients vectors in the correct direction, leading to faster convergence.
   - **Test Accuracy**: 94.1%

3. **Adam (Adaptive Moment Estimation)**:  
   - A popular optimizer that combines the benefits of both **Momentum** and **RMSProp**. Adam maintains per-parameter learning rates that are adapted based on first (mean) and second (variance) moments of the gradients. This leads to faster convergence and often better performance on large datasets.
   - **Test Accuracy**: 97.3%

4. **RMSProp (Root Mean Square Propagation)**:  
   - Similar to Adam, **RMSProp** keeps a moving average of squared gradients to normalize the gradient update. It works particularly well for non-stationary objectives and improves stability in training.
   - **Test Accuracy**: 96.5%

5. **Adagrad (Adaptive Gradient Algorithm)**:  
   - Adagrad adapts the learning rate for each parameter based on the frequency of updates. While it performs well on sparse data, its accumulation of squared gradients can lead to overly small learning rates in the later stages of training, causing premature convergence.
   - **Test Accuracy**: 93.2%

## Results
After experimenting with the different optimizers, the following accuracy results were obtained on the MNIST test set:

| Optimizer  | Test Accuracy |
|------------|---------------|
| SGD        | 92.0%         |
| Momentum   | 94.1%         |
| Adam       | 97.3%         |
| RMSProp    | 96.5%         |
| Adagrad    | 93.2%         |

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sathish0552/MLP-Digit-Classification.git
2. Navigate to the project directory:
    ```bash
    cd MLP-Digit-Classification
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Contributing
Feel free to open issues or submit pull requests if you have suggestions for improving this project!

