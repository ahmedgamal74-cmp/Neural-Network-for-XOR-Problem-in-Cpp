# XOR Neural Network in C++ 

This project implements a simple **2-layer neural network** trained in **C++** to solve the classic **XOR classification** problem. It also includes a **Python-based visualization pipeline** that renders the learned decision boundary and training loss curve.

---

## 🧠 Project Overview

- **Language**: C++ (training), Python (visualization)
- **Task**: Classify XOR logic with a feedforward neural network
- **Architecture**:
  - Input layer: 2 neurons
  - Hidden layer: 2 neurons (with ReLU or Sigmoid activation)
  - Output layer: 1 neuron (Sigmoid)

---

## 🚀 Features

- **Fully Manual Neural Network in C++**
  - Matrix operations, activations, forward/backward pass, gradient descent
  - Training on XOR data from scratch with no external libraries
  - Supports different activations (Sigmoid, ReLU)

- **Logging**
  - Loss values logged to `log_loss.txt`
  - Final trained weights saved to `log_weights.txt` in a clean, parsable format

- **Python Visualizer**
  - Loads weights from training output
  - Generates:
    - Decision boundary visualization (`xor_boundary.png`)
    - Loss curve over epochs (`xor_loss.png`)
  - Highlights hidden layer neuron activations and their contribution to the final decision

---

## 📂 File Structure
```
NN_cpp/
│
├── main.cpp # Neural network training logic
├── myOps.cpp # Neural network operations and helpers
│
├── log_weights.txt # Output weights from C++ training
├── log_loss.txt # Training loss per epoch
│
├── logs_plot.py # Python script for visualizing boundary & loss
├── xor_boundary.png # Decision surface plot
├── xor_loss.png # Loss curve plot
│
└── build.sh # Build script using CMake
```
---

## 📦 Requirements

### C++
- CMake
- g++ / clang++

### Python (for visualization)
Install with:
```bash
pip install numpy matplotlib
```
## 🛠️ Usage

### 🔧 1. Build and Run C++ Code

```bash
bash build.sh      # or cmake && make
./NN_cpp           # runs training
```

- **Outputs:**
 - log_loss.txt with loss values
 - log_weights.txt with W1, b1, W2, b2

### 📊 2. Visualize Results in Python

```bash
python3 logs_plot.py
```
- **Generates:**

 - xor_boundary.png → shows classification surface and neuron boundaries

 - xor_loss.png → shows training convergence



## 📈 Example Output
- **🧠 A properly trained network will predict:**

```bash
Input: [0,0] → 0
Input: [0,1] → 1
Input: [1,0] → 1
Input: [1,1] → 0
```

- 🖼️ Decision boundary will show smooth, nonlinear contours separating the XOR classes.

## 🔍 Notes

- Training uses random initialization: each run can produce different decision boundaries

- You can adjust epochs, learning rate, or hidden activation in main.cpp

- The project is designed for educational clarity, not production performance

