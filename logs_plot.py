import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_weights(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = {}
    current = None
    rows = []

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if current and rows:
                data[current] = np.array(rows, dtype=float)
                rows = []
            current = line[2:].split()[0]
        elif line:
            rows.append([float(x) for x in line.split()])

    if current and rows:
        data[current] = np.array(rows, dtype=float)

    return data

def load_loss_log(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]

# Load
weights = load_weights("log_weights.txt")
W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
epochs, losses = load_loss_log("log_loss.txt")

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Grid for output surface
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                     np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Forward pass
z1 = np.dot(W1, grid.T) + b1
a1 = sigmoid(z1)
z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2).reshape(xx.shape)

# === Plot 1: Decision boundary ===
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, a2, levels=50, cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, a2, levels=[0.5], colors='k', linewidths=2)

# Hidden layer boundaries
for idx in range(W1.shape[0]):
    boundary = sigmoid(z1[idx]).reshape(xx.shape)
    plt.contour(xx, yy, boundary, levels=[0.5],
                linestyles='--', linewidths=1.5,
                colors=[f'C{idx+2}'])
    plt.plot([], [], color=f'C{idx+2}', linestyle='--', label=f"Neuron {idx+1} boundary")

# XOR inputs
for label in [0, 1]:
    plt.scatter(X[y == label, 0], X[y == label, 1],
                label=f"Class {label}", edgecolor='k', s=60)

plt.title("XOR Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("log_boundary.png", dpi=150)
print("✅ Saved decision boundary to xor_boundary.png")
plt.close()

# === Plot 2: Loss curve ===
plt.figure(figsize=(7, 5))
plt.plot(epochs, losses, label="Loss", color="black")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("log_loss.png", dpi=150)
print("✅ Saved loss plot to xor_loss.png")
plt.close()



# W1 = np.array([
#     [-5.107408,      3.798374] ,    
#     [2.782252,      -4.622981 ] 
# ])
# b1 = np.array([
#     # [1.142828],
#     # [-0.510048]
#     [-2.571419 ],     
#     [-1.666486]  
# ])

# W2 = np.array([
#     [7.921304,      8.367737]
# ])
# b2 = np.array([
#     [-4.097018],
  
# ])



