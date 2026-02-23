
### TensorFlow From Scratch (PyTorch-Based Educational Implementation)

This repository contains a from-scratch educational implementation of core TensorFlow-like concepts, built using low-level tensor operations and PyTorch primitives for demonstration purposes.
The goal of this project is to understand how deep learning frameworks work internally — including:

 - Tensor operations
 - Automatic differentiation (autograd)
 - GPU acceleration (CUDA)
 - Neural network modules
 - Loss functions
 - Optimizers
 - Dataset & DataLoader handling
 - Training loops

This project is designed for learning and experimentation — not production use.
**Overview**

Modern deep learning frameworks like TensorFlow and PyTorch abstract many complex components. This repository breaks those abstractions down to explore:
	 - How tensors are created and manipulated
	 - How gradients are computed and propagated
	 - How neural networks are structured
	 - How optimizers update model parameters
	 - How datasets and batching work
  
## Core Concepts Implemented
 

 1. Tensor Operations 
	 - Tensor creation
	 - Shape inspection
	 - Reshaping and broadcasting
	 - Matrix multiplication
	 - Indexing & slicing
	 - Device transfers (CPU ↔ GPU)
	 
2. Autograd (Automatic Differentiation)
	- requires_grad
	- Backpropagation using .backward()
	- Gradient accumulation
	- Zeroing gradients
	- No-grad context

3. CUDA Support
	- Detect GPU availability
	- Move tensors/models to GPU
	- Mixed device handling

4. Neural Networks
	- Linear layers
	- Activation functions
	- Sequential models
	- Custom modules via subclassing
	
    Example structure:


```   
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
```

5. Loss Functions
	- CrossEntropyLoss
	- MSELoss
	- Reduction modes

6. Optimizers
	- SGD
	- Adam
	- Learning rate control
	- Gradient zeroing

7. Dataset & DataLoader
	- Custom Dataset class  
	- __getitem__
	- __len__
	- Batching
	- Shuffling
	
    Example:

```
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
```
## Installation

1️⃣ Clone the repository
```
git clone https://github.com/yourusername/your-repo-name.git

cd your-repo-name
```
2️⃣ Create virtual environment (recommended)
```
python -m venv venv

source venv/bin/activate # Mac/Linux

venv\Scripts\activate # Windows
```
3️⃣ Install dependencies
```
pip install torch torchvision
```
## Running the Notebook

Open the Jupyter notebook:
```
jupyter notebook
```
Then open:
```
TensorFlow.ipynb
```
## GPU Usage

To check CUDA availability:
```
torch.cuda.is_available()
```
To move tensors to GPU:
```
device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = tensor.to(device)

model.to(device)
```
## Example Training Loop
```
for epoch in range(epochs):
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
## Learning Objectives

By working through this repository, you will understand:

- SGD
- How gradient-based optimization works
- How backpropagation updates parameters
- How neural network modules are structured
- How GPU acceleration improves training
- How batching impacts performance
- The internal mechanics behind TensorFlow/PyTorch