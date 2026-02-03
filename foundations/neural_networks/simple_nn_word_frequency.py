"""
Simple Neural Network – Learning Example

Purpose:
- Understand forward pass and backpropagation in PyTorch
- Learn how loss.backward() updates weights
- Observe limitations of one-hot encoding

This is a learning-focused implementation.
"""
# Neural Networks Basics & Backpropagation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Sample data: Word counts for prediction task
input_text = "The quick brown fox jumps over the quick brown dog"
words = input_text.lower().split()
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# Convert to features
unique_words = list(word_count.keys())
X = np.zeros((len(unique_words), len(unique_words)))
y = np.zeros((len(unique_words), 1))

# Create a simple word prediction dataset
for i, word in enumerate(unique_words):
    X[i, i] = 1  # One-hot encoding
    y[i] = word_count[word]  # Word frequency as target

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Initialize the model
input_size = len(unique_words)
hidden_size = 5
output_size = 1
model = SimpleNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Visualize backpropagation
losses = []

# Training loop
for epoch in range(1000):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    losses.append(loss.item())

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Visualize loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Visualize weights
weights1 = model.layer1.weight.data.numpy()
weights2 = model.layer2.weight.data.numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(weights1, cmap='viridis')
plt.colorbar()
plt.title('Layer 1 Weights')

plt.subplot(1, 2, 2)
plt.imshow(weights2, cmap='viridis')
plt.colorbar()
plt.title('Layer 2 Weights')
plt.tight_layout()
plt.show()

# Explain backpropagation
print("Backpropagation Process:")
print("1. Forward pass computes the prediction")
print("2. Loss is calculated between prediction and actual values")
print("3. Gradients flow backward through the network")
print("4. Weights are updated to minimize loss")

model.eval()  # switch to evaluation mode

with torch.no_grad():
    final_outputs = model(X_tensor)

print("\nFinal Predictions:")
for i, word in enumerate(unique_words):
    print(
        f"Word: {word:>6} | "
        f"Predicted count: {final_outputs[i].item():.2f} | "
        f"Actual count: {y_tensor[i].item()}"
    )
