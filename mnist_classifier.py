# MNIST Digit Classification Project
# We will build this step-by-step.
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Step 1: Device Configuration & Hyperparameters ---
# Select GPU (MPS for Apple Silicon Mac, CUDA for Nvidia) or fallback to CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# --- Step 2: Prepare Dataset and DataLoaders ---
# Data Augmentation for training: Random slight rotation, translation & scale to handle custom drawing styles
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.92, 1.08)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test transform (clean, no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training & test datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=train_transform, 
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=test_transform, 
    download=True
)

# DataLoaders manage batching, shuffling, and multi-threaded loading
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# --- Step 3: Define the Neural Network Architecture ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Conv layer 1: 1 input channel (grayscale), 32 output filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Conv layer 2: 32 input channels, 64 output filters, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Fully connected layers: 64 * 7 * 7 flattened features -> 128 -> 10 (digits 0-9)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten: [batch_size, 64, 7, 7] -> [batch_size, 64 * 7 * 7]
        x = x.view(x.size(0), -1)
        
        # Classification head
        x = self.dropout(self.relu3(self.fc1(x)))
        out = self.fc2(x)
        return out

model = ConvNet().to(device)
print(model)

# --- Step 4: Loss Function and Optimizer ---
# Loss function: Measures how far off our predictions are from the true labels
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam algorithm adjusts network weights based on the loss gradients
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Step 5: Training Loop ---
print("\n--- Starting Training ---")
total_steps = len(train_loader)

for epoch in range(num_epochs):
    model.train()  # Put model in training mode (enables dropout)
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Move data to the selected device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Forward pass: Compute predicted outputs by passing images to the model
        outputs = model(images)
        
        # 2. Calculate the batch loss
        loss = criterion(outputs, labels)
        
        # 3. Clear gradients from previous step
        optimizer.zero_grad()
        
        # 4. Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # 5. Optimization: Perform a single parameter update
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress every 300 batches
        if (i + 1) % 300 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")

    avg_epoch_loss = running_loss / total_steps
    print(f"--> Epoch [{epoch + 1}/{num_epochs}] Finished. Average Loss: {avg_epoch_loss:.4f}")

# --- Step 6: Evaluate Model on Test Data ---
print("\n--- Evaluating on Test Data ---")
model.eval()  # Put model in evaluation mode (disables dropout)

correct = 0
total = 0

# torch.no_grad() disables gradient calculation to save memory and speed up testing
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # Get the predicted digit (index with maximum score)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Accuracy of the model on the 10,000 test images: {accuracy:.2f}%")

# --- Step 7: Save the Trained Model ---
model_path = 'mnist_cnn.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved successfully to {model_path}!")
