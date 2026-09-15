import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Load test dataset without normalization to save readable PNG images
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False)

# Save 5 sample images with their actual labels
for i in range(5):
    img, label = test_dataset[i]
    filename = f"sample_digit_{label}_index{i}.png"
    img.save(filename)
    print(f"Saved {filename} (True Label: {label})")

print("\nSample images saved successfully! You can pass any of them to predict.py.")

