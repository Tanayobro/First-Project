import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2 as cv
import numpy as np

# 1. Define the exact same architecture as trained
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        out = self.fc2(x)
        return out

def load_trained_model(weights_path='mnist_cnn.pth'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

def predict_image(image_path, model, device):
    """
    Takes any custom image file (PNG/JPG), formats it for MNIST, and predicts the digit.
    """
    # Open image and convert to grayscale ('L' mode)
    img = Image.open(image_path).convert('L')
    
    # MNIST images have white digits on a black background.
    # If drawn with black ink on white paper, invert colors:
    pixel_values = list(img.get_flattened_data()) if hasattr(img, 'get_flattened_data') else list(img.getdata())
    avg_brightness = sum(pixel_values) / max(len(pixel_values), 1)
    if avg_brightness > 128:
        # Invert: white background -> black, black ink -> white
        img = ImageOps.invert(img)
    
    # Resize to standard MNIST 28x28 resolution
    img = img.resize((28, 28), Image.Resampling.BILINEAR)
    
    # Apply standard MNIST normalization transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor_img = transform(img).unsqueeze(0).to(device)  # Add batch dimension: [1, 1, 28, 28]
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_digit].item() * 100
        
    print(f"\n==========================================")
    print(f"Image: {image_path}")
    print(f"Predicted Digit: >> {predicted_digit} << (Confidence: {confidence:.2f}%)")
    print(f"==========================================")
    print("Probabilities across all digits:")
    for digit in range(10):
        bar = "█" * int(probabilities[digit].item() * 30)
        print(f"Digit {digit}: {probabilities[digit].item()*100:5.1f}%  {bar}") # IDHAR DEKH
    print(f"==========================================\n")
    
    return predicted_digit

def merge_boxes(boxes):
    """Merge vertically overlapping or nested contours (e.g. disconnected strokes for 4, 5, 7)"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        prev_x, prev_y, prev_w, prev_h = merged[-1]
        curr_x, curr_y, curr_w, curr_h = box
        
        # Calculate horizontal overlap
        overlap = (prev_x + prev_w) - curr_x
        
        # Merge only if they overlap significantly 
        # (e.g. overlap is more than 30% of the narrower box's width)
        # This prevents adjacent slanted digits from being swallowed!
        if overlap > 0 and overlap > 0.3 * min(prev_w, curr_w):
            new_x = min(prev_x, curr_x)
            new_y = min(prev_y, curr_y)
            new_w = max(prev_x + prev_w, curr_x + curr_w) - new_x
            new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(box)
    return merged

def prepare_digit(digicrop):
    """
    Centers the digit using Center of Mass (Centroid) and thickens strokes 
    to match the standard MNIST dataset format.
    """
    # 1. Thicken stroke if drawn with a thin digital pen
    kernel = np.ones((2, 2), np.uint8)
    if max(digicrop.shape) > 35:
        digicrop = cv.dilate(digicrop, kernel, iterations=1)

    h, w = digicrop.shape
    scale = 20.0 / max(w, h)
    hnew, wnew = max(int(h * scale), 1), max(int(w * scale), 1)

    resized = cv.resize(digicrop, (wnew, hnew), interpolation=cv.INTER_AREA)

    # 2. Place on 28x28 canvas with geometric padding initially
    canvas = np.zeros((28, 28), dtype=np.uint8)
    pad_y = (28 - hnew) // 2
    pad_x = (28 - wnew) // 2
    canvas[pad_y:pad_y + hnew, pad_x:pad_x + wnew] = resized

    # 3. Center of Mass Shift (MNIST standard alignment)
    M = cv.moments(canvas)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        warp_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv.warpAffine(canvas, warp_matrix, (28, 28), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    return canvas

def process_number(img_path, model, device)->int:
    # 1. Import image as grayscale
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return ""

    # 2. High contrast threshold (auto-detect background color)
    if np.mean(img) > 128:
        _, thres = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    else:
        _, thres = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 3. Find contours and bounding boxes
    contours, _ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Keep strokes at least 5px tall and 2px wide to catch smaller handwriting
        if h >= 5 and w >= 2:
            bounding_boxes.append((x, y, w, h))

    # 4. Merge overlapping strokes and sort left-to-right
    bounding_boxes = merge_boxes(bounding_boxes)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    digits = []
    confidences = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    for (x, y, w, h) in bounding_boxes:
        digicrop = thres[y:y+h, x:x+w]
        canvas = prepare_digit(digicrop)
        tensor = transform(Image.fromarray(canvas)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            prob = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(prob).item()
            conf = prob[pred].item() * 100.0
            
            digits.append(str(pred))
            confidences.append(conf)

    full_number = int("".join(digits))
    print(f"\n==========================================")
    print(f"Image: {img_path}")
    for i, (d, c) in enumerate(zip(digits, confidences)):
        print(f"  Digit {i+1} -> Predicted: '{d}' (Confidence: {c:.2f}%)")
    print(f"------------------------------------------")
    print(f"Final Multi-Digit Number: >> {full_number} <<")
    print(f"==========================================\n")
    return full_number


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image.png>")
        print("Enter the name of your file :")
        image_file = input()

    else:
        image_file = sys.argv[1]
    model, device = load_trained_model('mnist_cnn.pth')
    process_number(image_file, model, device)

