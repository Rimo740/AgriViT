import torch
import torch.nn as nn
import torchvision.models as models
import os
import json

# Force CPU to avoid CUDA errors if you don't have it set up locally
device = torch.device("cpu")

# Load Classes
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
except:
    num_classes = 38

def compress(filename):
    print(f"📉 Compressing {filename}...")
    if not os.path.exists(filename):
        print("❌ File not found.")
        return

    # Initialize Swin-T structure
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)

    # Load Weights (map_location is critical)
    state_dict = torch.load(filename, map_location=device)
    model.load_state_dict(state_dict)

    # Convert to Half (Float16) - This cuts size by 50%
    model.half()

    # Save back to the same file
    torch.save(model.state_dict(), filename)

    size = os.path.getsize(filename) / (1024 * 1024)
    print(f"✅ Success! New size: {size:.2f} MB")

if __name__ == "__main__":
    compress("mobilevit.pth")