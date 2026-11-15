import torch
from models.backbone.resnet import ResNetBackbone

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)

# Create ResNet-18 backbone
backbone = ResNetBackbone().to(device)

# Load ImageNet pretrained weights
print("\nLoading ImageNet pretrained weights...")
backbone.load_pretrained()

# Test feature extraction
image = torch.randn(1, 3, 256, 256, device=device)

backbone.eval()

with torch.no_grad():
    c3, c4, c5 = backbone(image)

print("\nResNet feature extraction:")
print("C3:", c3.shape)
print("C4:", c4.shape)
print("C5:", c5.shape)

print("\nPRETRAINED RESNET TEST PASSED")
