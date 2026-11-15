import torch

from models.backbone.resnet import ResNetBackbone
from models.detection.fpn import FPN
from models.detection.head import DetectionHead
from models.detection.anchors import AnchorGenerator
from models.detection.detector import FPNDetector
from models.segmentation.unet import UNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)

# Sample input image
images = torch.randn(1, 3, 256, 256, device=device)

# 1. Object Detection: ResNet-18 + FPN + Detection Head
backbone = ResNetBackbone().to(device)
fpn = FPN([128, 256, 512], 256).to(device)
head = DetectionHead(256, 3, 3).to(device)

anchor_generator = AnchorGenerator(
    scales=[128, 64, 32],
    aspect_ratios=[0.5, 1.0, 2.0],
    strides=[32, 16, 8]
)

detector = FPNDetector(
    backbone, fpn, head, anchor_generator, num_classes=3
).to(device)

detector.train()

with torch.no_grad():
    cls_logits, bbox_deltas, anchors = detector(images)

print("\nOBJECT DETECTION")
for i, output in enumerate(cls_logits):
    print(f"Level {i+1} classification:", output.shape)
    print(f"Level {i+1} boxes:", bbox_deltas[i].shape)

print("Anchors:", anchors.shape)

# 2. Semantic Segmentation: ResNet-18 + U-Net
seg_backbone = ResNetBackbone().to(device)
segmenter = UNet(seg_backbone, num_classes=5).to(device)

with torch.no_grad():
    segmentation = segmenter(images)

print("\nSEMANTIC SEGMENTATION")
print("Output:", segmentation.shape)

print("\nALL MODEL FORWARD TESTS PASSED")
