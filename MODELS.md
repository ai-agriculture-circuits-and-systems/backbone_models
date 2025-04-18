# Pre-Trained Models

## Installation
```bash
# Install required packages
pip install torch torchvision timm
```

## EfficientNet Backbone Models
Available through timm (PyTorch Image Models):
```python
import timm

# Available models:
# efficientnet_b0
# efficientnet_b1
# efficientnet_b2
# efficientnet_b3
# efficientnet_b4
# efficientnet_b5
# efficientnet_b6
# efficientnet_b7

model = timm.create_model('efficientnet_b0', pretrained=True)
```

## ResNet Backbone Models
Available through torchvision:
```python
import torchvision.models as models

# Available models:
# resnet18
# resnet34
# resnet50
# resnet101
# resnet152
# resnext50_32x4d
# resnext101_32x8d

model = models.resnet50(pretrained=True)
```

## CSP (Cross Stage Partial) Models
These models are part of YOLOv5 architecture and are typically used as part of the complete YOLOv5 model. They are not available as standalone backbone models.

## Mobile Models
Available through torchvision:
```python
import torchvision.models as models

# Available models:
# mobilenet_v2
# mobilenet_v3_small
# mobilenet_v3_large

model = models.mobilenet_v2(pretrained=True)
```

## Vision Transformer Models
Available through timm:
```python
import timm

# Available models:
# vit_small_patch16_224
# vit_base_patch16_224
# vit_large_patch16_224

model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

