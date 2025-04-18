# Backbone Models Collection

A collection of pre-trained backbone models for computer vision tasks, including ResNet, EfficientNet, MobileNet, and Vision Transformer architectures.

## Project Structure

```
backbone_models/
├── images/              # Sample images for testing
├── models/              # Directory for storing downloaded models
│   └── pretrained/      # Pre-trained model weights
├── results/             # Output results from model evaluation
├── src/                 # Source code
│   ├── imagenet_labels.txt  # ImageNet class labels
│   ├── model_evaluation.py  # Script for evaluating models
│   └── download_models.py   # Script to download pre-trained models
├── venv/                # Python virtual environment
├── requirements.txt     # Python dependencies
└── run.sh               # Example commands for running the project
```

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip download --no-cache-dir -r requirements.txt -d wheels
pip install --no-index --find-links=wheels -r requirements.txt
```

## Downloading Pre-trained Models

The project includes a script to download various pre-trained models:

```bash
python src/download_models.py
```

This will download the following model architectures:
- ResNet (18, 34, 50, 101, 152, ResNeXt)
- EfficientNet (B0-B7)
- MobileNet (V2, V3-Small, V3-Large)
- Vision Transformer (Small, Base, Large)

## Evaluating Models

The `model_evaluation.py` script allows you to evaluate models on images:

```bash
# Basic usage
python src/model_evaluation.py --model models/pretrained/resnet50_model.pt --img images/apple.jpg

# With visualization
python src/model_evaluation.py --model models/pretrained/resnet50_model.pt --img images/apple.jpg --view

# With custom normalization
python src/model_evaluation.py --model models/pretrained/resnet50_model.pt --img images/apple.jpg --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5
```

See `run.sh` for more example commands.

## Available Models

The project supports various backbone architectures:

### ResNet Models
- ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- ResNeXt50, ResNeXt101

### EfficientNet Models
- EfficientNet-B0 through EfficientNet-B7

### MobileNet Models
- MobileNetV2, MobileNetV3-Small, MobileNetV3-Large

### Vision Transformer Models
- ViT-Small, ViT-Base, ViT-Large

For more details, see [MODELS.md](MODELS.md).

## Requirements

- Python 3.7+
- PyTorch 1.9+
- TensorFlow 2.5+ (optional)
- See [requirements.txt](requirements.txt) for full dependencies

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.