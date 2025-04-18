import os
import torch
import torchvision.models as models
import timm
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import json
import torch.nn as nn

class ModelDownloader:
    def __init__(self, save_dir='models/pretrained'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def model_exists(self, name, format='pt'):
        """Check if model already exists"""
        if format == 'pt':
            model_path = self.save_dir / f"{name}_model.pt"
            weights_path = self.save_dir / f"{name}_weights.pth"
            return model_path.exists() and weights_path.exists()
        elif format == 'pb':
            model_path = self.save_dir / name
            return model_path.exists() and (model_path / "saved_model.pb").exists()
        return False
        
    def save_model(self, model, name, format='pt'):
        """Save model and its state dict"""
        if format == 'pt':
            # Save full model
            model_path = self.save_dir / f"{name}_model.pt"
            torch.save(model, model_path)
            
            # Save state dict with .pth extension
            state_dict_path = self.save_dir / f"{name}_weights.pth"
            torch.save(model.state_dict(), state_dict_path)
            
            # Save model config for fine-tuning
            config_path = self.save_dir / f"{name}_config.json"
            if hasattr(model, 'config'):
                with open(config_path, 'w') as f:
                    json.dump(model.config, f)
            
            print(f"Saved {name} model to {model_path}")
            print(f"Saved {name} weights to {state_dict_path}")
            if hasattr(model, 'config'):
                print(f"Saved {name} config to {config_path}")
        elif format == 'pb':
            # Save TensorFlow model
            model_path = self.save_dir / name
            tf.saved_model.save(model, model_path)
            
            # Save model config for fine-tuning
            config_path = self.save_dir / f"{name}_config.json"
            if hasattr(model, 'config'):
                with open(config_path, 'w') as f:
                    json.dump(model.config, f)
            
            print(f"Saved {name} TensorFlow model to {model_path}")
            if hasattr(model, 'config'):
                print(f"Saved {name} config to {config_path}")
        
    def download_resnet_models(self):
        """Download ResNet models"""
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
            'resnext50_32x4d': models.resnext50_32x4d,
            'resnext101_32x8d': models.resnext101_32x8d
        }
        
        for name, model_fn in resnet_models.items():
            if not self.model_exists(name):
                print(f"Downloading {name}...")
                model = model_fn(pretrained=True)
                self.save_model(model, name)
            else:
                print(f"{name} already exists, skipping download")
            
    def download_efficientnet_models(self):
        """Download EfficientNet models"""
        efficientnet_models = [
            'efficientnet_b0',
            'efficientnet_b1',
            'efficientnet_b2',
            'efficientnet_b3',
            'efficientnet_b4',
            'efficientnet_b5',
            'efficientnet_b6',
            'efficientnet_b7'
        ]
        
        for name in efficientnet_models:
            if not self.model_exists(name):
                print(f"Downloading {name}...")
                try:
                    model = timm.create_model(name, pretrained=True)
                    self.save_model(model, name)
                except RuntimeError as e:
                    if "No pretrained weights exist" in str(e):
                        print(f"Warning: No pretrained weights available for {name}. Skipping.")
                    else:
                        raise e
            else:
                print(f"{name} already exists, skipping download")
            
    def download_mobilenet_models(self):
        """Download MobileNet models"""
        # PyTorch MobileNet models
        mobilenet_models = {
            'mobilenet_v2': models.mobilenet_v2,
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'mobilenet_v3_large': models.mobilenet_v3_large
        }
        
        for name, model_fn in mobilenet_models.items():
            if not self.model_exists(name):
                print(f"Downloading PyTorch {name}...")
                model = model_fn(pretrained=True)
                self.save_model(model, name)
            else:
                print(f"PyTorch {name} already exists, skipping download")
        
        # TensorFlow MobileNet models
        tf_mobilenet_models = {
            'mobilenet_v2_tf': "https://tfhub.dev/google/imagenet/MobileNet_V2_100_224/classification/4",
            'mobilenet_v3_small_tf': "https://tfhub.dev/google/imagenet/MobileNet_V3_Small_100_224/classification/5",
            'mobilenet_v3_large_tf': "https://tfhub.dev/google/imagenet/MobileNet_V3_Large_100_224/classification/5"
        }
        
        for name, model_url in tf_mobilenet_models.items():
            if not self.model_exists(name, format='pb'):
                print(f"Downloading TensorFlow {name}...")
                model = hub.load(model_url)
                self.save_model(model, name, format='pb')
            else:
                print(f"TensorFlow {name} already exists, skipping download")
            
    def download_vit_models(self):
        """Download Vision Transformer models"""
        vit_models = [
            'vit_small_patch16_224',
            'vit_base_patch16_224',
            'vit_large_patch16_224'
        ]
        
        for name in vit_models:
            if not self.model_exists(name):
                print(f"Downloading {name}...")
                model = timm.create_model(name, pretrained=True)
                self.save_model(model, name)
            else:
                print(f"{name} already exists, skipping download")

    def prepare_for_finetuning(self, name, num_classes, format='pt'):
        """Prepare a model for fine-tuning with new number of classes"""
        if format == 'pt':
            # Load the model
            model_path = self.save_dir / f"{name}_model.pt"
            if model_path.exists():
                model = torch.load(model_path)
            else:
                weights_path = self.save_dir / f"{name}_weights.pth"
                if not weights_path.exists():
                    raise FileNotFoundError(f"Neither model nor weights found for {name}")
                # Create model based on name
                if 'mobilenet' in name:
                    model = models.mobilenet_v2(pretrained=False)
                elif 'resnet' in name:
                    model = getattr(models, name)(pretrained=False)
                elif 'efficientnet' in name:
                    model = timm.create_model(name, pretrained=False)
                else:
                    raise ValueError(f"Unsupported model architecture: {name}")
                model.load_state_dict(torch.load(weights_path))
            
            # Modify the final layer
            if hasattr(model, 'fc'):  # ResNet style
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'classifier'):  # EfficientNet style
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'head'):  # ViT style
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
            
            return model
            
        elif format == 'pb':
            # Load the TensorFlow model
            model_path = self.save_dir / name
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {name}")
            
            # Load as Keras model for fine-tuning
            model = tf.keras.models.load_model(model_path)
            
            # Modify the final layer
            model.layers[-1] = tf.keras.layers.Dense(num_classes)
            
            return model
        
        raise ValueError(f"Unsupported format: {format}")

def main():
    downloader = ModelDownloader()
    
    # Download all model types
    print("Downloading ResNet models...")
    downloader.download_resnet_models()
    
    print("\nDownloading EfficientNet models...")
    downloader.download_efficientnet_models()
    
    print("\nDownloading MobileNet models...")
    downloader.download_mobilenet_models()
    
    print("\nDownloading Vision Transformer models...")
    downloader.download_vit_models()
    
    print("\nAll models have been downloaded and saved!")

if __name__ == "__main__":
    main() 