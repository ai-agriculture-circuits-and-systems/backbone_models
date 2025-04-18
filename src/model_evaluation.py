#!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
import time
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import densenet121, densenet169, densenet201
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
import requests
import tempfile

# Add all necessary classes to safe globals
torch.serialization.add_safe_globals([
    ResNet, BasicBlock, Bottleneck, Conv2d, BatchNorm2d, Linear, MaxPool2d, 
    AdaptiveAvgPool2d, ReLU, Sequential, ModuleList, ModuleDict
])

def parse_args():
    parser = argparse.ArgumentParser(description='Classification model inference on images')
    parser.add_argument('--model', type=str, required=True, help='Path to classification model')
    parser.add_argument('--img', type=str, required=True, help='Path to input image')
    parser.add_argument('--labels', type=str, help='Path to labels file (JSON or text file)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--view', action='store_true', help='Display results')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--input_size', type=str, default='224,224', help='Input size for the model (width,height)')
    parser.add_argument('--mean', type=str, default='0.485,0.456,0.406', help='Mean values for normalization (R,G,B)')
    parser.add_argument('--std', type=str, default='0.229,0.224,0.225', help='Standard deviation values for normalization (R,G,B)')
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture (e.g., resnet18, resnet50, vgg16, etc.)')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--custom_model', type=str, help='Path to a Python file containing a custom model class')
    parser.add_argument('--custom_class', type=str, help='Name of the custom model class in the custom_model file')
    parser.add_argument('--use_imagenet_labels', action='store_true', help='Use ImageNet labels for standard architectures')
    return parser.parse_args()

def load_labels(labels_path, arch_name=None, use_imagenet_labels=False):
    """
    Load class labels from a file or use standard labels for known architectures.
    
    Args:
        labels_path (str): Path to labels file
        arch_name (str): Name of the model architecture
        use_imagenet_labels (bool): Whether to use ImageNet labels for standard architectures
        
    Returns:
        list: List of class labels
    """
    # If use_imagenet_labels is True and we have a standard architecture, try to load ImageNet labels
    if use_imagenet_labels and arch_name:
        imagenet_labels = load_imagenet_labels()
        if imagenet_labels:
            return imagenet_labels
    
    # If labels_path is provided, try to load from file
    if labels_path and os.path.exists(labels_path):
        # Try to load as JSON first
        try:
            with open(labels_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If not JSON, try as text file with one label per line
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
    
    # If no labels found, return None
    return None

def load_imagenet_labels():
    """
    Load ImageNet class labels from a standard source.
    
    Returns:
        list: List of ImageNet class labels
    """
    # Try to load from a local file first
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels.txt')
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    # If not found locally, try to download from a standard source
    try:
        # URL for ImageNet labels
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Read the labels
        with open(temp_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        
        # Clean up
        os.unlink(temp_path)
        
        return labels
    except Exception as e:
        print(f"Warning: Could not load ImageNet labels: {str(e)}")
        return None

def preprocess_image(image_path, input_size, mean, std):
    """Preprocess image for model input."""
    # Parse input size
    width, height = map(int, input_size.split(','))
    
    # Parse mean and std values
    mean_values = np.array([float(x) for x in mean.split(',')])
    std_values = np.array([float(x) for x in std.split(',')])
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize image
    img = cv2.resize(img, (width, height))
    
    # Convert to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - mean_values) / std_values
    
    # Transpose to CHW format (PyTorch standard)
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension
    img = np.expand_dims(img, 0)
    
    return img

def get_model_architecture(arch_name, num_classes=1000, custom_model_path=None, custom_class_name=None):
    """
    Get a model architecture by name.
    
    Args:
        arch_name (str): Name of the architecture
        num_classes (int): Number of output classes
        custom_model_path (str): Path to a Python file containing a custom model class
        custom_class_name (str): Name of the custom model class in the custom_model file
        
    Returns:
        torch.nn.Module: The model architecture
    """
    # Check if custom model is specified
    if custom_model_path and custom_class_name:
        try:
            import importlib.util
            import sys
            
            # Load the custom model module
            spec = importlib.util.spec_from_file_location("custom_model", custom_model_path)
            custom_module = importlib.util.module_from_spec(spec)
            sys.modules["custom_model"] = custom_module
            spec.loader.exec_module(custom_module)
            
            # Get the custom model class
            if hasattr(custom_module, custom_class_name):
                model_class = getattr(custom_module, custom_class_name)
                # Create an instance of the custom model
                model = model_class(num_classes=num_classes)
                return model
            else:
                raise ValueError(f"Custom class '{custom_class_name}' not found in {custom_model_path}")
        except Exception as e:
            raise ValueError(f"Error loading custom model: {str(e)}")
    
    # Dictionary of available architectures
    arch_dict = {
        # ResNet family
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        
        # VGG family
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19,
        
        # DenseNet family
        'densenet121': densenet121,
        'densenet169': densenet169,
        'densenet201': densenet201,
        
        # EfficientNet family
        'efficientnet_b0': efficientnet_b0,
        'efficientnet_b1': efficientnet_b1,
        'efficientnet_b2': efficientnet_b2,
        'efficientnet_b3': efficientnet_b3,
        'efficientnet_b4': efficientnet_b4,
        'efficientnet_b5': efficientnet_b5,
        'efficientnet_b6': efficientnet_b6,
        'efficientnet_b7': efficientnet_b7,
    }
    
    # Check if architecture is supported
    if arch_name not in arch_dict:
        raise ValueError(f"Unsupported architecture: {arch_name}. Available architectures: {', '.join(arch_dict.keys())}")
    
    # Get the model function
    model_fn = arch_dict[arch_name]
    
    # Create the model with the specified number of classes
    model = model_fn(pretrained=False, num_classes=num_classes)
    
    return model

def classify_image(args):
    """Run classification inference on an image."""
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model {args.model} not found")
    
    # Check if image exists
    if not os.path.exists(args.img):
        raise FileNotFoundError(f"Image {args.img} not found")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load labels if provided or use standard labels for the architecture
    labels = load_labels(args.labels, args.arch, args.use_imagenet_labels)
    
    # Determine model type from extension
    model_ext = os.path.splitext(args.model)[1].lower()
    
    # Preprocess image
    input_tensor = preprocess_image(args.img, args.input_size, args.mean, args.std)
    
    # Load model based on type
    if model_ext == '.pt' or model_ext == '.pth':  # PyTorch models
        try:
            # Load checkpoint
            checkpoint = torch.load(args.model, map_location=args.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                else:
                    # Create model instance with the specified architecture
                    model = get_model_architecture(
                        args.arch, 
                        args.num_classes,
                        args.custom_model,
                        args.custom_class
                    )
                    model.load_state_dict(checkpoint)
            else:
                # Checkpoint is already a model
                model = checkpoint
                
            model.eval()
            
            # Convert input to PyTorch tensor
            input_tensor = torch.from_numpy(input_tensor).to(args.device)
            
            # Ensure consistent data type
            if next(model.parameters()).dtype == torch.float64:
                input_tensor = input_tensor.double()
            elif next(model.parameters()).dtype == torch.float16:
                input_tensor = input_tensor.half()
            else:
                input_tensor = input_tensor.float()
            
            # Run inference
            with torch.no_grad():
                start_time = time.time()
                output = model(input_tensor)
                inference_time = time.time() - start_time
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get top-k predictions
            top_k_values, top_k_indices = torch.topk(probabilities, min(args.top_k, len(probabilities)))
            
            # Convert to numpy for easier handling
            top_k_values = top_k_values.cpu().numpy()
            top_k_indices = top_k_indices.cpu().numpy()
            
            # Get class names
            class_names = []
            for idx in top_k_indices:
                if labels and idx < len(labels):
                    class_names.append(labels[idx])
                else:
                    class_names.append(f"Class {idx}")
            
            # Print results
            print(f"Inference time: {inference_time:.4f} seconds")
            print("\nTop predictions:")
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values)):
                print(f"{i+1}. {class_names[i]}: {prob:.4f}")
            
            # Save results to file
            result_file = os.path.join(args.output, f"result_{os.path.basename(args.img)}.txt")
            with open(result_file, 'w') as f:
                f.write(f"Inference time: {inference_time:.4f} seconds\n\n")
                f.write("Top predictions:\n")
                for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values)):
                    f.write(f"{i+1}. {class_names[i]}: {prob:.4f}\n")
            
            # Visualize results if requested
            if args.view:
                visualize_results(args.img, class_names[0], top_k_values[0], args.output)
            
            return
            
        except ImportError:
            print("PyTorch not found. Please install with: pip install torch")
            return
    
    elif model_ext == '.onnx':  # ONNX models
        try:
            import onnxruntime
            
            # Create ONNX Runtime session
            session = onnxruntime.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                                                 if args.device == 'cuda' else ['CPUExecutionProvider'])
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            start_time = time.time()
            output = session.run(None, {input_name: input_tensor.astype(np.float32)})
            inference_time = time.time() - start_time
            
            # Get probabilities
            probabilities = output[0][0]
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))  # Softmax
            
            # Get top-k predictions
            top_k_indices = np.argsort(probabilities)[-args.top_k:][::-1]
            top_k_values = probabilities[top_k_indices]
            
            # Get class names
            class_names = []
            for idx in top_k_indices:
                if labels and idx < len(labels):
                    class_names.append(labels[idx])
                else:
                    class_names.append(f"Class {idx}")
            
            # Print results
            print(f"Inference time: {inference_time:.4f} seconds")
            print("\nTop predictions:")
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values)):
                print(f"{i+1}. {class_names[i]}: {prob:.4f}")
            
            # Save results to file
            result_file = os.path.join(args.output, f"result_{os.path.basename(args.img)}.txt")
            with open(result_file, 'w') as f:
                f.write(f"Inference time: {inference_time:.4f} seconds\n\n")
                f.write("Top predictions:\n")
                for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values)):
                    f.write(f"{i+1}. {class_names[i]}: {prob:.4f}\n")
            
            # Visualize results if requested
            if args.view:
                visualize_results(args.img, class_names[0], top_k_values[0], args.output)
            
            return
            
        except ImportError:
            print("ONNX Runtime not found. Please install with: pip install onnxruntime")
            return
    
    elif model_ext == '.pb' or model_ext == '.tflite':  # TensorFlow models
        try:
            if model_ext == '.pb':
                import tensorflow as tf
                
                # Load model
                model = tf.saved_model.load(args.model)
                
                # Run inference
                start_time = time.time()
                output = model(input_tensor.astype(np.float32))
                inference_time = time.time() - start_time
                
                # Get probabilities
                probabilities = output[0].numpy()
                
            else:  # TFLite
                import tflite_runtime.interpreter as tflite
                
                # Load model
                interpreter = tflite.Interpreter(model_path=args.model)
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
                
                # Run inference
                start_time = time.time()
                interpreter.invoke()
                inference_time = time.time() - start_time
                
                # Get output tensor
                probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Get top-k predictions
            top_k_indices = np.argsort(probabilities)[-args.top_k:][::-1]
            top_k_values = probabilities[top_k_indices]
            
            # Get class names
            class_names = []
            for idx in top_k_indices:
                if labels and idx < len(labels):
                    class_names.append(labels[idx])
                else:
                    class_names.append(f"Class {idx}")
            
            # Print results
            print(f"Inference time: {inference_time:.4f} seconds")
            print("\nTop predictions:")
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values)):
                print(f"{i+1}. {class_names[i]}: {prob:.4f}")
            
            # Save results to file
            result_file = os.path.join(args.output, f"result_{os.path.basename(args.img)}.txt")
            with open(result_file, 'w') as f:
                f.write(f"Inference time: {inference_time:.4f} seconds\n\n")
                f.write("Top predictions:\n")
                for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values)):
                    f.write(f"{i+1}. {class_names[i]}: {prob:.4f}\n")
            
            # Visualize results if requested
            if args.view:
                visualize_results(args.img, class_names[0], top_k_values[0], args.output)
            
            return
            
        except ImportError:
            print("TensorFlow or TFLite Runtime not found. Please install with: pip install tensorflow or pip install tflite-runtime")
            return
    
    else:
        print(f"Unsupported model format: {model_ext}")
        print("Supported formats: .pt, .pth (PyTorch), .onnx (ONNX), .pb (TensorFlow), .tflite (TFLite)")
        return

def visualize_results(image_path, top_class, confidence, output_dir):
    """Visualize classification results on the image."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    # Add text with prediction
    text = f"{top_class}: {confidence:.2f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save visualization
    save_path = os.path.join(output_dir, f"visualization_{os.path.basename(image_path)}")
    cv2.imwrite(save_path, img)
    
    # Display image if requested
    cv2.imshow("Classification Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    classify_image(args)

if __name__ == "__main__":
    main()