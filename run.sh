#!/bin/bash

# Create necessary directories
mkdir -p results

# Example 1: Using a PyTorch model
echo "Example 1: Using a PyTorch model"
echo "python src/model_evaluation.py --model models/resnet50.pt --img images/apple.jpg --view"
echo ""

# Example 2: Using a TensorFlow model with custom labels
echo "Example 2: Using a TensorFlow model with custom labels"
echo "python src/model_evaluation.py --model models/mobilenet.pb --img images/apple.jpg --labels labels.txt --view"
echo ""

# Example 3: Using a PyTorch model with custom normalization
echo "Example 5: Using a PyTorch model with custom normalization"
echo "python src/model_evaluation.py --model models/custom_model.pt --img images/apple.jpg --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5"
echo ""

# Example 6: Batch processing multiple images
echo "Example 6: Batch processing multiple images"
echo "for img in images/*.jpg; do"
echo "  python src/model_evaluation.py --model models/resnet50.pt --img \"\$img\" --output results/batch"
echo "done"
echo ""

echo "Note: Replace the model and image paths with your actual files."
echo "Make sure you have installed all dependencies from requirements.txt first."
echo "pip install -r requirements.txt" 