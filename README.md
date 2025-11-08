# Robotic Arm Grasping

A deep learning-based robotic grasping system using RGB-D images for pose estimation and grasp planning.

## Overview

This project implements an end-to-end deep learning pipeline for robotic arm grasping tasks. The system processes RGB-D images to predict optimal grasp poses, including position coordinates and orientation angles, enabling autonomous object manipulation.

## Features

- ResNet-based architecture for robust feature extraction
- RGB-D image processing for accurate pose estimation
- Multi-metric optimization with combined loss function
- Data augmentation techniques (rotation, flipping, color jittering)
- Performance tracking and visualization tools
- GPU acceleration support

## Requirements

```
python >= 3.7
torch >= 1.8.0
torchvision >= 0.9.0
numpy >= 1.19.0
opencv-python >= 4.5.0
matplotlib >= 3.3.0
pillow >= 8.0.0
```

## Installation

```bash
git clone https://github.com/dongdapengyuyan/robotic-arm-grasping.git
cd robotic-arm-grasping
pip install -r requirements.txt
```

## Usage

Training:
```bash
python train.py --epochs 30 --batch_size 32 --lr 0.001
```

Evaluation:
```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

Inference:
```bash
python inference.py --image_path test_image.png
```

## Dataset Structure

```
data/
├── train/
│   ├── rgb/
│   ├── depth/
│   └── labels.csv
└── val/
    ├── rgb/
    ├── depth/
    └── labels.csv
```

## Performance

- Validation Accuracy: 87.41%
- Position Error: 13.30 pixels
- Angle Error: 5.50 degrees
- Training Improvement: 83.8% reduction in position error

## Model Architecture

Modified ResNet backbone with custom prediction heads for grasp position, orientation, and success probability.

## Project Structure

- model.py - Neural network architecture
- dataset.py - Dataset loader with augmentation
- train.py - Training pipeline
- evaluate.py - Evaluation tools
- utils.py - Helper functions

## License

MIT License

## Author

dongdapengyuyan

## Contact

For questions, please open an issue on GitHub.
