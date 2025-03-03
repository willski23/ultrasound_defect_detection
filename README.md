# Ultrasound Probe Defect Detection

A deep learning project for analyzing ultrasound probe data and detecting defects using segmentation.

## Project Overview

This project implements a complete pipeline for processing ultrasound probe data from a Verasonics testing system to detect defects in linear ultrasound probes. The implementation uses transfer learning with ResNet-50 as the backbone for a segmentation model.

### Key Features

- Convert MAT files to PNG images
- Create segmentation masks for defect detection
- Organize data by different experimental conditions
- Implement data augmentation to avoid overfitting
- Build and train a segmentation model with ResNet-50 backbone
- Evaluate model performance and visualize results

## Project Structure

```
ultrasound_defect_detection/
├── README.md                       # Project documentation
├── requirements.txt                # Dependencies
├── config.py                       # Configuration parameters
├── data/                           # Data directory
│   ├── raw/                        # Raw MAT files
│   ├── processed/                  # Processed PNG images
│   ├── masks/                      # Segmentation masks
│   ├── organized/                  # Data organized by condition
│   └── augmented/                  # Augmented data
├── models/                         # Model checkpoints and saved models
│   └── checkpoints/
├── src/                            # Source code
│   ├── data/                       # Data processing modules
│   ├── model/                      # Model architecture and training
│   └── utils/                      # Utility functions
├── notebooks/                      # Jupyter notebooks
└── scripts/                        # CLI scripts
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ultrasound_defect_detection.git
cd ultrasound_defect_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project is organized into sequential steps. Execute them in order:

1. Convert MAT files to PNG:
```bash
python scripts/01_convert_data.py
```

2. Create segmentation masks:
```bash
python scripts/02_create_masks.py
```

3. Organize data by condition:
```bash
python scripts/03_organize_unified.py
```

4. Augment data:
```bash
python scripts/04_augment_unified.py
```

5. Train the model:
```bash
python scripts/05_train_unified.py
```

6. Evaluate model performance:
```bash
python scripts/06_evaluate_unified.py
```

7. Make predictions on new data:
```bash
python scripts/07_predict_unified.py --input path/to/image.png --output prediction.png
```

## Data

The project expects MAT files with the following structure:
- `deadElements`: 1 x 128 double array indicating disabled elements with a value of 1
- `imgData`: 118 x 128 double array with the amplitude of signals in B-mode image
- `xAxis`: 1 x 128 double array
- `zAxis`: 1 x 128 double array

The data covers 5 different conditions:
- All elements enabled
- One element off sequentially
- Two contiguous elements off sequentially
- Five contiguous elements off
- Five random elements off

## Model

The model architecture is a U-Net-like segmentation network with a pre-trained ResNet-50 backbone:
- Encoder: ResNet-50 pre-trained on ImageNet
- Decoder: Custom upsampling path with skip connections
- Output: Binary segmentation mask of defects

## License

[Specify your license here]

## Contact

[Your contact information]