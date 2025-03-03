"""
Configuration parameters for the ultrasound defect detection project.
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Data subdirectories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MASK_DIR = os.path.join(DATA_DIR, "masks")
ORGANIZED_DATA_DIR = os.path.join(DATA_DIR, "organized")
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "augmented")
UNIFIED_DATA_DIR = os.path.join(DATA_DIR, "unified")  # New unified data directory
AUGMENTED_UNIFIED_DATA_DIR = os.path.join(DATA_DIR, "augmented_unified")  # New augmented unified data directory

# Model subdirectories
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ultrasound_defect_segmentation_model.h5")
UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "ultrasound_unified_model.h5")  # Path for unified model

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MASK_DIR, ORGANIZED_DATA_DIR,
                 AUGMENTED_DATA_DIR, UNIFIED_DATA_DIR, AUGMENTED_UNIFIED_DATA_DIR,
                 MODEL_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset parameters
IMAGE_SIZE = (224, 224)  # Target image size for ResNet50
BATCH_SIZE = 16
NUM_AUGMENTATIONS_PER_IMAGE = 5

# Conditions
CONDITIONS = [
    "all_elements_enabled",
    "one_element_off",
    "two_contiguous_off",
    "five_contiguous_off",
    "five_random_off"
]

# Training parameters
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Hardware parameters
NUM_TRANSDUCER_ELEMENTS = 128  # Number of elements in the ultrasound probe