#!/usr/bin/env python
"""
Script to make predictions on new images using the unified approach.
"""
import os
import sys
import argparse
import glob
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.model.evaluate_unified import predict_on_new_image


def main():
    """Main function to make predictions on new images with element detection."""
    parser = argparse.ArgumentParser(description="Make predictions with element detection")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--output_dir", type=str, default="predictions_unified",
                        help="Directory to save prediction results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation")
    parser.add_argument("--num_elements", type=int, default=128,
                        help="Number of elements in the ultrasound probe")

    args = parser.parse_args()

    # Find the latest model if no model path is provided
    if args.model_path is None:
        model_files = [f for f in os.listdir(config.CHECKPOINT_DIR)
                       if f.endswith('.h5') and not f.endswith('_final.h5') and 'unified' in f]
        if not model_files:
            # Try to find any model
            model_files = [f for f in os.listdir(config.CHECKPOINT_DIR)
                           if f.endswith('.h5') and not f.endswith('_final.h5')]
            if not model_files:
                print("Error: No model files found. Please provide a model path.")
                return 1

        # Get the most recent model
        latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(config.CHECKPOINT_DIR, f)))
        args.model_path = os.path.join(config.CHECKPOINT_DIR, latest_model)
        print(f"Using latest model: {args.model_path}")

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return 1

    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file prediction
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # Directory of images
        image_paths = glob.glob(os.path.join(args.input, "*.png"))
        image_paths.extend(glob.glob(os.path.join(args.input, "*.jpg")))
        image_paths.extend(glob.glob(os.path.join(args.input, "*.jpeg")))

        if not image_paths:
            print(f"Error: No image files found in {args.input}")
            return 1
    else:
        print(f"Error: Input path {args.input} does not exist.")
        return 1

    print(f"Found {len(image_paths)} images for prediction.")

    # Track all detected defective elements
    all_defective_elements = {}

    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}: {image_path}")

        # Create output path
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")

        try:
            # Make prediction with element detection
            _, _, defective_elements = predict_on_new_image(
                model,
                image_path,
                output_path,
                threshold=args.threshold,
                num_elements=args.num_elements
            )

            # Store detected elements
            all_defective_elements[base_name] = defective_elements

            # Create a text file with detected elements
            elements_path = os.path.join(args.output_dir, f"{base_name}_elements.txt")
            with open(elements_path, 'w') as f:
                f.write(f"Detected {len(defective_elements)} defective elements:\n")
                f.write(", ".join(map(str, defective_elements)))

            print(f"Prediction saved to {output_path}")
            print(f"Detected {len(defective_elements)} defective elements: {defective_elements}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save summary of all defective elements
    if all_defective_elements:
        summary_path = os.path.join(args.output_dir, "defective_elements_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Defective Elements Summary\n")
            f.write("==========================\n\n")

            for name, elements in all_defective_elements.items():
                f.write(f"{name}: {len(elements)} elements - {elements}\n")

        print(f"\nSummary of all defective elements saved to {summary_path}")

    print("\nPrediction complete. Results saved to", args.output_dir)
    return 0


if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error setting memory growth: {e}")

    sys.exit(main())