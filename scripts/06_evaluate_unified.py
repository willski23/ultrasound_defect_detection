#!/usr/bin/env python
"""
Script to evaluate the trained model using the unified approach with element-based metrics.
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
import tensorflow as tf
from src.model.dataset_unified import prepare_unified_dataset
from src.model.evaluate_unified import evaluate_model_unified
import json


def main():
    """Main function to evaluate the model with element-based metrics."""
    parser = argparse.ArgumentParser(description="Evaluate model with element-based metrics")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--data_dir", default=os.path.join(config.DATA_DIR, "unified"),
                        help="Directory with unified data")
    parser.add_argument("--dataset_info", type=str, default=None,
                        help="Path to dataset info pickle file (optional)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for evaluation")

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

    # If dataset_info wasn't provided, try to find a matching one based on model name
    if args.dataset_info is None:
        model_basename = os.path.basename(args.model_path).replace('.h5', '')
        dataset_info_path = os.path.join(config.CHECKPOINT_DIR, f"{model_basename}_dataset_info.pkl")

        if os.path.exists(dataset_info_path):
            args.dataset_info = dataset_info_path
            print(f"Found matching dataset info: {args.dataset_info}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Set up output directory
    if args.output_dir is None:
        # Create output directory based on model name
        model_basename = os.path.basename(args.model_path).replace('.h5', '')
        args.output_dir = os.path.join(config.MODEL_DIR, "evaluation_unified", model_basename)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Evaluation results will be saved to {args.output_dir}")

    # Load dataset info if provided
    dataset_info = None
    if args.dataset_info and os.path.exists(args.dataset_info):
        try:
            with open(args.dataset_info, 'rb') as f:
                dataset_info = pickle.load(f)
            print(f"Loaded dataset info from {args.dataset_info}")
        except Exception as e:
            print(f"Warning: Could not load dataset info from {args.dataset_info}: {e}")
            print("Will recreate dataset info instead.")

    # Prepare the dataset
    if dataset_info is None:
        print("Preparing unified dataset...")
        _, _, test_ds, dataset_info = prepare_unified_dataset(
            args.data_dir,
            img_size=model.input_shape[1:3],  # Get expected input size from model
            batch_size=args.batch_size
        )
    else:
        # Still need to prepare the test dataset
        print("Preparing test dataset from existing info...")
        from src.model.dataset_unified import create_tf_dataset

        test_images = dataset_info['test']['images']
        test_masks = dataset_info['test']['masks']
        test_weights = dataset_info.get('test', {}).get('weights')

        if test_weights is not None:
            test_ds = create_tf_dataset(
                test_images, test_masks,
                img_size=model.input_shape[1:3],
                batch_size=args.batch_size,
                weights=test_weights
            )
        else:
            test_ds = create_tf_dataset(
                test_images, test_masks,
                img_size=model.input_shape[1:3],
                batch_size=args.batch_size
            )

    # Evaluate model with unified approach
    print("\nEvaluating model with element-based metrics...")
    metrics = evaluate_model_unified(model, test_ds, dataset_info, args.output_dir)

    # Print element-based metrics
    print("\nElement-based evaluation metrics:")
    element_metrics = {k: v for k, v in metrics.items() if k.startswith('element_')}
    for metric, value in element_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Print defect type metrics
    print("\nMetrics by defect type:")
    if 'defect_type_metrics' in metrics:
        defect_types = metrics['defect_type_metrics']
        for defect_type, defect_metrics in defect_types.items():
            if defect_metrics.get('samples', 0) > 0:
                print(f"  {defect_type} ({defect_metrics['samples']} samples):")
                for metric, value in defect_metrics.items():
                    if metric not in ['samples', 'tp', 'fp', 'fn']:
                        print(f"    {metric}: {value:.4f}")

    # Create a summary file
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Unified Evaluation Summary\n")
        f.write("=========================\n\n")

        f.write("Model: " + os.path.basename(args.model_path) + "\n\n")

        f.write("Standard Metrics:\n")
        standard_metrics = {k: v for k, v in metrics.items() if
                            not k.startswith('element_') and k != 'defect_type_metrics'}
        for metric, value in standard_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nElement-based Metrics:\n")
        for metric, value in element_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nMetrics by Defect Type:\n")
        if 'defect_type_metrics' in metrics:
            for defect_type, defect_metrics in defect_types.items():
                if defect_metrics.get('samples', 0) > 0:
                    f.write(f"  {defect_type} ({defect_metrics['samples']} samples):\n")
                    for metric, value in defect_metrics.items():
                        if metric not in ['samples', 'tp', 'fp', 'fn']:
                            f.write(f"    {metric}: {value:.4f}\n")

    print(f"\nEvaluation summary saved to {summary_path}")
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