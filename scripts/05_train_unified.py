#!/usr/bin/env python
"""
Script to train the segmentation model using the unified data approach.
"""
import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
from src.model.dataset_unified import prepare_unified_dataset, analyze_dataset_composition, visualize_dataset_samples
from src.model.network import build_unet_resnet50, build_unet, get_model_summary
from src.model.train import train_model
import tensorflow as tf
import pickle


def inspect_dataset(dataset, name):
    """
    Inspect the structure and shapes of a TensorFlow dataset.
    """
    print(f"\nInspecting {name} dataset...")

    # Get a batch from the dataset
    for batch in dataset.take(1):
        # Check if dataset includes sample weights (will have 3 elements instead of 2)
        if len(batch) == 3:
            images, masks, weights = batch
            print(f"  Dataset includes sample weights")
            print(f"  Weights shape: {weights.shape}")
            print(f"  Weights min/max: {tf.reduce_min(weights).numpy()}, {tf.reduce_max(weights).numpy()}")
        else:
            images, masks = batch
            print(f"  Dataset does not include sample weights")

        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Masks dtype: {masks.dtype}")

        # Check value ranges
        print(f"  Images min/max: {tf.reduce_min(images).numpy()}, {tf.reduce_max(images).numpy()}")
        print(f"  Masks min/max: {tf.reduce_min(masks).numpy()}, {tf.reduce_max(masks).numpy()}")

        # Check mask distribution
        mask_mean = tf.reduce_mean(masks).numpy()
        mask_zeros = tf.reduce_sum(tf.cast(tf.equal(masks, 0), tf.float32)).numpy()
        mask_ones = tf.reduce_sum(tf.cast(tf.equal(masks, 1), tf.float32)).numpy()
        total_pixels = tf.size(masks).numpy()

        print(f"  Mask mean value: {mask_mean}")
        print(f"  Percent zeros: {mask_zeros / total_pixels * 100:.2f}%")
        print(f"  Percent ones: {mask_ones / total_pixels * 100:.2f}%")

        # If masks are not already binary, print a warning
        if not (tf.reduce_min(masks).numpy() == 0 and tf.reduce_max(masks).numpy() == 1):
            print("  WARNING: Masks are not binary (0 and 1 values only)")

        # If shapes don't match for segmentation (should be same H,W)
        if images.shape[1:3] != masks.shape[1:3]:
            print(f"  WARNING: Image and mask spatial dimensions don't match!")


def main():
    """Main function to train the segmentation model using unified data approach."""
    parser = argparse.ArgumentParser(description="Train segmentation model with unified data")
    parser.add_argument("--data_dir", default=config.AUGMENTED_UNIFIED_DATA_DIR,
                        help="Directory with unified data (default: augmented unified data)")
    parser.add_argument("--checkpoint_dir", default=config.CHECKPOINT_DIR,
                        help="Directory to save model checkpoints")
    parser.add_argument("--model_type", choices=["resnet50", "unet"], default="resnet50",
                        help="Type of model architecture to use")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze the encoder weights")
    parser.add_argument("--image_size", type=int, nargs=2, default=config.IMAGE_SIZE,
                        help="Input image size (height width)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Number of epochs to train")
    parser.add_argument("--balance_classes", action="store_true", default=True,
                        help="Balance classes by weighting samples")
    parser.add_argument("--continue_training", type=str, default=None,
                        help="Path to model to continue training from")
    parser.add_argument("--visualize_samples", action="store_true",
                        help="Visualize dataset samples before training")

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        if args.data_dir == config.AUGMENTED_UNIFIED_DATA_DIR:
            print("Please run scripts in this order:")
            print("1. python scripts/01_convert_data.py")
            print("2. python scripts/02_create_masks.py")
            print("3. python scripts/03_organize_unified.py")
            print("4. python scripts/04_augment_unified.py")
        return 1

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create a timestamp for this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"ultrasound_unified_{args.model_type}_{timestamp}"

    # Prepare dataset
    print("Preparing unified datasets...")
    train_ds, val_ds, test_ds, dataset_info = prepare_unified_dataset(
        args.data_dir,
        img_size=tuple(args.image_size),
        batch_size=args.batch_size,
        balance_classes=args.balance_classes
    )

    inspect_dataset(train_ds, "training")
    inspect_dataset(val_ds, "validation")
    inspect_dataset(test_ds, "test")

    # Analyze dataset composition
    stats = analyze_dataset_composition(dataset_info)
    print("\nDataset composition:")
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.capitalize()} set ({split_stats['total']} samples):")
        print(
            f"  With defects: {split_stats['with_defects']} ({split_stats['with_defects'] / split_stats['total'] * 100:.1f}%)")
        print(
            f"  Without defects: {split_stats['without_defects']} ({split_stats['without_defects'] / split_stats['total'] * 100:.1f}%)")
        print(f"  Pattern distribution:")
        for pattern, count in split_stats['patterns'].items():
            print(f"    {pattern}: {count} ({count / split_stats['total'] * 100:.1f}%)")

    # Visualize dataset samples if requested
    if args.visualize_samples:
        vis_dir = os.path.join(project_root, "notebooks")
        os.makedirs(vis_dir, exist_ok=True)
        visualize_dataset_samples(
            train_ds,
            num_samples=3,
            output_path=os.path.join(vis_dir, "unified_dataset_samples.png")
        )

    # Initialize or load model
    if args.continue_training:
        print(f"Loading model from {args.continue_training}...")
        if not os.path.exists(args.continue_training):
            print(f"Error: Model file {args.continue_training} does not exist.")
            return 1

        try:
            model = tf.keras.models.load_model(args.continue_training)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
    else:
        print(f"Initializing new {args.model_type} model...")
        if args.model_type == "resnet50":
            model = build_unet_resnet50(
                input_shape=(*args.image_size, 3),
                freeze_encoder=args.freeze_encoder
            )
        else:  # unet
            model = build_unet(
                input_shape=(*args.image_size, 3)
            )

    # Print model summary
    print("\nModel Summary:")
    summary = get_model_summary(model)
    print(summary)

    # Save model summary to file
    summary_path = os.path.join(args.checkpoint_dir, f"{run_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    # Train the model
    print(f"\nStarting model training for {args.epochs} epochs...")

    trained_model, history, model_path = train_model(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        model_name=run_name
    )

    print(f"Model training complete. Final model saved to {model_path}")

    # Save dataset info for later use in evaluation
    dataset_info_path = os.path.join(args.checkpoint_dir, f"{run_name}_dataset_info.pkl")
    with open(dataset_info_path, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"Dataset information saved to {dataset_info_path} for later evaluation.")

    # Simple evaluation on test set
    print("\nEvaluating on test set...")
    test_results = trained_model.evaluate(test_ds, verbose=1)

    print("Test results:")
    for i, metric_name in enumerate(trained_model.metrics_names):
        print(f"  {metric_name}: {test_results[i]:.4f}")

    # Save test results
    test_results_path = os.path.join(args.checkpoint_dir, f"{run_name}_test_results.txt")
    with open(test_results_path, 'w', encoding='utf-8') as f:
        f.write("Test Results:\n")
        for i, metric_name in enumerate(trained_model.metrics_names):
            f.write(f"{metric_name}: {test_results[i]:.4f}\n")

    print(f"Test results saved to {test_results_path}.")
    return 0


if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
        except Exception as e:
            print(f"Error setting memory growth: {e}")

    sys.exit(main())