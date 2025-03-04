#!/usr/bin/env python
"""
Test script to build and train a simplified segmentation model with minimal data.
This is for debugging the ultrasound defect detection pipeline.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import glob
import json
import time
import argparse
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config

# Set memory growth for GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except Exception as e:
        print(f"Error setting memory growth: {e}")


def build_simple_unet(input_shape=(224, 224, 3)):
    """
    Build a lightweight U-Net model for segmentation.

    Args:
        input_shape (tuple): Input shape

    Returns:
        tf.keras.Model: Compiled model
    """
    # Input layer
    inputs = Input(input_shape)

    # Encoder path (downsampling)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bridge
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)

    # Decoder path (upsampling)
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Conv2D(64, 2, activation='relu', padding='same')(up1)
    merge1 = Concatenate()([conv2, up1])
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(merge1)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Conv2D(32, 2, activation='relu', padding='same')(up2)
    merge2 = Concatenate()([conv1, up2])
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(merge2)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def prepare_test_dataset(image_dir, mask_dir, img_size=(224, 224), batch_size=8, limit=None):
    """
    Prepare a small test dataset from images and masks.

    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        img_size (tuple): Target image size
        batch_size (int): Batch size
        limit (int): Maximum number of samples to use

    Returns:
        tuple: (train_ds, val_ds) TensorFlow datasets
    """
    # Get list of images
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    image_files = [f for f in image_files if not f.endswith('_mask.png')]

    if limit:
        image_files = image_files[:limit]

    print(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")

    # Create lists for images and masks
    image_paths = []
    mask_paths = []

    # Match images with masks
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")

        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)

    print(f"Found {len(image_paths)} image-mask pairs")

    if len(image_paths) == 0:
        raise ValueError(f"No matching image-mask pairs found")

    # Convert to numpy arrays
    images = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

            # Load and preprocess mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = (mask > 128).astype(np.float32)  # Binarize and convert to [0,1]
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

            images.append(img)
            masks.append(mask)
        except Exception as e:
            print(f"Error processing {img_path} or {mask_path}: {e}")

    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)

    # Split into training and validation sets (80/20)
    split_idx = int(len(images) * 0.8)
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]

    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    train_ds = train_ds.shuffle(len(train_images)).batch(batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds


def visualize_predictions(model, image_dir, mask_dir, output_dir, num_samples=5):
    """
    Make predictions on sample images and visualize the results.

    Args:
        model (tf.keras.Model): Trained model
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    image_files = [f for f in image_files if not f.endswith('_mask.png')]

    if len(image_files) == 0:
        print(f"No image files found in {image_dir}")
        return

    # Sample images
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

    for img_path in sample_files:
        try:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")

            if not os.path.exists(mask_path):
                print(f"No mask found for {base_name}")
                continue

            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (224, 224))
            img_normalized = img_resized.astype(np.float32) / 255.0

            # Load ground truth mask
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask_resized = cv2.resize(gt_mask, (224, 224))
            gt_mask_binary = (gt_mask_resized > 128).astype(np.uint8) * 255

            # Make prediction
            input_batch = np.expand_dims(img_normalized, axis=0)
            pred_mask = model.predict(input_batch)[0, :, :, 0]

            # Convert prediction to binary mask
            pred_binary = (pred_mask > 0.5).astype(np.uint8) * 255

            # Create visualization
            plt.figure(figsize=(15, 5))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_resized)
            plt.title('Original Image')
            plt.axis('off')

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask_binary, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_binary, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            # Save figure
            output_path = os.path.join(output_dir, f"{base_name}_prediction.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            # Create overlay visualization
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Image with ground truth overlay
            ax[0].imshow(img_resized)
            ax[0].imshow(gt_mask_binary, alpha=0.3, cmap='Reds')
            ax[0].set_title('Ground Truth Overlay')
            ax[0].axis('off')

            # Image with prediction overlay
            ax[1].imshow(img_resized)
            ax[1].imshow(pred_binary, alpha=0.3, cmap='Blues')
            ax[1].set_title('Prediction Overlay')
            ax[1].axis('off')

            # Image with both overlays
            ax[2].imshow(img_resized)
            # Red for ground truth, blue for prediction
            gt_overlay = np.zeros((*gt_mask_binary.shape, 4), dtype=np.uint8)
            gt_overlay[..., 0] = 255  # Red channel
            gt_overlay[..., 3] = (gt_mask_binary > 0) * 100  # Alpha channel

            pred_overlay = np.zeros((*pred_binary.shape, 4), dtype=np.uint8)
            pred_overlay[..., 2] = 255  # Blue channel
            pred_overlay[..., 3] = (pred_binary > 0) * 100  # Alpha channel

            ax[2].imshow(gt_overlay)
            ax[2].imshow(pred_overlay)
            ax[2].set_title('Combined Overlay')
            ax[2].axis('off')

            # Save overlay figure
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            plt.tight_layout()
            plt.savefig(overlay_path)
            plt.close()

            print(f"Created visualization for {base_name}")

        except Exception as e:
            print(f"Error visualizing {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test segmentation model")
    parser.add_argument("--image_dir", default=None,
                        help="Directory containing images")
    parser.add_argument("--mask_dir", default=None,
                        help="Directory containing masks")
    parser.add_argument("--unified_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory with unified data")
    parser.add_argument("--output_dir", default=os.path.join(project_root, "test_results"),
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum number of samples to use")
    parser.add_argument("--model_path", default=None,
                        help="Path to save model")

    args = parser.parse_args()

    # Set default directories if not provided
    if args.image_dir is None:
        args.image_dir = os.path.join(args.unified_dir, "images")

    if args.mask_dir is None:
        args.mask_dir = os.path.join(args.unified_dir, "masks")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Testing segmentation model with limited data...")
    print(f"Image directory: {args.image_dir}")
    print(f"Mask directory: {args.mask_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Sample limit: {args.limit}")

    # Prepare dataset
    try:
        train_ds, val_ds = prepare_test_dataset(
            args.image_dir,
            args.mask_dir,
            img_size=(224, 224),
            batch_size=args.batch_size,
            limit=args.limit
        )
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return 1

    # Build model
    model = build_simple_unet(input_shape=(224, 224, 3))
    model.summary()

    # Train model
    print("Training model...")
    start_time = time.time()

    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping]
    )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot
    history_path = os.path.join(args.output_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(history_path)
    plt.close()

    print(f"Training history saved to {history_path}")

    # Save the model if requested
    if args.model_path:
        model.save(args.model_path)
        print(f"Model saved to {args.model_path}")
    else:
        model_path = os.path.join(args.output_dir, "simple_unet_model.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # Visualize predictions
    print("Generating predictions and visualizations...")
    visualize_predictions(
        model,
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        num_samples=5
    )

    print("Testing completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())