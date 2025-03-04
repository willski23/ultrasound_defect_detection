#!/usr/bin/env python
"""
Script optimized for GTX 1060 GPU (Compute Capability 6.1).
Resumes training from a checkpoint using GPU acceleration but without mixed precision.
"""
import os
import sys
import argparse
import time
import pickle
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

import config

# Print TensorFlow and GPU info
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
print("GPU devices:", physical_devices)
print("Is GPU available:", len(physical_devices) > 0)

# Enable memory growth to avoid allocating all GPU memory at once
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled on all GPUs")
    except Exception as e:
        print(f"Error setting memory growth: {e}")

    # We're NOT using mixed precision since GTX 1060 has compute capability 6.1
    print("Mixed precision disabled (not recommended for GTX 1060)")


def load_dataset_from_info(dataset_info_path, img_size=(224, 224), batch_size=16):
    """
    Recreate datasets from saved dataset info.

    Args:
        dataset_info_path (str): Path to dataset info pickle file
        img_size (tuple): Image size for model input
        batch_size (int): Batch size for training

    Returns:
        tuple: (train_ds, val_ds, test_ds) TensorFlow datasets
    """
    print(f"Loading dataset info from {dataset_info_path}")

    with open(dataset_info_path, 'rb') as f:
        dataset_info = pickle.load(f)

    def load_and_preprocess(img_path, mask_path):
        """Load and preprocess image and mask."""
        # Load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0

        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, img_size)
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.where(mask > 0.5, 1.0, 0.0)

        return img, mask

    # Create datasets
    train_images = dataset_info['train']['images']
    train_masks = dataset_info['train']['masks']
    val_images = dataset_info['validation']['images']
    val_masks = dataset_info['validation']['masks']
    test_images = dataset_info['test']['images']
    test_masks = dataset_info['test']['masks']

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")

    # Create efficient datasets with parallel processing
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=min(len(train_images), 1000))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, dataset_info


def resume_training(model_path, train_ds, val_ds, epochs=20, checkpoint_dir=None):
    """
    Resume training from a checkpoint with GPU optimizations for GTX 1060.

    Args:
        model_path (str): Path to existing model checkpoint
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        epochs (int): Number of epochs to train
        checkpoint_dir (str): Directory to save checkpoints

    Returns:
        tuple: (model, history, model_path) trained model and history
    """
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Create a timestamp for this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"ultrasound_gtx1060_{timestamp}"

    # Setup callbacks
    callbacks = []

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Model checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        new_model_path = os.path.join(checkpoint_dir, f"{model_name}.h5")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=new_model_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        callbacks.append(checkpoint)
    else:
        new_model_path = None

    # TensorBoard logging
    log_dir = os.path.join(checkpoint_dir, f"logs_{model_name}")
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard)

    # Resume training
    print(f"Resuming training for {epochs} epochs with GTX 1060 optimizations...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model if not already saved by checkpoint
    if new_model_path and not os.path.exists(new_model_path):
        model.save(new_model_path)
        print(f"Model saved to {new_model_path}")

    # Plot training history
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f"{model_name}_history.png"))
    plt.close()

    return model, history, new_model_path or model_path


def main():
    parser = argparse.ArgumentParser(description="Resume training with GTX 1060 GPU optimizations")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model checkpoint")
    parser.add_argument("--dataset_info", type=str, default=None,
                        help="Path to dataset info pickle file")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (GTX 1060 recommended: 16-32)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train")

    args = parser.parse_args()

    # Find latest model if not specified
    if args.model_path is None:
        model_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "*.h5"))
        if model_files:
            args.model_path = max(model_files, key=os.path.getmtime)
            print(f"Using latest model: {args.model_path}")
        else:
            print("No existing model found. Please specify a model path.")
            return 1

    # Find dataset info if not specified
    if args.dataset_info is None:
        dataset_info_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "*dataset_info.pkl"))
        if dataset_info_files:
            args.dataset_info = max(dataset_info_files, key=os.path.getmtime)
            print(f"Using latest dataset info: {args.dataset_info}")
        else:
            print("No dataset info found. Please specify a dataset info path.")
            return 1

    # Load datasets
    train_ds, val_ds, test_ds, dataset_info = load_dataset_from_info(
        args.dataset_info,
        batch_size=args.batch_size
    )

    # Resume training
    model, history, model_path = resume_training(
        args.model_path,
        train_ds,
        val_ds,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )

    print(f"Training completed successfully. Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())