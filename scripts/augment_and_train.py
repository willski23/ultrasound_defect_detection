#!/usr/bin/env python
"""
Script to augment unified data and train the model.
This script follows after the batch processor to complete the pipeline.
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import gc
import shutil
import traceback
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.absolute())
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


def augment_data(input_dir, output_dir, num_augmentations=3, batch_size=50, verbose=False):
    """
    Augment the unified data for model training.

    Args:
        input_dir (str): Directory with unified data
        output_dir (str): Directory to save augmented data
        num_augmentations (int): Number of augmentations per image
        batch_size (int): Size of batches to process
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (original_count, augmented_count) counts of images
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} not found")
        return 0, 0

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create new metadata for augmented data
    augmented_metadata = metadata.copy()

    # Get list of images
    image_files = glob.glob(os.path.join(input_dir, "images", "*.png"))

    # Filter out visualization and overlay images
    image_files = [f for f in image_files if not "_visualization.png" in f and not "_overlay.png" in f]

    original_count = len(image_files)
    augmented_count = 0

    print(f"Found {original_count} original images to augment")

    # Define augmentation parameters
    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        shear_range=5,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # First, copy all original files
    print("Copying original files...")
    for image_file in tqdm(image_files, desc="Copying originals"):
        base_name = os.path.splitext(os.path.basename(image_file))[0]

        # Copy original image
        shutil.copy(
            image_file,
            os.path.join(output_dir, "images", os.path.basename(image_file))
        )

        # Copy corresponding mask
        mask_file = os.path.join(input_dir, "masks", f"{base_name}_mask.png")
        if os.path.exists(mask_file):
            shutil.copy(
                mask_file,
                os.path.join(output_dir, "masks", f"{base_name}_mask.png")
            )

    # Process in batches
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    print(f"Augmenting images in {len(batches)} batches...")

    # Create augmentations for each batch
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)}...")

        for image_file in tqdm(batch, desc=f"Augmenting batch {batch_idx + 1}"):
            base_name = os.path.splitext(os.path.basename(image_file))[0]

            # Skip if no metadata
            if base_name not in metadata:
                if verbose:
                    print(f"No metadata for {base_name}, skipping")
                continue

            # Skip visualization images
            if '_visualization' in base_name or '_overlay' in base_name:
                continue

            # Get metadata for this image
            img_metadata = metadata[base_name]

            # Load image and mask
            image = cv2.imread(image_file)
            if image is None:
                if verbose:
                    print(f"Failed to load image {image_file}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_file = os.path.join(input_dir, "masks", f"{base_name}_mask.png")
            if not os.path.exists(mask_file):
                if verbose:
                    print(f"No mask for {base_name}, skipping")
                continue

            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                if verbose:
                    print(f"Failed to load mask {mask_file}")
                continue

            # Create augmentations
            for aug_idx in range(num_augmentations):
                try:
                    # Use a different seed for each augmentation
                    seed = int(time.time()) + aug_idx + hash(image_file) % 10000

                    # Prepare images for augmentation
                    image_batch = np.expand_dims(image, 0)
                    mask_batch = np.expand_dims(np.expand_dims(mask, 0), -1)

                    # Create data generators
                    image_datagen = ImageDataGenerator(**data_gen_args)
                    mask_datagen = ImageDataGenerator(**data_gen_args)

                    # Generate augmented images
                    image_generator = image_datagen.flow(
                        image_batch,
                        batch_size=1,
                        seed=seed
                    )
                    mask_generator = mask_datagen.flow(
                        mask_batch,
                        batch_size=1,
                        seed=seed
                    )

                    # Get augmented images
                    aug_image = next(image_generator)[0].astype(np.uint8)
                    aug_mask = next(mask_generator)[0, :, :, 0].astype(np.uint8)

                    # New filename for augmented image
                    aug_base_name = f"{base_name}_aug{aug_idx}"
                    aug_image_file = f"{aug_base_name}.png"
                    aug_mask_file = f"{aug_base_name}_mask.png"

                    # Save augmented image and mask
                    cv2.imwrite(
                        os.path.join(output_dir, "images", aug_image_file),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(
                        os.path.join(output_dir, "masks", aug_mask_file),
                        aug_mask
                    )

                    # Copy the metadata for the augmented image
                    augmented_metadata[aug_base_name] = img_metadata.copy()

                    augmented_count += 1

                except Exception as e:
                    if verbose:
                        print(f"Error augmenting {base_name}: {e}")
                        print(traceback.format_exc())

        # Save metadata after each batch
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(augmented_metadata, f, indent=2)

        # Clean up resources
        gc.collect()

    print(f"Augmentation complete.")
    print(f"Original images: {original_count}")
    print(f"Augmented images: {augmented_count}")
    print(f"Total images: {original_count + augmented_count}")

    return original_count, augmented_count


def build_unet_model(input_shape=(224, 224, 3), model_type="resnet50", freeze_encoder=True):
    """
    Build segmentation model based on chosen architecture.

    Args:
        input_shape (tuple): Input image shape
        model_type (str): Model type ("resnet50" or "unet")
        freeze_encoder (bool): Whether to freeze encoder weights

    Returns:
        tf.keras.Model: Compiled model
    """
    if model_type == "resnet50":
        # Use ResNet50 as encoder
        from tensorflow.keras.applications import ResNet50

        # Input
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Use ResNet50 as encoder (without top layers)
        resnet_base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )

        # Freeze encoder if specified
        if freeze_encoder:
            for layer in resnet_base.layers:
                layer.trainable = False

        # Get skip connections from specific layers
        skips = [
            resnet_base.get_layer('conv1_relu').output,  # 128x128
            resnet_base.get_layer('conv2_block3_out').output,  # 64x64
            resnet_base.get_layer('conv3_block4_out').output,  # 32x32
            resnet_base.get_layer('conv4_block6_out').output  # 16x16
        ]

        # Get bottleneck output
        x = resnet_base.get_layer('conv5_block3_out').output  # 8x8

        # Decoder (upsampling) path
        for i in range(len(skips)):
            # Upsample
            x = tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=3, strides=2, padding='same')(x)

            # Get skip connection and resize
            skip = skips[-(i + 1)]
            skip_height, skip_width = skip.shape[1], skip.shape[2]

            # Resize x to match skip connection dimensions
            x = tf.keras.layers.Resizing(
                height=skip_height,
                width=skip_width,
                interpolation="bilinear"
            )(x)

            # Concatenate with skip connection
            x = tf.keras.layers.Concatenate()([x, skip])

            # Apply convolutions
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)

        # Final upsampling to original image size
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

        # Output layer
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    else:  # unet
        # Use a simpler U-Net architecture
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
        from tensorflow.keras.layers import Concatenate, BatchNormalization

        # Input
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Encoder
        x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        skip1 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        skip2 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        skip3 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Bridge
        x = Conv2D(512, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Decoder
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, 2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, skip3])
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, 2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, skip2])
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, 2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, skip1])
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Output layer
        outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryIoU(threshold=0.5)
        ]
    )

    return model


def prepare_dataset(data_dir, img_size=(224, 224), batch_size=16, validation_split=0.15, test_split=0.15):
    """
    Prepare dataset for training.

    Args:
        data_dir (str): Directory with augmented data
        img_size (tuple): Target image size
        batch_size (int): Batch size
        validation_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing

    Returns:
        tuple: (train_ds, val_ds, test_ds, dataset_info) prepared datasets
    """
    from sklearn.model_selection import train_test_split

    # Get paths
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    # Get all image and mask files
    image_files = glob.glob(os.path.join(images_dir, "*.png"))

    # Filter out visualization images
    image_files = [f for f in image_files if not "_visualization.png" in f and not "_overlay.png" in f]

    print(f"Found {len(image_files)} image files")

    # Match images with masks
    all_images = []
    all_masks = []

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")

        if os.path.exists(mask_path):
            all_images.append(img_path)
            all_masks.append(mask_path)

    print(f"Found {len(all_images)} matching image-mask pairs")

    # Split into training, validation, and test sets
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        all_images, all_masks,
        test_size=test_split,
        random_state=42
    )

    # Split validation set from remaining training data
    val_size = validation_split / (1 - test_split)
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks,
        test_size=val_size,
        random_state=42
    )

    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"Test set: {len(test_images)} samples")

    # Create TensorFlow datasets
    def load_and_preprocess(img_path, mask_path):
        # Load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]

        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, img_size)
        mask = tf.cast(mask, tf.float32) / 255.0  # Normalize to [0,1]
        mask = tf.where(mask > 0.5, 1.0, 0.0)  # Binarize

        return img, mask

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(train_images))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)

    # Prepare dataset info for later use
    dataset_info = {
        'train': {
            'images': train_images,
            'masks': train_masks
        },
        'validation': {
            'images': val_images,
            'masks': val_masks
        },
        'test': {
            'images': test_images,
            'masks': test_masks
        }
    }

    return train_ds, val_ds, test_ds, dataset_info


def train_model(model, train_ds, val_ds, epochs=20, checkpoint_dir=None, name_prefix="ultrasound_model"):
    """
    Train the segmentation model.

    Args:
        model (tf.keras.Model): Model to train
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        epochs (int): Number of epochs to train
        checkpoint_dir (str): Directory to save checkpoints
        name_prefix (str): Prefix for model name

    Returns:
        tuple: (model, history, model_path) trained model and history
    """
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{name_prefix}_{timestamp}"

    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"{model_name}.h5")
    else:
        model_path = None

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
    if model_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        callbacks.append(checkpoint)

    # Train model
    print(f"Training model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model
    if model_path and not os.path.exists(model_path):
        model.save(model_path)
        print(f"Model saved to {model_path}")

    return model, history, model_path


def plot_training_history(history, output_dir):
    """
    Plot training history.

    Args:
        history (tf.keras.callbacks.History): Training history
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

    print(f"Training history plot saved to {os.path.join(output_dir, 'training_history.png')}")


def main():
    parser = argparse.ArgumentParser(description="Augment data and train ultrasound segmentation model")
    parser.add_argument("--unified_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory with unified data")
    parser.add_argument("--augmented_dir", default=config.AUGMENTED_UNIFIED_DATA_DIR,
                        help="Directory for augmented data")
    parser.add_argument("--checkpoint_dir", default=config.CHECKPOINT_DIR,
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_augmentations", type=int, default=3,
                        help="Number of augmentations per image")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for training")
    parser.add_argument("--model_type", choices=["resnet50", "unet"], default="resnet50",
                        help="Type of model architecture to use")
    parser.add_argument("--skip_augmentation", action="store_true",
                        help="Skip data augmentation (use if already augmented)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip model training")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    args = parser.parse_args()

    # Step 1: Augment data (if not skipped)
    if not args.skip_augmentation:
        print("\n=== Step 1: Augmenting Data ===")
        augment_data(
            args.unified_dir,
            args.augmented_dir,
            args.num_augmentations,
            batch_size=50,
            verbose=args.verbose
        )
    else:
        print("\n=== Skipping Data Augmentation ===")

    # Step 2: Train model (if not skipped)
    if not args.skip_training:
        print("\n=== Step 2: Training Model ===")

        # Prepare dataset
        print("Preparing dataset...")
        train_ds, val_ds, test_ds, dataset_info = prepare_dataset(
            args.augmented_dir,
            img_size=(224, 224),
            batch_size=args.batch_size
        )

        # Save dataset info for later evaluation
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        dataset_info_path = os.path.join(args.checkpoint_dir, "dataset_info.pkl")
        import pickle
        with open(dataset_info_path, 'wb') as f:
            pickle.dump(dataset_info, f)

        # Build model
        print(f"Building {args.model_type} model...")
        model = build_unet_model(
            input_shape=(224, 224, 3),
            model_type=args.model_type,
            freeze_encoder=True
        )

        # Print model summary
        model.summary()

        # Train model
        trained_model, history, model_path = train_model(
            model,
            train_ds,
            val_ds,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            name_prefix=f"ultrasound_{args.model_type}"
        )

        # Plot training history
        plot_training_history(history, args.checkpoint_dir)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trained_model.evaluate(test_ds, verbose=1)

        # Save test results
        results_path = os.path.join(args.checkpoint_dir, "test_results.txt")
        with open(results_path, 'w') as f:
            f.write("Test Results:\n")
            for name, value in zip(trained_model.metrics_names, test_results):
                f.write(f"{name}: {value}\n")
                print(f"{name}: {value}")
    else:
        print("\n=== Skipping Model Training ===")

    print("\nProcess completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())