#!/usr/bin/env python
"""
Integrated Master Pipeline for ultrasound defect detection.
This script combines batch processing for data conversion and organization
with efficient model training to handle large datasets.
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
from pathlib import Path
import cv2
import json
import gc
import shutil
import traceback
import pickle
from tqdm import tqdm
from datetime import datetime
import logging

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config

# Setup logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"integrated_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set memory growth for GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except Exception as e:
        logging.error(f"Error setting memory growth: {e}")


def print_step_header(title):
    """Print a formatted step header."""
    logging.info("\n" + "=" * 80)
    logging.info(f"STEP: {title}")
    logging.info("=" * 80)
    print("\n" + "=" * 80)
    print(f"STEP: {title}")
    print("=" * 80)


def batch_convert_mat_to_png(mat_dir, output_dir, batch_size=20, max_files=None):
    """
    Convert MAT files to PNG images in batches.

    Args:
        mat_dir (str): Directory with MAT files
        output_dir (str): Directory to save processed images
        batch_size (int): Number of files to process in each batch
        max_files (int): Maximum number of files to process

    Returns:
        int: Number of successfully converted files
    """
    print_step_header("Converting MAT Files to PNG")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of MAT files
    mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))
    if max_files:
        mat_files = mat_files[:max_files]

    total_files = len(mat_files)
    logging.info(f"Found {total_files} MAT files")
    print(f"Found {total_files} MAT files")

    if total_files == 0:
        logging.error(f"No MAT files found in {mat_dir}")
        return 0

    # Process in batches
    success_count = 0
    for batch_idx in range(0, len(mat_files), batch_size):
        batch = mat_files[batch_idx:batch_idx + batch_size]
        logging.info(
            f"Processing batch {batch_idx // batch_size + 1}/{(len(mat_files) + batch_size - 1) // batch_size}")
        print(f"Processing batch {batch_idx // batch_size + 1}/{(len(mat_files) + batch_size - 1) // batch_size}")

        for mat_file in tqdm(batch, desc="Converting to PNG"):
            try:
                # Load MAT file
                mat_data = loadmat(mat_file)

                # Extract filename
                filename = os.path.splitext(os.path.basename(mat_file))[0]

                # Extract data
                img_data = mat_data['imgData']
                dead_elements = mat_data['deadElements'].flatten()

                # Save as PNG
                png_path = os.path.join(output_dir, f"{filename}.png")
                plt.figure(figsize=(8, 8), dpi=100)
                plt.imshow(img_data, cmap='gray', aspect='auto')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                # Save metadata
                np.save(os.path.join(output_dir, f"{filename}_dead_elements.npy"), dead_elements)
                np.save(os.path.join(output_dir, f"{filename}_img_data.npy"), img_data)

                success_count += 1

            except Exception as e:
                logging.error(f"Error processing {mat_file}: {e}")
                logging.error(traceback.format_exc())

        # Clean up resources
        gc.collect()
        plt.close('all')

    logging.info(f"Successfully converted {success_count}/{total_files} MAT files")
    print(f"Successfully converted {success_count}/{total_files} MAT files")

    return success_count


def batch_create_masks(processed_dir, mask_dir, batch_size=50, max_files=None):
    """
    Create segmentation masks in batches.

    Args:
        processed_dir (str): Directory with processed images and dead elements
        mask_dir (str): Directory to save masks
        batch_size (int): Number of files to process in each batch
        max_files (int): Maximum number of files to process

    Returns:
        int: Number of successfully created masks
    """
    print_step_header("Creating Segmentation Masks")

    # Create output directory
    os.makedirs(mask_dir, exist_ok=True)

    # Get list of dead elements files
    dead_element_files = glob.glob(os.path.join(processed_dir, "*_dead_elements.npy"))
    if max_files:
        dead_element_files = dead_element_files[:max_files]

    total_files = len(dead_element_files)
    logging.info(f"Found {total_files} files for mask creation")
    print(f"Found {total_files} files for mask creation")

    if total_files == 0:
        logging.error(f"No dead elements files found in {processed_dir}")
        return 0

    # Process in batches
    success_count = 0
    for batch_idx in range(0, len(dead_element_files), batch_size):
        batch = dead_element_files[batch_idx:batch_idx + batch_size]
        logging.info(
            f"Processing batch {batch_idx // batch_size + 1}/{(len(dead_element_files) + batch_size - 1) // batch_size}")
        print(
            f"Processing batch {batch_idx // batch_size + 1}/{(len(dead_element_files) + batch_size - 1) // batch_size}")

        for dead_file in tqdm(batch, desc="Creating masks"):
            try:
                # Get base filename
                base_name = os.path.basename(dead_file).replace('_dead_elements.npy', '')

                # Load dead elements data
                dead_elements = np.load(dead_file)

                # Get corresponding image data
                img_data_file = os.path.join(processed_dir, f"{base_name}_img_data.npy")

                if os.path.exists(img_data_file):
                    img_data = np.load(img_data_file)
                    mask_shape = img_data.shape
                else:
                    mask_shape = (118, 128)  # Default shape

                # Create mask
                mask = np.zeros(mask_shape, dtype=np.uint8)

                # Mark dead elements
                for i, is_dead in enumerate(dead_elements):
                    if is_dead == 1 and i < mask_shape[1]:
                        mask[:, i] = 255

                # Save mask
                mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
                plt.figure(figsize=(8, 8), dpi=100)
                plt.imshow(mask, cmap='binary', aspect='auto')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                success_count += 1

            except Exception as e:
                logging.error(f"Error creating mask for {dead_file}: {e}")
                logging.error(traceback.format_exc())

        # Clean up resources
        gc.collect()
        plt.close('all')

    logging.info(f"Successfully created {success_count}/{total_files} masks")
    print(f"Successfully created {success_count}/{total_files} masks")

    return success_count


def batch_organize_unified(processed_dir, mask_dir, unified_dir, batch_size=50, max_files=None):
    """
    Organize data in unified structure in batches.

    Args:
        processed_dir (str): Directory with processed images
        mask_dir (str): Directory with masks
        unified_dir (str): Directory for unified organization
        batch_size (int): Number of files to process in each batch
        max_files (int): Maximum number of files to process

    Returns:
        tuple: (success_count, metadata) count of organized files and metadata
    """
    print_step_header("Organizing Data with Unified Approach")

    # Create output directories
    os.makedirs(os.path.join(unified_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, "masks"), exist_ok=True)

    # Get list of images
    image_files = glob.glob(os.path.join(processed_dir, "*.png"))
    image_files = [f for f in image_files if
                   not "_mask.png" in f and not "_visualization.png" in f and not "_overlay.png" in f]

    if max_files:
        image_files = image_files[:max_files]

    total_files = len(image_files)
    logging.info(f"Found {total_files} images for organization")
    print(f"Found {total_files} images for organization")

    if total_files == 0:
        logging.error(f"No images found in {processed_dir}")
        return 0, {}

    # Metadata dictionary
    metadata = {}

    # Statistics
    stats = {
        "with_defects": 0,
        "without_defects": 0,
        "patterns": {}
    }

    # Process in batches
    success_count = 0
    for batch_idx in range(0, len(image_files), batch_size):
        batch = image_files[batch_idx:batch_idx + batch_size]
        logging.info(
            f"Processing batch {batch_idx // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}")
        print(f"Processing batch {batch_idx // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}")

        for img_file in tqdm(batch, desc="Organizing files"):
            try:
                # Get base filename
                base_name = os.path.splitext(os.path.basename(img_file))[0]

                # Find corresponding dead elements file
                dead_elements_file = os.path.join(processed_dir, f"{base_name}_dead_elements.npy")

                if not os.path.exists(dead_elements_file):
                    logging.warning(f"No dead elements found for {base_name}")
                    continue

                # Load dead elements
                dead_elements = np.load(dead_elements_file)

                # Determine pattern characteristics
                num_dead = int(np.sum(dead_elements))
                dead_indices = np.where(dead_elements == 1)[0].tolist()

                # Check if elements are contiguous
                contiguous = False
                if num_dead > 1:
                    contiguous = True
                    for i in range(len(dead_indices) - 1):
                        if dead_indices[i + 1] - dead_indices[i] != 1:
                            contiguous = False
                            break

                # Create pattern description
                if num_dead == 0:
                    pattern = "all_enabled"
                else:
                    pattern = f"{num_dead}_{'contiguous' if contiguous else 'random'}"

                # Update statistics
                if pattern not in stats["patterns"]:
                    stats["patterns"][pattern] = 0
                stats["patterns"][pattern] += 1

                if num_dead > 0:
                    stats["with_defects"] += 1
                else:
                    stats["without_defects"] += 1

                # Create metadata for this sample
                metadata[base_name] = {
                    "dead_elements": dead_indices,
                    "num_dead": num_dead,
                    "contiguous": contiguous,
                    "pattern": pattern
                }

                # Copy files
                shutil.copy(
                    img_file,
                    os.path.join(unified_dir, "images", os.path.basename(img_file))
                )

                mask_file = os.path.join(mask_dir, f"{base_name}_mask.png")
                if os.path.exists(mask_file):
                    shutil.copy(
                        mask_file,
                        os.path.join(unified_dir, "masks", f"{base_name}_mask.png")
                    )

                success_count += 1

            except Exception as e:
                logging.error(f"Error organizing {img_file}: {e}")
                logging.error(traceback.format_exc())

        # Save metadata after each batch
        metadata_path = os.path.join(unified_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Clean up resources
        gc.collect()

    # Print statistics
    logging.info(f"Organization statistics:")
    logging.info(f"  Total organized: {success_count}")
    logging.info(f"  With defects: {stats['with_defects']}")
    logging.info(f"  Without defects: {stats['without_defects']}")
    logging.info(f"  Patterns: {stats['patterns']}")

    print(f"Organization statistics:")
    print(f"  Total organized: {success_count}")
    print(f"  With defects: {stats['with_defects']}")
    print(f"  Without defects: {stats['without_defects']}")

    return success_count, metadata


def batch_augment_data(unified_dir, augmented_dir, num_augmentations=3, batch_size=50, max_files=None):
    """
    Augment data in batches.

    Args:
        unified_dir (str): Directory with unified data
        augmented_dir (str): Directory to save augmented data
        num_augmentations (int): Number of augmentations per image
        batch_size (int): Number of files to process in each batch
        max_files (int): Maximum number of files to process

    Returns:
        tuple: (original_count, augmented_count) counts of original and augmented images
    """
    print_step_header("Augmenting Data")

    # Create output directories
    os.makedirs(os.path.join(augmented_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(augmented_dir, "masks"), exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(unified_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        return 0, 0

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create augmented metadata
    augmented_metadata = metadata.copy()

    # Get list of images
    image_files = glob.glob(os.path.join(unified_dir, "images", "*.png"))
    image_files = [f for f in image_files if not "_visualization.png" in f and not "_overlay.png" in f]

    if max_files:
        image_files = image_files[:max_files]

    original_count = len(image_files)
    logging.info(f"Found {original_count} original images for augmentation")
    print(f"Found {original_count} original images for augmentation")

    # First, copy all original files
    logging.info("Copying original files...")
    print("Copying original files...")

    for img_file in tqdm(image_files, desc="Copying originals"):
        base_name = os.path.splitext(os.path.basename(img_file))[0]

        # Copy image
        shutil.copy(
            img_file,
            os.path.join(augmented_dir, "images", os.path.basename(img_file))
        )

        # Copy mask
        mask_file = os.path.join(unified_dir, "masks", f"{base_name}_mask.png")
        if os.path.exists(mask_file):
            shutil.copy(
                mask_file,
                os.path.join(augmented_dir, "masks", f"{base_name}_mask.png")
            )

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

    # Process in batches
    augmented_count = 0
    for batch_idx in range(0, len(image_files), batch_size):
        batch = image_files[batch_idx:batch_idx + batch_size]
        logging.info(
            f"Augmenting batch {batch_idx // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}")
        print(f"Augmenting batch {batch_idx // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}")

        for img_file in tqdm(batch, desc="Augmenting images"):
            try:
                base_name = os.path.splitext(os.path.basename(img_file))[0]

                # Skip if no metadata
                if base_name not in metadata:
                    logging.warning(f"No metadata for {base_name}, skipping")
                    continue

                # Get metadata
                img_metadata = metadata[base_name]

                # Load image and mask
                image = cv2.imread(img_file)
                if image is None:
                    logging.warning(f"Failed to load image {img_file}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                mask_file = os.path.join(unified_dir, "masks", f"{base_name}_mask.png")
                if not os.path.exists(mask_file):
                    logging.warning(f"No mask for {base_name}, skipping")
                    continue

                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    logging.warning(f"Failed to load mask {mask_file}")
                    continue

                # Create augmentations
                for aug_idx in range(num_augmentations):
                    # Use a different seed for each augmentation
                    seed = int(time.time()) + aug_idx + hash(img_file) % 10000

                    # Prepare for augmentation
                    image_batch = np.expand_dims(image, 0)
                    mask_batch = np.expand_dims(np.expand_dims(mask, 0), -1)

                    # Create generators
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

                    # Save augmented images
                    aug_base_name = f"{base_name}_aug{aug_idx}"
                    aug_image_file = f"{aug_base_name}.png"
                    aug_mask_file = f"{aug_base_name}_mask.png"

                    cv2.imwrite(
                        os.path.join(augmented_dir, "images", aug_image_file),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    )
                    cv2.imwrite(
                        os.path.join(augmented_dir, "masks", aug_mask_file),
                        aug_mask
                    )

                    # Update metadata
                    augmented_metadata[aug_base_name] = img_metadata.copy()

                    augmented_count += 1

            except Exception as e:
                logging.error(f"Error augmenting {img_file}: {e}")
                logging.error(traceback.format_exc())

        # Save metadata after each batch
        augmented_metadata_path = os.path.join(augmented_dir, "metadata.json")
        with open(augmented_metadata_path, 'w') as f:
            json.dump(augmented_metadata, f, indent=2)

        # Clean up resources
        gc.collect()

    logging.info(f"Data augmentation complete:")
    logging.info(f"  Original images: {original_count}")
    logging.info(f"  Augmented images: {augmented_count}")
    logging.info(f"  Total images: {original_count + augmented_count}")

    print(f"Data augmentation complete:")
    print(f"  Original images: {original_count}")
    print(f"  Augmented images: {augmented_count}")
    print(f"  Total images: {original_count + augmented_count}")

    return original_count, augmented_count


def prepare_dataset(data_dir, img_size=(224, 224), batch_size=16, validation_split=0.15, test_split=0.15):
    """
    Prepare TensorFlow datasets for training.

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

    print_step_header("Preparing Dataset")

    # Get paths
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    # Get image and mask files
    image_files = glob.glob(os.path.join(images_dir, "*.png"))
    image_files = [f for f in image_files if not "_visualization.png" in f and not "_overlay.png" in f]

    logging.info(f"Found {len(image_files)} image files")
    print(f"Found {len(image_files)} image files")

    # Match images with masks
    all_images = []
    all_masks = []

    for img_path in tqdm(image_files, desc="Matching images with masks"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")

        if os.path.exists(mask_path):
            all_images.append(img_path)
            all_masks.append(mask_path)

    logging.info(f"Found {len(all_images)} matching image-mask pairs")
    print(f"Found {len(all_images)} matching image-mask pairs")

    # Split into train, validation, and test sets
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        all_images, all_masks,
        test_size=test_split,
        random_state=42
    )

    val_size = validation_split / (1 - test_split)
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks,
        test_size=val_size,
        random_state=42
    )

    logging.info(f"Dataset splits:")
    logging.info(f"  Training: {len(train_images)} samples")
    logging.info(f"  Validation: {len(val_images)} samples")
    logging.info(f"  Test: {len(test_images)} samples")

    print(f"Dataset splits:")
    print(f"  Training: {len(train_images)} samples")
    print(f"  Validation: {len(val_images)} samples")
    print(f"  Test: {len(test_images)} samples")

    # Create TensorFlow datasets
    def load_and_preprocess(img_path, mask_path):
        """Load and preprocess image and mask."""
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

    # Create datasets
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

    # Prepare dataset info
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


def build_and_train_model(train_ds, val_ds, model_type="resnet50", img_size=(224, 224),
                          batch_size=16, epochs=20, checkpoint_dir=None):
    """
    Build and train the segmentation model.

    Args:
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        model_type (str): Model type ("resnet50" or "unet")
        img_size (tuple): Input image size
        batch_size (int): Batch size
        epochs (int): Number of training epochs
        checkpoint_dir (str): Directory to save checkpoints

    Returns:
        tuple: (model, history, model_path) trained model and history
    """
    print_step_header(f"Training {model_type.upper()} Model")

    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"ultrasound_{model_type}_{timestamp}"

    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"{model_name}.h5")
    else:
        model_path = None

    # Build model based on type
    if model_type == "resnet50":
        logging.info("Building ResNet50 model...")
        print("Building ResNet50 model...")

        # Import only when needed
        from tensorflow.keras.applications import ResNet50

        # Input layer
        inputs = tf.keras.layers.Input(shape=(*img_size, 3))

        # Use ResNet50 as encoder
        resnet_base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )

        # Freeze encoder
        for layer in resnet_base.layers:
            layer.trainable = False

        # Get skip connections
        skips = [
            resnet_base.get_layer('conv1_relu').output,
            resnet_base.get_layer('conv2_block3_out').output,
            resnet_base.get_layer('conv3_block4_out').output,
            resnet_base.get_layer('conv4_block6_out').output
        ]

        # Get bottleneck feature
        x = resnet_base.get_layer('conv5_block3_out').output

        # Decoder path
        for i in range(len(skips)):
            # Upsample
            x = tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=3, strides=2, padding='same')(x)

            # Get skip connection
            skip = skips[-(i + 1)]

            # Resize to match skip dimensions
            skip_height = skip.shape[1]
            skip_width = skip.shape[2]

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

        # Final upsampling
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

        # Output layer
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    else:  # unet
        logging.info("Building U-Net model...")
        print("Building U-Net model...")

        # Input layer
        inputs = tf.keras.layers.Input(shape=(*img_size, 3))

        # Encoder (downsampling) path
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # Bridge
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)

        # Decoder (upsampling) path
        up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
        up5 = tf.keras.layers.BatchNormalization()(up5)
        merge5 = tf.keras.layers.Concatenate()([conv3, up5])
        conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)

        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
        up6 = tf.keras.layers.BatchNormalization()(up6)
        merge6 = tf.keras.layers.Concatenate()([conv2, up6])
        conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)

        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
        up7 = tf.keras.layers.BatchNormalization()(up7)
        merge7 = tf.keras.layers.Concatenate()([conv1, up7])
        conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
        conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
        conv7 = tf.keras.layers.BatchNormalization()(conv7)

        # Output layer
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryIoU(threshold=0.5)
        ]
    )

    # Show model summary
    logging.info("Model summary:")
    model.summary(print_fn=logging.info)
    print("Model built successfully. Starting training...")

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
    if model_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        callbacks.append(checkpoint)

    # Train model
    logging.info(f"Training model for {epochs} epochs...")
    print(f"Training model for {epochs} epochs...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model if not already saved by checkpoint
    if model_path and not os.path.exists(model_path):
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        print(f"Model saved to {model_path}")

    # Plot training history
    if checkpoint_dir:
        history_path = os.path.join(checkpoint_dir, f"{model_name}_history.png")
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

        plt.tight_layout()
        plt.savefig(history_path)
        plt.close()

        logging.info(f"Training history plot saved to {history_path}")
        print(f"Training history plot saved to {history_path}")

    return model, history, model_path


def evaluate_model(model, test_ds, dataset_info, output_dir):
    """
    Evaluate the trained model.

    Args:
        model (tf.keras.Model): Trained model
        test_ds (tf.data.Dataset): Test dataset
        dataset_info (dict): Dataset information
        output_dir (str): Directory to save evaluation results

    Returns:
        dict: Evaluation metrics
    """
    print_step_header("Evaluating Model")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Standard evaluation
    logging.info("Running standard evaluation...")
    print("Running standard evaluation...")

    results = model.evaluate(test_ds, verbose=1)

    # Create metrics dictionary
    metrics = {}
    for i, name in enumerate(model.metrics_names):
        metrics[name] = float(results[i])
        logging.info(f"  {name}: {results[i]:.4f}")
        print(f"  {name}: {results[i]:.4f}")

    # Visualize some predictions
    logging.info("Generating prediction visualizations...")
    print("Generating prediction visualizations...")

    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Get some test examples
    test_images = dataset_info['test']['images']
    test_masks = dataset_info['test']['masks']

    # Sample some images for visualization
    import random
    sample_indices = random.sample(range(len(test_images)), min(5, len(test_images)))

    for i, idx in enumerate(sample_indices):
        img_path = test_images[idx]
        mask_path = test_masks[idx]

        # Load image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, (224, 224))
        mask_binary = (mask_resized > 128).astype(np.uint8) * 255

        # Make prediction
        input_batch = np.expand_dims(img_normalized, axis=0)
        pred_mask = model.predict(input_batch)[0, :, :, 0]
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
        plt.imshow(mask_binary, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred_binary, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        # Save figure
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        vis_path = os.path.join(vis_dir, f"{base_name}_prediction.png")
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close()

        # Create overlay visualization
        plt.figure(figsize=(15, 5))

        # Image with ground truth overlay
        plt.subplot(1, 3, 1)
        plt.imshow(img_resized)
        plt.imshow(mask_binary, alpha=0.3, cmap='Reds')
        plt.title('Ground Truth Overlay')
        plt.axis('off')

        # Image with prediction overlay
        plt.subplot(1, 3, 2)
        plt.imshow(img_resized)
        plt.imshow(pred_binary, alpha=0.3, cmap='Blues')
        plt.title('Prediction Overlay')
        plt.axis('off')

        # Combined overlay - red for ground truth, blue for prediction
        plt.subplot(1, 3, 3)
        plt.imshow(img_resized)

        # Red for ground truth
        overlay_gt = np.zeros((*mask_binary.shape, 4), dtype=np.uint8)
        overlay_gt[..., 0] = 255  # Red channel
        overlay_gt[..., 3] = (mask_binary > 0) * 100  # Alpha channel

        # Blue for prediction
        overlay_pred = np.zeros((*pred_binary.shape, 4), dtype=np.uint8)
        overlay_pred[..., 2] = 255  # Blue channel
        overlay_pred[..., 3] = (pred_binary > 0) * 100  # Alpha channel

        plt.imshow(overlay_gt)
        plt.imshow(overlay_pred)
        plt.title('Combined Overlay')
        plt.axis('off')

        # Save overlay
        overlay_path = os.path.join(vis_dir, f"{base_name}_overlay.png")
        plt.tight_layout()
        plt.savefig(overlay_path)
        plt.close()

    logging.info(f"Evaluation visualizations saved to {vis_dir}")
    print(f"Evaluation visualizations saved to {vis_dir}")

    # Save metrics to file
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Evaluation metrics saved to {metrics_path}")
    print(f"Evaluation metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Integrated Pipeline for Ultrasound Defect Detection")
    parser.add_argument("--raw_dir", default=config.RAW_DATA_DIR,
                        help="Directory with raw MAT files")
    parser.add_argument("--processed_dir", default=config.PROCESSED_DATA_DIR,
                        help="Directory for processed data")
    parser.add_argument("--mask_dir", default=config.MASK_DIR,
                        help="Directory for masks")
    parser.add_argument("--unified_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory for unified organization")
    parser.add_argument("--augmented_dir", default=config.AUGMENTED_UNIFIED_DATA_DIR,
                        help="Directory for augmented data")
    parser.add_argument("--checkpoint_dir", default=config.CHECKPOINT_DIR,
                        help="Directory for model checkpoints")
    parser.add_argument("--output_dir", default=os.path.join(config.MODEL_DIR, "evaluation"),
                        help="Directory for evaluation results")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for processing (reduces memory usage)")
    parser.add_argument("--model_batch_size", type=int, default=16,
                        help="Batch size for model training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--model_type", choices=["resnet50", "unet"], default="resnet50",
                        help="Type of model architecture to use")
    parser.add_argument("--num_augmentations", type=int, default=3,
                        help="Number of augmentations per image")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip MAT to PNG conversion")
    parser.add_argument("--skip_masks", action="store_true",
                        help="Skip mask creation")
    parser.add_argument("--skip_organization", action="store_true",
                        help="Skip data organization")
    parser.add_argument("--skip_augmentation", action="store_true",
                        help="Skip data augmentation")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip model training")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip model evaluation")

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 80)
    print("ULTRASOUND DEFECT DETECTION - INTEGRATED PIPELINE")
    print("=" * 80)

    start_time = time.time()

    # Step 1: Convert MAT to PNG
    if not args.skip_conversion:
        converted_count = batch_convert_mat_to_png(
            args.raw_dir,
            args.processed_dir,
            batch_size=args.batch_size,
            max_files=args.max_files
        )
        if converted_count == 0:
            logging.error("No files were converted. Pipeline stopped.")
            return 1
    else:
        logging.info("Skipping MAT to PNG conversion as requested.")
        print("Skipping MAT to PNG conversion as requested.")

    # Step 2: Create masks
    if not args.skip_masks:
        mask_count = batch_create_masks(
            args.processed_dir,
            args.mask_dir,
            batch_size=args.batch_size,
            max_files=args.max_files
        )
        if mask_count == 0:
            logging.error("No masks were created. Pipeline stopped.")
            return 1
    else:
        logging.info("Skipping mask creation as requested.")
        print("Skipping mask creation as requested.")

    # Step 3: Organize data
    if not args.skip_organization:
        organized_count, metadata = batch_organize_unified(
            args.processed_dir,
            args.mask_dir,
            args.unified_dir,
            batch_size=args.batch_size,
            max_files=args.max_files
        )
        if organized_count == 0:
            logging.error("No files were organized. Pipeline stopped.")
            return 1
    else:
        logging.info("Skipping data organization as requested.")
        print("Skipping data organization as requested.")

    # Step 4: Augment data
    if not args.skip_augmentation:
        original_count, augmented_count = batch_augment_data(
            args.unified_dir,
            args.augmented_dir,
            args.num_augmentations,
            batch_size=args.batch_size,
            max_files=args.max_files
        )
        if original_count + augmented_count == 0:
            logging.error("No files were augmented. Pipeline stopped.")
            return 1
    else:
        logging.info("Skipping data augmentation as requested.")
        print("Skipping data augmentation as requested.")

    # Step 5: Train model
    if not args.skip_training:
        # Prepare dataset
        train_ds, val_ds, test_ds, dataset_info = prepare_dataset(
            args.augmented_dir,
            img_size=(224, 224),
            batch_size=args.model_batch_size
        )

        # Save dataset info
        dataset_info_path = os.path.join(args.checkpoint_dir, "dataset_info.pkl")
        with open(dataset_info_path, 'wb') as f:
            pickle.dump(dataset_info, f)

        # Build and train model
        model, history, model_path = build_and_train_model(
            train_ds,
            val_ds,
            model_type=args.model_type,
            img_size=(224, 224),
            batch_size=args.model_batch_size,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir
        )
    else:
        logging.info("Skipping model training as requested.")
        print("Skipping model training as requested.")

        # If skipping training, try to load existing model and dataset info
        model_path = None
        for path in glob.glob(os.path.join(args.checkpoint_dir, "*.h5")):
            model_path = path
            break

        if model_path:
            logging.info(f"Using existing model: {model_path}")
            print(f"Using existing model: {model_path}")
            model = tf.keras.models.load_model(model_path)

            # Load dataset info
            dataset_info_path = os.path.join(args.checkpoint_dir, "dataset_info.pkl")
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'rb') as f:
                    dataset_info = pickle.load(f)

                # Prepare test dataset
                test_ds = tf.data.Dataset.from_tensor_slices(
                    (dataset_info['test']['images'], dataset_info['test']['masks']))
                test_ds = test_ds.map(lambda img_path, mask_path: (
                    tf.cast(tf.image.resize(tf.image.decode_png(tf.io.read_file(img_path), channels=3), (224, 224)),
                            tf.float32) / 255.0,
                    tf.cast(tf.image.resize(tf.image.decode_png(tf.io.read_file(mask_path), channels=1), (224, 224)),
                            tf.float32) / 255.0
                ))
                test_ds = test_ds.batch(args.model_batch_size)
        else:
            logging.warning("No existing model found. Skipping evaluation step.")
            model = None
            dataset_info = None
            test_ds = None

    # Step 6: Evaluate model
    if not args.skip_evaluation and model is not None and test_ds is not None and dataset_info is not None:
        metrics = evaluate_model(
            model,
            test_ds,
            dataset_info,
            args.output_dir
        )
    else:
        if args.skip_evaluation:
            logging.info("Skipping model evaluation as requested.")
            print("Skipping model evaluation as requested.")
        elif model is None or test_ds is None or dataset_info is None:
            logging.warning("Cannot evaluate model: model, test dataset, or dataset info not available.")
            print("Cannot evaluate model: model, test dataset, or dataset info not available.")

    # Print pipeline summary
    elapsed_time = time.time() - start_time
    logging.info("\n" + "=" * 80)
    logging.info(f"PIPELINE COMPLETED in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    logging.info("=" * 80)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print("=" * 80)

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