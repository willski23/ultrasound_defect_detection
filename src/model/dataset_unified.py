"""
Module for preparing unified datasets for model training.
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

def prepare_unified_dataset(base_dir, img_size=(224, 224), batch_size=16,
                           val_split=0.15, test_split=0.15, seed=42,
                           balance_classes=True):
    """
    Prepare a TensorFlow dataset from the unified data organization.

    Args:
        base_dir (str): Base directory containing unified data
        img_size (tuple): Target image size (height, width)
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        test_split (float): Fraction of data to use for testing
        seed (int): Random seed for reproducibility
        balance_classes (bool): Whether to balance classes during training

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, metadata)
    """
    print("Preparing unified dataset...")

    # Paths for unified data
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    metadata_path = os.path.join(base_dir, "metadata.json")

    # Verify directories and files exist
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory {images_dir} does not exist.")

    if not os.path.exists(masks_dir):
        raise ValueError(f"Masks directory {masks_dir} does not exist.")

    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file {metadata_path} does not exist.")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded metadata for {len(metadata)} samples.")

    # Get all image and mask files
    image_files = glob.glob(os.path.join(images_dir, "*.png"))
    print(f"Found {len(image_files)} image files.")

    # Match image files with metadata
    all_images = []
    all_masks = []
    sample_weights = []
    all_metadata = []

    for img_path in image_files:
        base_name = os.path.basename(img_path).split('.')[0]

        # Skip if no metadata
        if base_name not in metadata:
            print(f"Warning: No metadata for {base_name}")
            continue

        # Find corresponding mask
        mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")
        if not os.path.exists(mask_path):
            print(f"Warning: No mask for {base_name}")
            continue

        # Get metadata for this sample
        sample_metadata = metadata[base_name]

        # Determine sample weight for class balancing
        if balance_classes:
            # Assign higher weight to samples with defects
            weight = 2.0 if sample_metadata["num_dead"] > 0 else 1.0
        else:
            weight = 1.0

        all_images.append(img_path)
        all_masks.append(mask_path)
        sample_weights.append(weight)
        all_metadata.append(sample_metadata)

    print(f"Matched {len(all_images)} image-mask pairs with metadata.")

    # Split into training, validation, and test sets
    train_val_images, test_images, train_val_masks, test_masks, \
    train_val_weights, test_weights, train_val_meta, test_meta = train_test_split(
        all_images, all_masks, sample_weights, all_metadata,
        test_size=test_split, random_state=seed, stratify=[meta["pattern"] for meta in all_metadata]
    )

    # Compute validation split from the remaining training data
    val_size = val_split / (1 - test_split)

    train_images, val_images, train_masks, val_masks, \
    train_weights, val_weights, train_meta, val_meta = train_test_split(
        train_val_images, train_val_masks, train_val_weights, train_val_meta,
        test_size=val_size, random_state=seed,
        stratify=[meta["pattern"] for meta in train_val_meta]
    )

    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"Test set: {len(test_images)} samples")

    # Create TensorFlow datasets
    train_ds = create_tf_dataset(
        train_images, train_masks, img_size, batch_size,
        weights=train_weights, shuffle=True
    )

    val_ds = create_tf_dataset(
        val_images, val_masks, img_size, batch_size,
        weights=val_weights, shuffle=False
    )

    test_ds = create_tf_dataset(
        test_images, test_masks, img_size, batch_size,
        weights=test_weights, shuffle=False
    )

    # Prepare dataset splits information for later use
    dataset_info = {
        'train': {
            'images': train_images,
            'masks': train_masks,
            'metadata': train_meta,
            'weights': train_weights
        },
        'validation': {
            'images': val_images,
            'masks': val_masks,
            'metadata': val_meta,
            'weights': val_weights
        },
        'test': {
            'images': test_images,
            'masks': test_masks,
            'metadata': test_meta,
            'weights': test_weights
        },
        'all_metadata': metadata
    }

    return train_ds, val_ds, test_ds, dataset_info


def create_tf_dataset(image_paths, mask_paths, img_size, batch_size, weights=None, shuffle=False):
    """
    Create a TensorFlow dataset from image and mask paths, with caching for performance.
    """
    # Use standard dataset creation without weights
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Cache file paths before preprocessing
    dataset = dataset.cache(f"/tmp/tf_cache_{hash(str(image_paths))}")

    # Map the preprocessing function
    dataset = dataset.map(
        lambda img_path, mask_path: preprocess_image_mask(img_path, mask_path, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def preprocess_image_mask(image_path, mask_path, img_size):
    """
    Preprocess an image and its mask.

    Args:
        image_path (str): Path to image
        mask_path (str): Path to mask
        img_size (tuple): Target image size (height, width)

    Returns:
        tuple: (preprocessed_image, preprocessed_mask)
    """
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]

    # Read mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.cast(mask, tf.float32) / 255.0  # Normalize to [0,1]

    # Threshold mask for binary segmentation
    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return image, mask

def analyze_dataset_composition(dataset_info):
    """
    Analyze the composition of the dataset splits.

    Args:
        dataset_info (dict): Dataset information

    Returns:
        dict: Statistics about dataset composition
    """
    stats = {}

    # Analyze each split
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset_info[split_name]
        metadata_list = split_data['metadata']

        # Count samples by pattern
        pattern_counts = {}
        for meta in metadata_list:
            pattern = meta['pattern']
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
            pattern_counts[pattern] += 1

        # Count samples with/without defects
        with_defects = sum(1 for meta in metadata_list if meta['num_dead'] > 0)
        without_defects = len(metadata_list) - with_defects

        stats[split_name] = {
            'total': len(metadata_list),
            'with_defects': with_defects,
            'without_defects': without_defects,
            'patterns': pattern_counts
        }

    return stats

def visualize_dataset_samples(dataset, num_samples=3, output_path=None):
    """
    Visualize samples from a dataset.

    Args:
        dataset (tf.data.Dataset): Dataset to visualize
        num_samples (int): Number of samples to visualize
        output_path (str): Path to save visualization image

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    # Get samples from the dataset
    samples = next(iter(dataset.take(1)))

    # Check if we have sample weights
    has_weights = len(samples) == 3

    if has_weights:
        images, masks, weights = samples
    else:
        images, masks = samples
        weights = None

    # Ensure we don't try to display more samples than we have
    num_samples = min(num_samples, images.shape[0])

    # Create a figure
    plt.figure(figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(images[i])
        title = f"Image {i+1}"
        if weights is not None:
            title += f" (Weight: {weights[i].numpy():.1f})"
        plt.title(title)
        plt.axis('off')

        # Mask
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks[i, :, :, 0], cmap='gray')
        plt.title(f"Mask {i+1}")
        plt.axis('off')

        # Overlay
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(images[i])
        plt.imshow(masks[i, :, :, 0], cmap='jet', alpha=0.3)
        plt.title(f"Overlay {i+1}")
        plt.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config
    import os

    # Define unified data directory
    unified_data_dir = os.path.join(config.DATA_DIR, "unified")

    if not os.path.exists(unified_data_dir):
        print(f"Unified data directory {unified_data_dir} does not exist.")
        print("Please run the organize_unified.py script first.")
        sys.exit(1)

    # Prepare dataset
    train_ds, val_ds, test_ds, dataset_info = prepare_unified_dataset(
        unified_data_dir,
        img_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE
    )

    # Analyze dataset composition
    stats = analyze_dataset_composition(dataset_info)
    print("\nDataset composition:")
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.capitalize()} set ({split_stats['total']} samples):")
        print(f"  With defects: {split_stats['with_defects']} ({split_stats['with_defects']/split_stats['total']*100:.1f}%)")
        print(f"  Without defects: {split_stats['without_defects']} ({split_stats['without_defects']/split_stats['total']*100:.1f}%)")
        print(f"  Pattern distribution:")
        for pattern, count in split_stats['patterns'].items():
            print(f"    {pattern}: {count} ({count/split_stats['total']*100:.1f}%)")

    # Visualize some samples
    os.makedirs("notebooks", exist_ok=True)
    visualize_dataset_samples(train_ds,
                             output_path=os.path.join("notebooks", "unified_dataset_samples.png"))