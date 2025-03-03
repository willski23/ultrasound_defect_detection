"""
Module for augmenting unified data to prevent overfitting.
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import matplotlib.pyplot as plt
import time
import json
import random

def augment_unified_data(input_dir, output_dir, num_augmentations=5):
    """
    Augment images and their corresponding masks from the unified data structure.

    Args:
        input_dir (str): Base directory containing unified data
        output_dir (str): Base directory for augmented unified data
        num_augmentations (int): Number of augmentations per image

    Returns:
        dict: Statistics about the augmentation process and updated metadata
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file {metadata_path} not found. Please run organize_unified.py first.")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create new metadata for augmented data
    augmented_metadata = metadata.copy()

    # Input and output paths
    image_dir = os.path.join(input_dir, "images")
    mask_dir = os.path.join(input_dir, "masks")

    # Statistics dictionary
    stats = {
        "original": 0,
        "augmented": 0,
        "by_pattern": {}
    }

    print(f"Augmenting unified data...")

    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # Copy original files first (no augmentation)
    for image_file in tqdm(image_files, desc="Copying original files"):
        base_name = os.path.splitext(image_file)[0]

        # Copy original image
        shutil.copy(
            os.path.join(image_dir, image_file),
            os.path.join(output_dir, "images", image_file)
        )

        # Copy corresponding mask
        mask_file = f"{base_name}_mask.png"
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path):
            shutil.copy(
                mask_path,
                os.path.join(output_dir, "masks", mask_file)
            )

        # Update statistics
        stats["original"] += 1

        # Update pattern statistics if applicable
        if base_name in metadata:
            pattern = metadata[base_name].get("pattern", "unknown")
            if pattern not in stats["by_pattern"]:
                stats["by_pattern"][pattern] = {"original": 0, "augmented": 0}
            stats["by_pattern"][pattern]["original"] += 1

    # Define augmentation parameters
    data_gen_args = dict(
        rotation_range=10,            # Rotation within 10 degrees
        width_shift_range=0.1,        # Horizontal shift
        height_shift_range=0.1,       # Vertical shift
        brightness_range=[0.8, 1.2],  # Brightness adjustment
        shear_range=5,                # Shear transformation
        zoom_range=0.1,               # Zoom in/out
        horizontal_flip=True,         # Horizontal flip
        fill_mode='nearest'           # How to fill newly created pixels
    )

    # Create generators with the same seed for images and masks
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Process each image and its mask
    for image_file in tqdm(image_files, desc="Augmenting images"):
        base_name = os.path.splitext(image_file)[0]

        # Skip if no metadata
        if base_name not in metadata:
            print(f"Warning: No metadata found for {image_file}, skipping augmentation")
            continue

        # Get metadata for this image
        img_metadata = metadata[base_name]
        pattern = img_metadata.get("pattern", "unknown")

        # Load image and mask
        image_path = os.path.join(image_dir, image_file)
        mask_file = f"{base_name}_mask.png"
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {image_file}, skipping augmentation")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, 0)  # Add batch dimension

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue

        mask = np.expand_dims(mask, 0)  # Add batch dimension
        mask = np.expand_dims(mask, 3)  # Add channel dimension

        # Generate augmented images with seeds
        for aug_idx in range(num_augmentations):
            # Use a different seed for each augmentation but same for image and mask
            aug_seed = int(time.time()) + aug_idx + hash(image_file) % 10000

            # New filename for augmented image
            aug_base_name = f"{base_name}_aug{aug_idx}"
            aug_image_file = f"{aug_base_name}.png"
            aug_mask_file = f"{aug_base_name}_mask.png"

            # Output paths
            aug_image_path = os.path.join(output_dir, "images", aug_image_file)
            aug_mask_path = os.path.join(output_dir, "masks", aug_mask_file)

            # Generate and save augmented image
            image_generator = image_datagen.flow(
                image,
                batch_size=1,
                seed=aug_seed,
                save_to_dir=None
            )
            aug_image = next(image_generator)[0].astype(np.uint8)
            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            # Generate and save augmented mask
            mask_generator = mask_datagen.flow(
                mask,
                batch_size=1,
                seed=aug_seed,
                save_to_dir=None
            )
            aug_mask = next(mask_generator)[0, :, :, 0].astype(np.uint8)
            cv2.imwrite(aug_mask_path, aug_mask)

            # Copy the original metadata for the augmented image
            augmented_metadata[aug_base_name] = img_metadata.copy()

            # Update statistics
            stats["augmented"] += 1

            # Update pattern statistics
            if pattern in stats["by_pattern"]:
                stats["by_pattern"][pattern]["augmented"] += 1

    # Save augmented metadata
    augmented_metadata_path = os.path.join(output_dir, "metadata.json")
    with open(augmented_metadata_path, 'w') as f:
        json.dump(augmented_metadata, f, indent=2)

    print("\nAugmentation complete.")
    print(f"Original images: {stats['original']}")
    print(f"Augmented images: {stats['augmented']}")
    print(f"Total images: {stats['original'] + stats['augmented']}")

    # Print pattern statistics
    print("\nAugmentation by pattern:")
    for pattern, pattern_stats in stats["by_pattern"].items():
        print(f"  {pattern}: {pattern_stats['original']} original, {pattern_stats['augmented']} augmented")

    print(f"\nAugmented data is saved to {output_dir}")
    print(f"Augmented metadata is saved to {augmented_metadata_path}")

    return stats, augmented_metadata

def visualize_unified_augmentations(input_dir, output_dir, num_samples=3):
    """
    Visualize augmentations from the unified data structure.

    Args:
        input_dir (str): Directory containing unified data
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of sample images to visualize

    Returns:
        list: Paths to visualization images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file {metadata_path} not found.")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get unique patterns from metadata
    patterns = set()
    for info in metadata.values():
        patterns.add(info.get("pattern", "unknown"))

    # Create visualization paths
    vis_paths = []

    # Visualize augmentations for each pattern
    for pattern in patterns:
        print(f"Visualizing augmentations for pattern: {pattern}")

        # Find samples with this pattern
        samples = [base_name for base_name, info in metadata.items()
                  if info.get("pattern") == pattern and not "_aug" in base_name]

        if not samples:
            continue

        # Select random samples
        selected_samples = random.sample(samples, min(num_samples, len(samples)))

        for sample in selected_samples:
            # Find augmented versions
            augmentations = [name for name in metadata.keys()
                           if name.startswith(f"{sample}_aug")]

            if not augmentations:
                continue

            # Sort augmentations by index
            augmentations.sort(key=lambda x: int(x.split("_aug")[1]))

            # Limit to 3 augmentations for display
            augmentations = augmentations[:3]

            # Create visualization
            vis_path = create_augmentation_visualization(
                input_dir, sample, augmentations, pattern,
                os.path.join(output_dir, f"{pattern}_{sample}_augmentations.png")
            )
            vis_paths.append(vis_path)

    return vis_paths

def create_augmentation_visualization(data_dir, original_name, augmentation_names, pattern, output_path):
    """
    Create a visualization of original and augmented images with defect overlays.

    Args:
        data_dir (str): Directory containing data
        original_name (str): Base name of original image
        augmentation_names (list): List of augmentation base names
        pattern (str): Defect pattern name
        output_path (str): Path to save visualization

    Returns:
        str: Path to saved visualization
    """
    # Load metadata
    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    # Load original image and mask
    original_img_path = os.path.join(data_dir, "images", f"{original_name}.png")
    original_mask_path = os.path.join(data_dir, "masks", f"{original_name}_mask.png")

    original_img = cv2.imread(original_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)

    # Get defective elements
    defective_elements = metadata[original_name].get("dead_elements", [])

    # Create figure
    num_augmentations = len(augmentation_names)
    plt.figure(figsize=(12, 4 * (num_augmentations + 1)))

    # Plot original image and mask
    plt.subplot(num_augmentations + 1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Original Image ({pattern})")
    plt.axis('off')

    plt.subplot(num_augmentations + 1, 2, 2)
    plt.imshow(original_img)

    # Highlight defective elements
    img_width = original_img.shape[1]
    element_width = img_width / 128  # Assuming 128 elements

    for element_idx in defective_elements:
        x_start = element_idx * element_width
        x_end = (element_idx + 1) * element_width
        plt.axvspan(x_start, x_end, color='red', alpha=0.5)

    plt.title(f"Defective Elements: {defective_elements}")
    plt.axis('off')

    # Plot augmentations
    for i, aug_name in enumerate(augmentation_names):
        # Load augmented image and mask
        aug_img_path = os.path.join(data_dir, "images", f"{aug_name}.png")
        aug_mask_path = os.path.join(data_dir, "masks", f"{aug_name}_mask.png")

        aug_img = cv2.imread(aug_img_path)
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)

        aug_mask = cv2.imread(aug_mask_path, cv2.IMREAD_GRAYSCALE)

        # Plot augmented image
        plt.subplot(num_augmentations + 1, 2, (i+1)*2 + 1)
        plt.imshow(aug_img)
        plt.title(f"Augmentation {i+1}")
        plt.axis('off')

        # Plot augmented image with defects highlighted
        plt.subplot(num_augmentations + 1, 2, (i+1)*2 + 2)
        plt.imshow(aug_img)

        # Highlight defective elements (should be the same as original)
        for element_idx in defective_elements:
            x_start = element_idx * element_width
            x_end = (element_idx + 1) * element_width
            plt.axvspan(x_start, x_end, color='red', alpha=0.5)

        plt.title(f"Same Defective Elements")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config
    import argparse

    parser = argparse.ArgumentParser(description="Augment unified data")
    parser.add_argument("--input_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory with unified data")
    parser.add_argument("--output_dir", default=os.path.join(config.DATA_DIR, "augmented_unified"),
                        help="Directory to save augmented unified data")
    parser.add_argument("--num_augmentations", type=int, default=config.NUM_AUGMENTATIONS_PER_IMAGE,
                        help="Number of augmentations per image")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Visualize augmentations")

    args = parser.parse_args()

    # Augment data
    stats, metadata = augment_unified_data(
        args.input_dir,
        args.output_dir,
        args.num_augmentations
    )

    # Visualize augmentations
    if args.visualize:
        vis_dir = os.path.join(project_root, "notebooks", "unified_augmentations")
        os.makedirs(vis_dir, exist_ok=True)

        vis_paths = visualize_unified_augmentations(
            args.output_dir,
            vis_dir
        )

        if vis_paths:
            print(f"\nCreated {len(vis_paths)} augmentation visualizations in {vis_dir}")