#!/usr/bin/env python
"""
Debug script to test MAT file conversion and troubleshoot issues.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import glob
import argparse
import cv2
import time
import json
import shutil  # Add missing shutil import here

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config


def inspect_mat_directory(input_dir, max_files=5, verbose=True):
    """
    Inspect MAT files in a directory to understand their structure.

    Args:
        input_dir (str): Directory containing MAT files
        max_files (int): Maximum number of files to inspect
        verbose (bool): Whether to print detailed information

    Returns:
        dict: Structure information about the MAT files
    """
    print(f"Inspecting MAT files in {input_dir}...")

    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist.")
        return None

    # Find MAT files
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

    if not mat_files:
        print(f"Error: No MAT files found in {input_dir}")
        return None

    print(f"Found {len(mat_files)} MAT files, inspecting up to {max_files}...")

    # Sample a few files
    sample_files = mat_files[:max_files]

    # Collect structure information
    structures = {}

    for file_path in sample_files:
        try:
            # Load MAT file
            mat_data = loadmat(file_path)

            # Extract basic information
            file_info = {
                "filename": os.path.basename(file_path),
                "keys": list(mat_data.keys()),
                "shapes": {}
            }

            # Check key shapes
            for key in mat_data.keys():
                if key not in ['__header__', '__version__', '__globals__']:
                    file_info["shapes"][key] = mat_data[key].shape

            # Store in structures dictionary
            structures[os.path.basename(file_path)] = file_info

            if verbose:
                print(f"\nFile: {os.path.basename(file_path)}")
                print("  Keys:", file_info["keys"])
                print("  Shapes:")
                for key, shape in file_info["shapes"].items():
                    print(f"    {key}: {shape}")

                # More detailed inspection of key fields
                if 'imgData' in mat_data:
                    img_data = mat_data['imgData']
                    print(f"  imgData min/max: {img_data.min()}/{img_data.max()}")
                    print(f"  imgData dtype: {img_data.dtype}")

                if 'deadElements' in mat_data:
                    dead_elements = mat_data['deadElements']
                    print(f"  deadElements unique values: {np.unique(dead_elements)}")
                    print(f"  deadElements sum: {np.sum(dead_elements)} (indicates # of dead elements)")
                    print(f"  deadElements indices: {np.where(dead_elements.flatten() == 1)[0].tolist()}")

        except Exception as e:
            print(f"Error inspecting {file_path}: {e}")

    # Analyze consistency across files
    print("\nConsistency Analysis:")

    # Check if all files have the same keys
    all_keys = set()
    for file_info in structures.values():
        all_keys.update(file_info["keys"])

    consistent_keys = True
    for file_info in structures.values():
        missing_keys = [key for key in all_keys if
                        key not in file_info["keys"] and key not in ['__header__', '__version__', '__globals__']]
        if missing_keys:
            consistent_keys = False
            print(f"  File {file_info['filename']} is missing keys: {missing_keys}")

    if consistent_keys:
        print("  All files have consistent keys.")

    # Check if shapes are consistent
    shape_consistency = {}
    for key in [k for k in all_keys if k not in ['__header__', '__version__', '__globals__']]:
        shapes = [info["shapes"].get(key) for info in structures.values() if key in info["shapes"]]
        if len(set(shapes)) == 1:
            shape_consistency[key] = "Consistent"
        else:
            shape_consistency[key] = "Inconsistent"

    print("  Shape consistency:")
    for key, status in shape_consistency.items():
        print(f"    {key}: {status}")

    return structures


def test_convert_mat_to_png(input_dir, output_dir, max_files=5):
    """
    Test converting MAT files to PNG images.

    Args:
        input_dir (str): Directory containing MAT files
        output_dir (str): Directory to save PNG images
        max_files (int): Maximum number of files to convert

    Returns:
        int: Number of files successfully converted
    """
    print(f"Testing MAT to PNG conversion...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find MAT files
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

    if not mat_files:
        print(f"Error: No MAT files found in {input_dir}")
        return 0

    # Sample a few files
    sample_files = mat_files[:max_files]

    successful = 0
    for mat_path in sample_files:
        try:
            # Load MAT file
            mat_data = loadmat(mat_path)

            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(mat_path))[0]

            # Extract relevant data
            try:
                img_data = mat_data['imgData']  # B-mode image data (usually 118 x 128)
                dead_elements = mat_data['deadElements'].flatten()  # Status of transducers (1 = disabled)
            except KeyError as e:
                print(f"Missing expected field in {mat_path}: {e}")
                continue

            # Save as PNG
            png_path = os.path.join(output_dir, f"{filename}.png")
            plt.figure(figsize=(8, 8), dpi=100)
            plt.imshow(img_data, cmap='gray', aspect='auto')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Save dead elements information
            dead_elements_path = os.path.join(output_dir, f"{filename}_dead_elements.npy")
            np.save(dead_elements_path, dead_elements)

            # Save raw image data
            img_data_path = os.path.join(output_dir, f"{filename}_img_data.npy")
            np.save(img_data_path, img_data)

            print(f"Successfully converted {filename} to PNG")
            successful += 1

            # Create a visualization of the dead elements
            vis_path = os.path.join(output_dir, f"{filename}_visualization.png")
            visualize_dead_elements(img_data, dead_elements, vis_path)

        except Exception as e:
            print(f"Error converting {mat_path}: {e}")

    print(f"Successfully converted {successful}/{len(sample_files)} MAT files to PNG")
    return successful


def visualize_dead_elements(img_data, dead_elements, output_path):
    """
    Visualize the original ultrasound image with dead elements highlighted.

    Args:
        img_data (numpy.ndarray): Original ultrasound image data (2D array)
        dead_elements (numpy.ndarray): 1D array indicating dead elements (1 = dead)
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(10, 8))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_data, cmap='gray', aspect='auto')
    plt.title("Original B-mode Image")
    plt.axis('off')

    # Create a version with dead elements highlighted
    plt.subplot(1, 2, 2)
    plt.imshow(img_data, cmap='gray', aspect='auto')

    # Highlight dead elements
    height = img_data.shape[0]
    for i, is_dead in enumerate(dead_elements):
        if is_dead == 1:
            plt.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.5)

    plt.title("Dead Elements Highlighted")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def test_create_segmentation_masks(input_dir, output_dir, max_files=5):
    """
    Test creating segmentation masks based on dead elements information.

    Args:
        input_dir (str): Directory containing PNGs and dead_elements.npy files
        output_dir (str): Directory to save segmentation masks
        max_files (int): Maximum number of masks to create

    Returns:
        int: Number of masks successfully created
    """
    print(f"Testing segmentation mask creation...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find dead elements files
    dead_element_files = glob.glob(os.path.join(input_dir, "*_dead_elements.npy"))

    if not dead_element_files:
        print(f"Error: No dead elements files found in {input_dir}")
        return 0

    # Sample a few files
    sample_files = dead_element_files[:max_files]

    successful = 0
    for dead_file in sample_files:
        try:
            # Load dead elements data
            dead_elements = np.load(dead_file)

            # Get corresponding image data
            img_data_file = dead_file.replace('_dead_elements.npy', '_img_data.npy')

            if os.path.exists(img_data_file):
                img_data = np.load(img_data_file)
                mask_shape = img_data.shape
            else:
                # Default shape if image data isn't available
                mask_shape = (118, 128)

            # Create a mask based on dead elements
            mask = np.zeros(mask_shape, dtype=np.uint8)

            # Map dead elements to columns in the image
            for i, is_dead in enumerate(dead_elements):
                if is_dead == 1 and i < mask_shape[1]:  # If element is disabled and index is valid
                    # Mark the entire column as defective
                    mask[:, i] = 255

            # Save the mask as PNG
            base_filename = os.path.basename(dead_file).replace('_dead_elements.npy', '')
            mask_path = os.path.join(output_dir, f"{base_filename}_mask.png")

            plt.figure(figsize=(8, 8), dpi=100)
            plt.imshow(mask, cmap='binary', aspect='auto')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Also save as NumPy array
            np.save(os.path.join(output_dir, f"{base_filename}_mask.npy"), mask)

            # Create an overlay of mask on image
            overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.png")
            overlay_mask_on_image(
                os.path.join(input_dir, f"{base_filename}.png"),
                mask_path,
                overlay_path
            )

            print(f"Successfully created mask for {base_filename}")
            successful += 1

        except Exception as e:
            print(f"Error creating mask: {e}")

    print(f"Successfully created {successful}/{len(sample_files)} segmentation masks")
    return successful


def overlay_mask_on_image(image_path, mask_path, output_path):
    """
    Overlay a segmentation mask on an image for visualization.

    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the segmentation mask
        output_path (str): Path to save the overlay image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load image and mask
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not load mask {mask_path}")
            return False

        # Resize mask to match image if needed
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Create overlay by adding a red channel where the mask is white
        overlay = image.copy()
        overlay[:, :, 2] = np.maximum(overlay[:, :, 2], mask)  # Red channel

        # Blend with original
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        # Save the result
        cv2.imwrite(output_path, result)
        return True

    except Exception as e:
        print(f"Error creating overlay: {e}")
        return False


def test_organize_unified(img_dir, mask_dir, output_dir, max_files=5):
    """
    Test organizing data with a unified approach.

    Args:
        img_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        output_dir (str): Directory for organized data
        max_files (int): Maximum number of files to process

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Testing unified data organization...")

    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Get list of images
    images = glob.glob(os.path.join(img_dir, "*.png"))
    images = [f for f in images if
              not f.endswith('_mask.png') and not f.endswith('_visualization.png') and not f.endswith('_overlay.png')]

    if not images:
        print(f"Error: No images found in {img_dir}")
        return False

    # Sample a few files
    sample_files = images[:max_files]

    # Metadata dictionary
    metadata = {}

    successful = 0
    for img_file in sample_files:
        try:
            base_name = os.path.splitext(os.path.basename(img_file))[0]

            # Find corresponding dead elements file
            dead_elements_file = os.path.join(img_dir, f"{base_name}_dead_elements.npy")

            if not os.path.exists(dead_elements_file):
                print(f"Warning: No dead elements found for {base_name}")
                continue

            # Load dead elements
            dead_elements = np.load(dead_elements_file)

            # Determine pattern characteristics
            num_dead = int(np.sum(dead_elements))
            dead_indices = np.where(dead_elements == 1)[0].tolist()

            # Check if the elements are contiguous
            contiguous = False
            if num_dead > 1:
                # Elements are contiguous if they form a continuous sequence
                for i in range(len(dead_indices) - 1):
                    if dead_indices[i + 1] - dead_indices[i] != 1:
                        contiguous = False
                        break
                else:
                    contiguous = True

            # Create pattern description
            if num_dead == 0:
                pattern = "all_enabled"
            else:
                pattern = f"{num_dead}_{'contiguous' if contiguous else 'random'}"

            # Create metadata for this sample
            metadata[base_name] = {
                "dead_elements": dead_indices,
                "num_dead": num_dead,
                "contiguous": contiguous,
                "pattern": pattern
            }

            # Copy image and mask to output directory
            # Copy image
            dest_img = os.path.join(output_dir, "images", os.path.basename(img_file))
            if not os.path.exists(dest_img):
                shutil.copy(img_file, dest_img)

            # Copy mask
            mask_file = os.path.join(mask_dir, f"{base_name}_mask.png")
            if os.path.exists(mask_file):
                dest_mask = os.path.join(output_dir, "masks", f"{base_name}_mask.png")
                if not os.path.exists(dest_mask):
                    shutil.copy(mask_file, dest_mask)

            successful += 1

        except Exception as e:
            print(f"Error organizing {base_name}: {e}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully organized {successful}/{len(sample_files)} files")
    print(f"Metadata saved to {metadata_path}")

    return successful > 0


def main():
    parser = argparse.ArgumentParser(description="Debug MAT file conversion and segmentation")
    parser.add_argument("--raw_dir", default=config.RAW_DATA_DIR,
                        help="Directory containing MAT files")
    parser.add_argument("--processed_dir", default=config.PROCESSED_DATA_DIR,
                        help="Directory to save processed PNG images")
    parser.add_argument("--mask_dir", default=config.MASK_DIR,
                        help="Directory to save masks")
    parser.add_argument("--unified_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory for unified data organization")
    parser.add_argument("--max_files", type=int, default=5,
                        help="Maximum number of files to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--step", type=int, default=0,
                        help="Specific step to run (0=all, 1=inspect, 2=convert, 3=mask, 4=organize)")

    args = parser.parse_args()

    print(f"Debug script starting...")
    print(f"Project root: {project_root}")
    print(f"Raw data directory: {args.raw_dir}")
    print(f"Processed directory: {args.processed_dir}")
    print(f"Mask directory: {args.mask_dir}")
    print(f"Unified directory: {args.unified_dir}")

    # Ensure directories exist
    for directory in [
        args.raw_dir, args.processed_dir, args.mask_dir,
        args.unified_dir
    ]:
        os.makedirs(directory, exist_ok=True)

    if args.step == 0 or args.step == 1:
        print("\n=== Step 1: Inspecting MAT files ===")
        structures = inspect_mat_directory(args.raw_dir, args.max_files, args.verbose)

        if structures is None:
            print("Failed to inspect MAT files. Please check the raw data directory.")
            return 1

    if args.step == 0 or args.step == 2:
        print("\n=== Step 2: Testing MAT to PNG conversion ===")
        converted = test_convert_mat_to_png(args.raw_dir, args.processed_dir, args.max_files)

        if converted == 0:
            print("Failed to convert any MAT files. Please check the raw data format.")
            if args.step == 2:
                return 1

    if args.step == 0 or args.step == 3:
        print("\n=== Step 3: Testing segmentation mask creation ===")
        created = test_create_segmentation_masks(args.processed_dir, args.mask_dir, args.max_files)

        if created == 0:
            print("Failed to create any segmentation masks. Please check the processed data.")
            if args.step == 3:
                return 1

    if args.step == 0 or args.step == 4:
        print("\n=== Step 4: Testing unified data organization ===")
        organized = test_organize_unified(args.processed_dir, args.mask_dir, args.unified_dir, args.max_files)

        if not organized:
            print("Failed to organize data. Please check the processed and mask directories.")
            if args.step == 4:
                return 1

    print("\nDebug script completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())