#!/usr/bin/env python
"""
Batch processor for ultrasound MAT files.
This script processes MAT files in small batches with comprehensive logging
and resource cleanup between batches to prevent stalling.
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import cv2
import json
import gc
import shutil
import traceback
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

import config


def process_mat_file(mat_path, output_dir, verbose=False):
    """
    Process a single MAT file to PNG with metadata.

    Args:
        mat_path (str): Path to the MAT file
        output_dir (str): Directory to save outputs
        verbose (bool): Whether to print detailed progress

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(mat_path))[0]

        if verbose:
            print(f"Processing {filename}...")

        # Load MAT file
        mat_data = loadmat(mat_path)

        # Extract relevant data
        img_data = mat_data['imgData']  # B-mode image data
        dead_elements = mat_data['deadElements'].flatten()  # Status of transducers

        # Create output paths
        png_path = os.path.join(output_dir, f"{filename}.png")
        dead_elements_path = os.path.join(output_dir, f"{filename}_dead_elements.npy")
        img_data_path = os.path.join(output_dir, f"{filename}_img_data.npy")

        # Save as PNG (with resource cleanup)
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(img_data, cmap='gray', aspect='auto')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close('all')  # Close all figures to free memory

        # Save numpy arrays
        np.save(dead_elements_path, dead_elements)
        np.save(img_data_path, img_data)

        return True

    except Exception as e:
        if verbose:
            print(f"Error processing {mat_path}: {e}")
            print(traceback.format_exc())
        return False


def create_mask(base_name, processed_dir, mask_dir, verbose=False):
    """
    Create a segmentation mask for a single image.

    Args:
        base_name (str): Base name of the image
        processed_dir (str): Directory with processed data
        mask_dir (str): Directory to save masks
        verbose (bool): Whether to print detailed progress

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Paths
        dead_elements_path = os.path.join(processed_dir, f"{base_name}_dead_elements.npy")
        img_data_path = os.path.join(processed_dir, f"{base_name}_img_data.npy")
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")

        # Load data
        if not os.path.exists(dead_elements_path) or not os.path.exists(img_data_path):
            if verbose:
                print(f"Missing data files for {base_name}")
            return False

        dead_elements = np.load(dead_elements_path)
        img_data = np.load(img_data_path)

        # Create mask
        mask = np.zeros(img_data.shape, dtype=np.uint8)

        # Mark dead elements
        for i, is_dead in enumerate(dead_elements):
            if is_dead == 1 and i < img_data.shape[1]:
                mask[:, i] = 255

        # Save mask
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(mask, cmap='binary', aspect='auto')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
        plt.close('all')  # Close all figures

        return True

    except Exception as e:
        if verbose:
            print(f"Error creating mask for {base_name}: {e}")
            print(traceback.format_exc())
        return False


def organize_unified_file(base_name, processed_dir, mask_dir, unified_dir, metadata, verbose=False):
    """
    Organize a single file within the unified structure.

    Args:
        base_name (str): Base name of the image
        processed_dir (str): Directory with processed data
        mask_dir (str): Directory with masks
        unified_dir (str): Directory for unified organization
        metadata (dict): Metadata dictionary to update
        verbose (bool): Whether to print detailed progress

    Returns:
        dict: Updated metadata
    """
    try:
        # Paths
        img_path = os.path.join(processed_dir, f"{base_name}.png")
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        dead_elements_path = os.path.join(processed_dir, f"{base_name}_dead_elements.npy")

        # Check if all files exist
        if not os.path.exists(img_path) or not os.path.exists(mask_path) or not os.path.exists(dead_elements_path):
            if verbose:
                print(f"Missing files for {base_name}")
            return metadata

        # Load dead elements
        dead_elements = np.load(dead_elements_path)

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

        # Update metadata
        metadata[base_name] = {
            "dead_elements": dead_indices,
            "num_dead": num_dead,
            "contiguous": contiguous,
            "pattern": pattern
        }

        # Copy files
        dest_img = os.path.join(unified_dir, "images", f"{base_name}.png")
        dest_mask = os.path.join(unified_dir, "masks", f"{base_name}_mask.png")

        shutil.copy(img_path, dest_img)
        shutil.copy(mask_path, dest_mask)

        if verbose:
            print(f"Organized {base_name} ({pattern})")

        return metadata

    except Exception as e:
        if verbose:
            print(f"Error organizing {base_name}: {e}")
            print(traceback.format_exc())
        return metadata


def process_in_batches(
        mat_dir,
        processed_dir,
        mask_dir,
        unified_dir,
        batch_size=10,
        max_files=None,
        verbose=False
):
    """
    Process MAT files in small batches with progress reporting.

    Args:
        mat_dir (str): Directory with MAT files
        processed_dir (str): Directory for processed images
        mask_dir (str): Directory for masks
        unified_dir (str): Directory for unified organization
        batch_size (int): Number of files to process in each batch
        max_files (int): Maximum number of files to process
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (success_count, total_count) counts of processed files
    """
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.join(unified_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, "masks"), exist_ok=True)

    # Get list of MAT files
    mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))
    if max_files is not None:
        mat_files = mat_files[:max_files]

    total_files = len(mat_files)
    print(f"Found {total_files} MAT files to process")

    # Initialize counters and metadata
    success_count = 0
    metadata = {}

    # Process in batches
    for batch_start in tqdm(range(0, total_files, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_files)
        batch = mat_files[batch_start:batch_end]

        print(
            f"\nProcessing batch {batch_start // batch_size + 1}: files {batch_start + 1}-{batch_end} of {total_files}")

        # Step 1: Convert MAT to PNG
        print("Step 1: Converting MAT to PNG")
        for mat_file in tqdm(batch, desc="Converting MAT files"):
            if process_mat_file(mat_file, processed_dir, verbose):
                success_count += 1

        # Step 2: Create masks
        print("Step 2: Creating masks")
        batch_base_names = [os.path.splitext(os.path.basename(f))[0] for f in batch]
        for base_name in tqdm(batch_base_names, desc="Creating masks"):
            create_mask(base_name, processed_dir, mask_dir, verbose)

        # Step 3: Organize with unified approach
        print("Step 3: Organizing data")
        for base_name in tqdm(batch_base_names, desc="Organizing data"):
            metadata = organize_unified_file(
                base_name,
                processed_dir,
                mask_dir,
                unified_dir,
                metadata,
                verbose
            )

        # Save metadata after each batch
        metadata_path = os.path.join(unified_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Free memory
        gc.collect()
        plt.close('all')

        print(f"Batch complete. Processed {success_count}/{total_files} files so far.")

    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count}/{total_files} files.")

    return success_count, total_files


def main():
    parser = argparse.ArgumentParser(description="Batch process ultrasound MAT files")
    parser.add_argument("--mat_dir", default=config.RAW_DATA_DIR,
                        help="Directory with MAT files")
    parser.add_argument("--processed_dir", default=config.PROCESSED_DATA_DIR,
                        help="Directory for processed data")
    parser.add_argument("--mask_dir", default=config.MASK_DIR,
                        help="Directory for masks")
    parser.add_argument("--unified_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory for unified organization")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Number of files to process in each batch")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    args = parser.parse_args()

    start_time = time.time()

    success_count, total_count = process_in_batches(
        args.mat_dir,
        args.processed_dir,
        args.mask_dir,
        args.unified_dir,
        args.batch_size,
        args.max_files,
        args.verbose
    )

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    if success_count == total_count:
        print("All files processed successfully!")
        return 0
    else:
        print(f"Warning: Only {success_count}/{total_count} files were processed successfully.")
        return 1


if __name__ == "__main__":
    sys.exit(main())