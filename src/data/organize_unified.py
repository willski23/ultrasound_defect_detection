"""
Module for organizing data with a unified approach based on element defects.
"""
import os
import shutil
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

def organize_unified(input_img_dir, input_mask_dir, output_dir):
    """
    Organize all data in a unified structure with metadata about defective elements.
    (all images in same directory classfied by number of defective elements)

    Args:
        input_img_dir (str): Directory containing images
        input_mask_dir (str): Directory containing masks
        output_dir (str): Directory for organized data

    Returns:
        dict: Statistics about the organized data
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Get list of images
    images = [f for f in os.listdir(input_img_dir) if f.endswith('.png')]

    # Metadata dictionary to store information about each sample
    metadata = {}

    # Counter for statistics
    stats = {
        "total_organized": 0,
        "with_defects": 0,
        "without_defects": 0,
        "defect_patterns": {}
    }

    print(f"Organizing {len(images)} images with unified approach...")

    for img_file in tqdm(images, desc="Organizing files"):
        base_name = Path(img_file).stem

        # Find corresponding dead elements file
        dead_elements_file = f"{base_name}_dead_elements.npy"
        dead_elements_path = os.path.join(input_img_dir, dead_elements_file)

        if not os.path.exists(dead_elements_path):
            print(f"Warning: No dead elements found for {img_file}")
            continue

        # Load dead elements
        try:
            dead_elements = np.load(dead_elements_path)

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

            # Update pattern statistics
            if pattern not in stats["defect_patterns"]:
                stats["defect_patterns"][pattern] = 0
            stats["defect_patterns"][pattern] += 1

            # Update defect counts
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

        except Exception as e:
            print(f"Error analyzing {dead_elements_path}: {e}")
            continue

        # Copy image and mask to output directory
        try:
            # Copy image
            shutil.copy(
                os.path.join(input_img_dir, img_file),
                os.path.join(output_dir, "images", img_file)
            )

            # Copy mask
            mask_file = f"{base_name}_mask.png"
            mask_path = os.path.join(input_mask_dir, mask_file)

            if os.path.exists(mask_path):
                shutil.copy(
                    mask_path,
                    os.path.join(output_dir, "masks", mask_file)
                )

            stats["total_organized"] += 1
        except Exception as e:
            print(f"Error copying files for {img_file}: {e}")

    # Save metadata to JSON file
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")

    # Print statistics
    print("\nOrganization complete. Statistics:")
    print(f"  Total organized: {stats['total_organized']} files")
    print(f"  With defects: {stats['with_defects']} files")
    print(f"  Without defects: {stats['without_defects']} files")
    print("  Defect patterns:")
    for pattern, count in stats["defect_patterns"].items():
        print(f"    {pattern}: {count} files")

    return stats, metadata

def verify_unified_organization(output_dir):
    """
    Verify that the unified data organization was successful.

    Args:
        output_dir (str): Base directory for organized data

    Returns:
        bool: True if verification passed, False otherwise
    """
    all_passed = True

    print("Verifying unified data organization...")

    # Check if required directories exist
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    metadata_path = os.path.join(output_dir, "metadata.json")

    if not os.path.exists(img_dir):
        print(f"Error: Images directory {img_dir} does not exist.")
        return False

    if not os.path.exists(mask_dir):
        print(f"Error: Masks directory {mask_dir} does not exist.")
        return False

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} does not exist.")
        return False

    # Check image and mask counts
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    print(f"Found {len(img_files)} images and {len(mask_files)} masks.")

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"Metadata contains information for {len(metadata)} samples.")

        # Verify metadata entries match image files
        for img_file in img_files:
            base_name = Path(img_file).stem
            if base_name not in metadata:
                print(f"Warning: No metadata entry for {img_file}")
                all_passed = False

        # Verify each image has a corresponding mask
        for img_file in img_files:
            base_name = Path(img_file).stem
            mask_file = f"{base_name}_mask.png"

            if mask_file not in mask_files:
                print(f"Warning: Missing mask for {img_file}")
                all_passed = False

    except Exception as e:
        print(f"Error verifying metadata: {e}")
        all_passed = False

    if all_passed:
        print("Verification passed: Unified data organization is valid.")
    else:
        print("Verification failed: Issues were detected in the data organization.")

    return all_passed

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config

    # Define a new unified data directory
    unified_data_dir = os.path.join(config.DATA_DIR, "unified")

    # Organize data with unified approach
    stats, metadata = organize_unified(
        config.PROCESSED_DATA_DIR,
        config.MASK_DIR,
        unified_data_dir
    )

    # Verify the organization
    verify_unified_organization(unified_data_dir)