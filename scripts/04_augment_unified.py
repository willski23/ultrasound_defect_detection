#!/usr/bin/env python
"""
Script to augment unified data to prevent overfitting.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
from src.data.augment_unified import augment_unified_data, visualize_unified_augmentations


def main():
    """Main function to augment unified data."""
    parser = argparse.ArgumentParser(description="Augment unified data to prevent overfitting")
    parser.add_argument("--input_dir", default=config.UNIFIED_DATA_DIR,
                        help="Directory with unified data")
    parser.add_argument("--output_dir", default=os.path.join(config.DATA_DIR, "augmented_unified"),
                        help="Directory to save augmented unified data")
    parser.add_argument("--num_augmentations", type=int, default=config.NUM_AUGMENTATIONS_PER_IMAGE,
                        help="Number of augmentations per image")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize examples of augmentations")

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        print("Please run the organize_unified.py script first.")
        return 1

    # Check if metadata file exists
    metadata_path = os.path.join(args.input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} not found.")
        print("Please run the organize_unified.py script first.")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Augment unified data
    print(f"Augmenting unified data...")
    stats, metadata = augment_unified_data(
        args.input_dir,
        args.output_dir,
        args.num_augmentations
    )

    # Visualize augmentations if requested
    if args.visualize:
        print("\nGenerating augmentation examples...")

        # Create visualization directory
        vis_dir = os.path.join(project_root, "notebooks", "unified_augmentations")
        os.makedirs(vis_dir, exist_ok=True)

        # Create visualizations
        vis_paths = visualize_unified_augmentations(
            args.output_dir,
            vis_dir
        )

        if vis_paths:
            print(f"Augmentation examples saved to {vis_dir}")

    print(f"\nUnified data augmentation complete.")
    print(f"Original images: {stats['original']}")
    print(f"Augmented images: {stats['augmented']}")
    print(f"Total images: {stats['original'] + stats['augmented']}")
    print(f"Augmented data is saved to {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())