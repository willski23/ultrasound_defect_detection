#!/usr/bin/env python
"""
Script to create segmentation masks from the processed data.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
from src.data.segmentation import create_segmentation_masks, overlay_mask_on_image
from src.utils.visualization import visualize_dead_elements
import numpy as np
import random


def main():
    """Main function to create segmentation masks."""
    parser = argparse.ArgumentParser(description="Create segmentation masks from processed data")
    parser.add_argument("--input_dir", default=config.PROCESSED_DATA_DIR,
                        help="Directory containing processed PNG images and metadata")
    parser.add_argument("--output_dir", default=config.MASK_DIR,
                        help="Directory to save segmentation masks")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize some examples of the created masks")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples to visualize")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return 1

    # Check for processed data
    dead_elements_files = [f for f in os.listdir(args.input_dir) if f.endswith('_dead_elements.npy')]
    if not dead_elements_files:
        print(f"Error: No processed data found in {args.input_dir}")
        return 1

    # Create segmentation masks
    num_created = create_segmentation_masks(args.input_dir, args.output_dir)

    if num_created == 0:
        print("No masks were created.")
        return 1

    print(f"Successfully created {num_created} segmentation masks.")
    print(f"Masks are saved to {args.output_dir}")

    # Visualize some examples if requested
    if args.visualize and num_created > 0:
        print(f"\nVisualizing {min(args.num_examples, num_created)} examples...")

        # Create visualization directory
        vis_dir = os.path.join(project_root, "notebooks", "mask_examples")
        os.makedirs(vis_dir, exist_ok=True)

        # Randomly select examples
        examples = random.sample(dead_elements_files, min(args.num_examples, len(dead_elements_files)))

        for example_file in examples:
            base_name = example_file.replace('_dead_elements.npy', '')

            # Load dead elements and image data
            dead_elements_path = os.path.join(args.input_dir, example_file)
            img_data_path = os.path.join(args.input_dir, f"{base_name}_img_data.npy")

            if os.path.exists(dead_elements_path) and os.path.exists(img_data_path):
                dead_elements = np.load(dead_elements_path)
                img_data = np.load(img_data_path)

                # Visualize dead elements on original image
                vis_path = os.path.join(vis_dir, f"{base_name}_visualization.png")
                visualize_dead_elements(img_data, dead_elements, vis_path)

                # Create an overlay of mask on image
                img_path = os.path.join(args.input_dir, f"{base_name}.png")
                mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")

                if os.path.exists(img_path) and os.path.exists(mask_path):
                    overlay_path = os.path.join(vis_dir, f"{base_name}_overlay.png")
                    overlay_mask_on_image(img_path, mask_path, overlay_path)

        print(f"Visualizations saved to {vis_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())