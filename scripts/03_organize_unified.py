#!/usr/bin/env python
"""
Script to organize data with a unified approach based on element defects.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
from src.data.organize_unified import organize_unified, verify_unified_organization


def main():
    """Main function to organize data with unified approach."""
    parser = argparse.ArgumentParser(description="Organize data with unified approach based on element defects")
    parser.add_argument("--img_dir", default=config.PROCESSED_DATA_DIR,
                        help="Directory containing processed PNG images")
    parser.add_argument("--mask_dir", default=config.MASK_DIR,
                        help="Directory containing segmentation masks")
    parser.add_argument("--output_dir", default=os.path.join(config.DATA_DIR, "unified"),
                        help="Directory for unified organized data")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify the organization after completion")

    args = parser.parse_args()

    # Check if input directories exist
    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory {args.img_dir} does not exist.")
        return 1

    if not os.path.exists(args.mask_dir):
        print(f"Error: Mask directory {args.mask_dir} does not exist.")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Organize data with unified approach
    print(f"Organizing data with unified approach...")

    stats, metadata = organize_unified(
        args.img_dir,
        args.mask_dir,
        args.output_dir
    )

    # Check if any data was organized
    if stats["total_organized"] == 0:
        print("No data was organized. Please check your input directories.")
        return 1

    # Verify organization if requested
    if args.verify:
        print("\nVerifying data organization...")
        passed = verify_unified_organization(args.output_dir)
        if not passed:
            print("Warning: Organization verification found issues.")
            return 1

    print(f"\nOrganized data is saved to {args.output_dir}")
    print(f"Metadata is saved to {os.path.join(args.output_dir, 'metadata.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())