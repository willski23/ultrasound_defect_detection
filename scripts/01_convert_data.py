#!/usr/bin/env python
"""
Script to convert MAT files to PNG images.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config
from src.data.convert import convert_mat_to_png, inspect_mat_file


def main():
    """Main function to convert MAT files to PNG."""
    parser = argparse.ArgumentParser(description="Convert MAT files to PNG images")
    parser.add_argument("--input_dir", default=config.RAW_DATA_DIR,
                        help="Directory containing MAT files")
    parser.add_argument("--output_dir", default=config.PROCESSED_DATA_DIR,
                        help="Directory to save PNG images")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect a MAT file before conversion")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return 1

    # Check if there are MAT files in the input directory
    mat_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mat')]
    if not mat_files:
        print(f"Error: No MAT files found in {args.input_dir}")
        return 1

    # Inspect a file if requested
    if args.inspect:
        sample_file = os.path.join(args.input_dir, mat_files[0])
        print(f"Inspecting sample file: {sample_file}")
        info = inspect_mat_file(sample_file)
        if info:
            print("\nMAT file structure:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print("\nPress Enter to continue with conversion, or Ctrl+C to abort.")
            input()

    # Convert MAT files to PNG
    num_converted = convert_mat_to_png(args.input_dir, args.output_dir)

    if num_converted > 0:
        print(f"Successfully converted {num_converted} MAT files to PNG.")
        print(f"PNG images and metadata are saved to {args.output_dir}")
        return 0
    else:
        print("No files were converted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())