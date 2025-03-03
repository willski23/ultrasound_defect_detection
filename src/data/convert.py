"""
Module for converting MAT files to PNG images.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm


def convert_mat_to_png(input_dir, output_dir):
    """
    Convert MAT files to PNG images while preserving the relationship with dead elements.

    Args:
        input_dir (str): Directory containing MAT files
        output_dir (str): Directory to save PNG images

    Returns:
        int: Number of files converted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of MAT files
    mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

    if not mat_files:
        print(f"No MAT files found in {input_dir}")
        return 0

    print(f"Converting {len(mat_files)} MAT files to PNG...")

    for mat_file in tqdm(mat_files, desc="Converting MAT to PNG"):
        # Load MAT file
        mat_path = os.path.join(input_dir, mat_file)
        try:
            mat_data = loadmat(mat_path)
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue

        # Extract relevant data
        try:
            img_data = mat_data['imgData']  # B-mode image data (118 x 128)
            dead_elements = mat_data['deadElements'].flatten()  # Status of transducers (1 = disabled)
        except KeyError as e:
            print(f"Missing expected field in {mat_path}: {e}")
            continue

        # Create filename without extension
        filename = os.path.splitext(mat_file)[0]

        # Save as PNG
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(img_data, cmap='gray', aspect='auto')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save dead elements information as a separate file (for segmentation masks later)
        np.save(os.path.join(output_dir, f"{filename}_dead_elements.npy"), dead_elements)

        # Also save raw image data for potential future use
        np.save(os.path.join(output_dir, f"{filename}_img_data.npy"), img_data)

    print(f"Successfully converted {len(mat_files)} MAT files to PNG format.")
    return len(mat_files)


def inspect_mat_file(file_path):
    """
    Inspect a single MAT file to understand its structure.

    Args:
        file_path (str): Path to MAT file

    Returns:
        dict: Dictionary of key information about the file
    """
    try:
        mat_data = loadmat(file_path)

        info = {
            "keys": list(mat_data.keys()),
            "imgData_shape": mat_data['imgData'].shape if 'imgData' in mat_data else None,
            "deadElements_shape": mat_data['deadElements'].shape if 'deadElements' in mat_data else None,
            "xAxis_shape": mat_data['xAxis'].shape if 'xAxis' in mat_data else None,
            "zAxis_shape": mat_data['zAxis'].shape if 'zAxis' in mat_data else None,
        }

        return info
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")
        return None


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config

    convert_mat_to_png(config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
