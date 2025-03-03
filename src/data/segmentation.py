"""
Module for creating segmentation masks from dead elements information.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


def create_segmentation_masks(png_dir, output_dir):
    """
    Create segmentation masks based on dead elements information.

    Args:
        png_dir (str): Directory containing PNGs and dead_elements.npy files
        output_dir (str): Directory to save segmentation masks

    Returns:
        int: Number of masks created
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of dead elements files
    dead_element_files = [f for f in os.listdir(png_dir) if f.endswith('_dead_elements.npy')]

    if not dead_element_files:
        print(f"No dead elements files found in {png_dir}")
        return 0

    print(f"Creating {len(dead_element_files)} segmentation masks...")

    for dead_file in tqdm(dead_element_files, desc="Creating segmentation masks"):
        # Load dead elements data
        dead_elements_path = os.path.join(png_dir, dead_file)
        try:
            dead_elements = np.load(dead_elements_path)
        except Exception as e:
            print(f"Error loading {dead_elements_path}: {e}")
            continue

        # Get corresponding image data if available
        img_data_file = dead_file.replace('_dead_elements.npy', '_img_data.npy')
        img_data_path = os.path.join(png_dir, img_data_file)

        try:
            if os.path.exists(img_data_path):
                img_data = np.load(img_data_path)
                mask_shape = img_data.shape
            else:
                # Default shape if image data isn't available
                mask_shape = (118, 128)
        except Exception as e:
            print(f"Error loading image data {img_data_path}: {e}")
            mask_shape = (118, 128)

        # Create a mask based on dead elements
        mask = np.zeros(mask_shape, dtype=np.uint8)

        # Map dead elements to columns in the image
        # Each element in dead_elements corresponds to a column in imgData
        for i, is_dead in enumerate(dead_elements):
            if is_dead == 1 and i < mask_shape[1]:  # If element is disabled and index is valid
                # Mark the entire column as defective
                mask[:, i] = 255

        # Save the mask as PNG
        base_filename = dead_file.replace('_dead_elements.npy', '')
        mask_path = os.path.join(output_dir, f"{base_filename}_mask.png")

        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(mask, cmap='binary', aspect='auto')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Also save as NumPy array for easier processing later
        np.save(os.path.join(output_dir, f"{base_filename}_mask.npy"), mask)

    print(f"Successfully created {len(dead_element_files)} segmentation masks.")
    return len(dead_element_files)


def detect_defect_pattern(dead_elements):
    """
    Detects the pattern of defects in the dead elements array.

    Args:
        dead_elements (numpy.ndarray): Array containing dead element information

    Returns:
        str: Detected condition based on pattern
    """
    num_dead = np.sum(dead_elements)

    if num_dead == 0:
        return "all_elements_enabled"
    elif num_dead == 1:
        return "one_element_off"
    elif num_dead == 2:
        # Check if the two elements are contiguous
        for i in range(len(dead_elements) - 1):
            if dead_elements[i] == 1 and dead_elements[i + 1] == 1:
                return "two_contiguous_off"
        return "two_non_contiguous_off"
    elif num_dead == 5:
        # Check if five elements are contiguous
        for i in range(len(dead_elements) - 4):
            if np.all(dead_elements[i:i + 5] == 1):
                return "five_contiguous_off"
        return "five_random_off"
    else:
        return f"unknown_pattern_{num_dead}_elements_off"


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
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

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


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config

    create_segmentation_masks(config.PROCESSED_DATA_DIR, config.MASK_DIR)