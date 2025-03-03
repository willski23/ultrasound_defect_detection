"""
Module for visualization utilities.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2
import tensorflow as tf


def plot_sample_images(images, masks=None, predictions=None, num_samples=5, output_path=None):
    """
    Plot sample images with optional masks and predictions.

    Args:
        images (list or numpy.ndarray): List of images
        masks (list or numpy.ndarray, optional): List of masks
        predictions (list or numpy.ndarray, optional): List of predictions
        num_samples (int): Number of samples to plot
        output_path (str, optional): Path to save the plot

    Returns:
        None
    """
    # Determine the number of samples to show
    n = min(num_samples, len(images))

    # Determine the number of columns (1 for just images, 2 for images+masks, 3 for images+masks+predictions)
    cols = 1 + (masks is not None) + (predictions is not None)

    # Create the figure
    plt.figure(figsize=(4 * cols, 4 * n))

    for i in range(n):
        # Plot original image
        plt.subplot(n, cols, i * cols + 1)
        plt.imshow(images[i])
        plt.title(f"Image {i + 1}")
        plt.axis('off')

        # Plot mask if provided
        if masks is not None:
            plt.subplot(n, cols, i * cols + 2)
            plt.imshow(masks[i], cmap='gray')
            plt.title(f"Mask {i + 1}")
            plt.axis('off')

        # Plot prediction if provided
        if predictions is not None:
            plt.subplot(n, cols, i * cols + 3)
            plt.imshow(predictions[i], cmap='jet')
            plt.title(f"Prediction {i + 1}")
            plt.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def create_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Create an overlay of a mask on an image.

    Args:
        image (numpy.ndarray): Original image (RGB)
        mask (numpy.ndarray): Binary mask (single channel)
        color (tuple): Color for the overlay (R, G, B)
        alpha (float): Transparency of the overlay (0-1)

    Returns:
        numpy.ndarray: Image with overlay
    """
    # Ensure the image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Ensure the mask is binary and single channel
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    # Normalize mask to 0-1 if needed
    if mask.max() > 1:
        mask = mask / 255.0

    # Create a colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = color[0]  # R
    colored_mask[:, :, 1] = color[1]  # G
    colored_mask[:, :, 2] = color[2]  # B

    # Apply the mask with alpha blending
    overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = (1 - alpha) * image[:, :, c] + alpha * colored_mask[:, :, c] * mask

    return overlay.astype(np.uint8)


def plot_side_by_side(image, true_mask, pred_mask, output_path=None):
    """
    Plot an image with its true mask and predicted mask side by side.

    Args:
        image (numpy.ndarray): Original image
        true_mask (numpy.ndarray): True mask
        pred_mask (numpy.ndarray): Predicted mask
        output_path (str, optional): Path to save the plot

    Returns:
        None
    """
    # Convert masks to binary if they're not already
    if true_mask.max() > 1:
        true_mask = true_mask / 255.0

    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0

    # Create overlays
    true_overlay = create_overlay(image, true_mask, color=(255, 0, 0), alpha=0.5)  # Red for true mask
    pred_overlay = create_overlay(image, pred_mask, color=(0, 255, 0), alpha=0.5)  # Green for predicted mask

    # Create a combined overlay showing both masks
    combined_mask = np.zeros_like(image)
    combined_mask[:, :, 0] = np.maximum(true_mask * 255, 0)  # Red channel for true mask
    combined_mask[:, :, 1] = np.maximum(pred_mask * 255, 0)  # Green channel for predicted mask
    combined_mask[:, :, 2] = 0  # Blue channel is empty

    # Convert combined mask to a more visually understandable representation
    # Red: True positive (both true and pred are 1)
    # Green: False positive (pred is 1, true is 0)
    # Blue: False negative (pred is 0, true is 1)
    comparison = np.zeros_like(image)

    # True positive (TP): where both masks are active (yellow)
    tp_mask = (true_mask > 0.5) & (pred_mask > 0.5)
    comparison[tp_mask, 0] = 255  # R
    comparison[tp_mask, 1] = 255  # G

    # False positive (FP): where pred is active but true is not (green)
    fp_mask = (true_mask <= 0.5) & (pred_mask > 0.5)
    comparison[fp_mask, 1] = 255  # G

    # False negative (FN): where true is active but pred is not (red)
    fn_mask = (true_mask > 0.5) & (pred_mask <= 0.5)
    comparison[fn_mask, 0] = 255  # R

    # Create the figure
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(true_overlay)
    plt.title("True Mask Overlay")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(pred_overlay)
    plt.title("Predicted Mask Overlay")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(comparison)
    plt.title("Comparison\nYellow: TP, Green: FP, Red: FN")
    plt.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Comparison plot saved to {output_path}")
    else:
        plt.show()


def visualize_model_layers(model, image, layer_names=None, output_path=None):
    """
    Visualize activations of selected layers in the model.

    Args:
        model (tensorflow.keras.Model): Model to visualize
        image (numpy.ndarray): Input image
        layer_names (list, optional): List of layer names to visualize
        output_path (str, optional): Path to save the visualization

    Returns:
        None
    """
    # If no layer names provided, use all Conv2D layers
    if layer_names is None:
        layer_names = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_names.append(layer.name)

    # Prepare input image
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Create a model that outputs all desired layer activations
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)

    # Get activations
    activations = activation_model.predict(image)

    # Make sure activations is a list even if there's only one layer
    if not isinstance(activations, list):
        activations = [activations]

    # Visualize activations for each layer
    for i, layer_name in enumerate(layer_names):
        layer_activation = activations[i]
        n_features = layer_activation.shape[-1]

        # Determine grid size for visualizing features
        size = int(np.ceil(np.sqrt(n_features)))

        # Create the figure
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Layer: {layer_name}", fontsize=16)

        # Plot up to 64 features to keep the figure manageable
        features_to_plot = min(n_features, 64)

        for j in range(features_to_plot):
            plt.subplot(size, size, j + 1)

            # Use a different normalization for each feature map
            feature = layer_activation[0, :, :, j]
            vmin, vmax = feature.min(), feature.max()

            if vmax > vmin:
                plt.imshow(feature, cmap='viridis', vmin=vmin, vmax=vmax)
            else:
                plt.imshow(feature, cmap='viridis')

            plt.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        if output_path:
            # If we're saving multiple layers, append layer name to filename
            if len(layer_names) > 1:
                base, ext = os.path.splitext(output_path)
                layer_output_path = f"{base}_{layer_name}{ext}"
            else:
                layer_output_path = output_path

            plt.savefig(layer_output_path)
            plt.close()
            print(f"Layer visualization saved to {layer_output_path}")
        else:
            plt.show()


def visualize_dead_elements(img_data, dead_elements, output_path=None):
    """
    Visualize the original ultrasound image with dead elements highlighted.

    Args:
        img_data (numpy.ndarray): Original ultrasound image data (2D array)
        dead_elements (numpy.ndarray): 1D array indicating dead elements (1 = dead)
        output_path (str, optional): Path to save the visualization

    Returns:
        None
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

    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Dead elements visualization saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage when run as a script
    from pathlib import Path
    import sys

    # Add the project root to the path
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config
    from scipy.io import loadmat

    # Load a sample MAT file to visualize dead elements
    sample_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.mat')]

    if sample_files:
        sample_file = os.path.join(config.RAW_DATA_DIR, sample_files[0])
        mat_data = loadmat(sample_file)

        img_data = mat_data['imgData']
        dead_elements = mat_data['deadElements'].flatten()

        output_dir = os.path.join(project_root, "notebooks")
        os.makedirs(output_dir, exist_ok=True)

        visualize_dead_elements(
            img_data,
            dead_elements,
            output_path=os.path.join(output_dir, "dead_elements_example.png")
        )

        print(f"Created example visualization in {output_dir}")
    else:
        print("No sample MAT files found in the raw data directory.")