"""
Module for evaluating the model with element-based metrics.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
import json

def evaluate_model_unified(model, test_ds, dataset_info, output_dir=None):
    """
    Evaluate the model with element-based metrics.

    Args:
        model (tensorflow.keras.Model): Trained model
        test_ds (tf.data.Dataset): Test dataset
        dataset_info (dict): Dataset information including metadata
        output_dir (str): Directory to save evaluation results

    Returns:
        dict: Evaluation metrics
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Evaluating model on test dataset...")

    # Standard evaluation metrics
    results = model.evaluate(test_ds, verbose=1)

    # Create a dictionary with the metrics
    metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        metrics[metric_name] = results[i]

    # Print the metrics
    print("\nTest Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Element-based evaluation
    print("\nPerforming element-based evaluation...")
    element_metrics = evaluate_element_detection(model, dataset_info, output_dir)

    # Combine metrics
    all_metrics = {**metrics, **element_metrics}

    # Save the metrics to a file if output directory is provided
    if output_dir:
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    return all_metrics

def evaluate_element_detection(model, dataset_info, output_dir=None):
    """
    Evaluate the model's ability to detect defective elements.

    Args:
        model (tensorflow.keras.Model): Trained model
        dataset_info (dict): Dataset information including metadata
        output_dir (str): Directory to save evaluation results

    Returns:
        dict: Element-based evaluation metrics
    """
    # Get test set information
    test_images = dataset_info['test']['images']
    test_masks = dataset_info['test']['masks']
    test_metadata = dataset_info['test']['metadata']

    # Initialize arrays to store true and predicted defective elements
    all_true_elements = []
    all_pred_elements = []

    # Process each image in the test set
    for i, (img_path, mask_path, metadata) in enumerate(zip(test_images, test_masks, test_metadata)):
        # Get ground truth defective elements
        true_elements = metadata['dead_elements']
        all_true_elements.append(true_elements)

        # Load and preprocess image for prediction
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img, axis=0)

        # Make prediction
        pred_mask = model.predict(img_batch)[0, :, :, 0]

        # Threshold the prediction
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # Determine predicted defective elements by analyzing columns with defects
        # We need to map from the 224x224 prediction back to the original element indices (typically 128)
        num_elements = 128  # Typical number of elements in the ultrasound probe

        # Reshape prediction to match element count
        # Average the binary mask along rows to see which columns have defects
        column_means = np.mean(binary_mask, axis=0)

        # Resize to match the number of elements
        column_means_resized = cv2.resize(column_means.reshape(1, -1), (num_elements, 1),
                                         interpolation=cv2.INTER_LINEAR)[0]

        # Columns with mean above threshold are considered defective
        pred_elements = np.where(column_means_resized > 0.3)[0].tolist()
        all_pred_elements.append(pred_elements)

        # Visualize if output directory is provided
        if output_dir and i < 5:  # Visualize first 5 examples
            visualize_element_prediction(
                img, pred_mask, true_elements, pred_elements,
                os.path.join(output_dir, f"element_pred_{i}.png")
            )

    # Calculate element-based metrics
    element_metrics = calculate_element_metrics(all_true_elements, all_pred_elements)

    # Generate confusion matrix
    if output_dir:
        plot_element_confusion_matrix(all_true_elements, all_pred_elements, num_elements,
                                    os.path.join(output_dir, "element_confusion_matrix.png"))

    return element_metrics

def calculate_element_metrics(all_true_elements, all_pred_elements):
    """
    Calculate metrics for element-based defect detection.

    Args:
        all_true_elements (list): List of lists with true defective elements
        all_pred_elements (list): List of lists with predicted defective elements

    Returns:
        dict: Metrics for element detection
    """
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Metrics by defect type
    defect_metrics = {
        "no_defect": {"tp": 0, "fp": 0, "fn": 0, "samples": 0},
        "single_element": {"tp": 0, "fp": 0, "fn": 0, "samples": 0},
        "contiguous_elements": {"tp": 0, "fp": 0, "fn": 0, "samples": 0},
        "random_elements": {"tp": 0, "fp": 0, "fn": 0, "samples": 0}
    }

    # Process each sample
    for true_elements, pred_elements in zip(all_true_elements, all_pred_elements):
        # Determine defect type
        if len(true_elements) == 0:
            defect_type = "no_defect"
        elif len(true_elements) == 1:
            defect_type = "single_element"
        else:
            # Check if elements are contiguous
            is_contiguous = True
            for i in range(len(true_elements) - 1):
                if true_elements[i + 1] - true_elements[i] != 1:
                    is_contiguous = False
                    break

            defect_type = "contiguous_elements" if is_contiguous else "random_elements"

        # Increment sample count
        defect_metrics[defect_type]["samples"] += 1

        # Calculate metrics for this sample
        sample_tp = len(set(true_elements) & set(pred_elements))
        sample_fp = len(set(pred_elements) - set(true_elements))
        sample_fn = len(set(true_elements) - set(pred_elements))

        # Update global counters
        true_positives += sample_tp
        false_positives += sample_fp
        false_negatives += sample_fn

        # Update defect-specific counters
        defect_metrics[defect_type]["tp"] += sample_tp
        defect_metrics[defect_type]["fp"] += sample_fp
        defect_metrics[defect_type]["fn"] += sample_fn

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate metrics for each defect type
    for defect_type, metrics in defect_metrics.items():
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]

        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["f1_score"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0

    # Compile results
    element_metrics = {
        "element_precision": precision,
        "element_recall": recall,
        "element_f1_score": f1_score,
        "defect_type_metrics": defect_metrics
    }

    # Print summary
    print(f"\nElement-based metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")

    print("\nMetrics by defect type:")
    for defect_type, metrics in defect_metrics.items():
        if metrics["samples"] > 0:
            print(f"  {defect_type} ({metrics['samples']} samples):")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")

    return element_metrics

def visualize_element_prediction(image, pred_mask, true_elements, pred_elements, output_path):
    """
    Visualize the image with true and predicted defective elements.

    Args:
        image (numpy.ndarray): Input image
        pred_mask (numpy.ndarray): Predicted mask
        true_elements (list): List of true defective elements
        pred_elements (list): List of predicted defective elements
        output_path (str): Path to save the visualization
    """
    # Create the figure
    plt.figure(figsize=(15, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    # Predicted mask
    plt.subplot(2, 2, 2)
    plt.imshow(pred_mask, cmap='jet')
    plt.colorbar(label='Defect Probability')
    plt.title("Predicted Defect Probability")
    plt.axis('off')

    # Image with true defective elements
    plt.subplot(2, 2, 3)
    plt.imshow(image)
    plt.title("True Defective Elements")

    # Highlight true defective elements
    img_width = image.shape[1]
    element_width = img_width / 128  # Assuming 128 elements

    for element_idx in true_elements:
        x_start = element_idx * element_width
        x_end = (element_idx + 1) * element_width
        plt.axvspan(x_start, x_end, color='red', alpha=0.5)

    plt.axis('off')

    # Image with predicted defective elements
    plt.subplot(2, 2, 4)
    plt.imshow(image)
    plt.title("Predicted Defective Elements")

    # Highlight predicted elements
    for element_idx in pred_elements:
        x_start = element_idx * element_width
        x_end = (element_idx + 1) * element_width
        plt.axvspan(x_start, x_end, color='green', alpha=0.5)

    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_element_confusion_matrix(all_true_elements, all_pred_elements, num_elements, output_path):
    """
    Create a confusion matrix of element predictions.

    Args:
        all_true_elements (list): List of lists with true defective elements
        all_pred_elements (list): List of lists with predicted defective elements
        num_elements (int): Total number of elements
        output_path (str): Path to save the visualization
    """
    # Initialize element-wise counts
    element_tp = np.zeros(num_elements)
    element_fp = np.zeros(num_elements)
    element_fn = np.zeros(num_elements)
    element_tn = np.zeros(num_elements)

    # Count TP, FP, FN, TN for each element
    for true_elements, pred_elements in zip(all_true_elements, all_pred_elements):
        true_set = set(true_elements)
        pred_set = set(pred_elements)

        for i in range(num_elements):
            if i in true_set and i in pred_set:
                element_tp[i] += 1
            elif i not in true_set and i in pred_set:
                element_fp[i] += 1
            elif i in true_set and i not in pred_set:
                element_fn[i] += 1
            else:
                element_tn[i] += 1

    # Calculate accuracy for each element
    element_accuracy = (element_tp + element_tn) / (element_tp + element_fp + element_fn + element_tn)

    # Calculate precision, recall, and F1 for each element
    element_precision = np.zeros(num_elements)
    element_recall = np.zeros(num_elements)
    element_f1 = np.zeros(num_elements)

    for i in range(num_elements):
        if element_tp[i] + element_fp[i] > 0:
            element_precision[i] = element_tp[i] / (element_tp[i] + element_fp[i])

        if element_tp[i] + element_fn[i] > 0:
            element_recall[i] = element_tp[i] / (element_tp[i] + element_fn[i])

        if element_precision[i] + element_recall[i] > 0:
            element_f1[i] = 2 * element_precision[i] * element_recall[i] / (element_precision[i] + element_recall[i])

    # Create visualization
    plt.figure(figsize=(15, 12))

    # Plot accuracy by element
    plt.subplot(2, 2, 1)
    plt.bar(range(num_elements), element_accuracy)
    plt.title("Element-wise Accuracy")
    plt.xlabel("Element Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot precision by element
    plt.subplot(2, 2, 2)
    plt.bar(range(num_elements), element_precision)
    plt.title("Element-wise Precision")
    plt.xlabel("Element Index")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot recall by element
    plt.subplot(2, 2, 3)
    plt.bar(range(num_elements), element_recall)
    plt.title("Element-wise Recall")
    plt.xlabel("Element Index")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot F1 score by element
    plt.subplot(2, 2, 4)
    plt.bar(range(num_elements), element_f1)
    plt.title("Element-wise F1 Score")
    plt.xlabel("Element Index")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Also save a heatmap of confusion matrices
    plt.figure(figsize=(15, 8))

    # Confusion matrix for TP rate
    plt.subplot(1, 2, 1)
    tp_rate = element_tp / (element_tp + element_fn)
    tp_rate = np.nan_to_num(tp_rate)  # Replace NaN with 0
    tp_rate = tp_rate.reshape(1, -1)  # Reshape for heatmap

    sns.heatmap(tp_rate, cmap='viridis', vmin=0, vmax=1,
                cbar_kws={'label': 'True Positive Rate'})
    plt.title("Element-wise True Positive Rate (Sensitivity)")
    plt.xlabel("Element Index")
    plt.yticks([])

    # Confusion matrix for FP rate
    plt.subplot(1, 2, 2)
    fp_rate = element_fp / (element_fp + element_tn)
    fp_rate = np.nan_to_num(fp_rate)  # Replace NaN with 0
    fp_rate = fp_rate.reshape(1, -1)  # Reshape for heatmap

    sns.heatmap(fp_rate, cmap='viridis', vmin=0, vmax=1,
                cbar_kws={'label': 'False Positive Rate'})
    plt.title("Element-wise False Positive Rate (1 - Specificity)")
    plt.xlabel("Element Index")
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_heatmap.png'))
    plt.close()

def predict_on_new_image(model, image_path, output_path=None, threshold=0.5, num_elements=128):
    """
    Make a prediction on a new image and identify defective elements.

    Args:
        model (tensorflow.keras.Model): Trained model
        image_path (str): Path to input image
        output_path (str): Path to save the output
        threshold (float): Threshold for binary segmentation
        num_elements (int): Number of elements in the ultrasound probe

    Returns:
        tuple: (original_image, prediction_mask, defective_elements)
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Store original size for visualization
    original_size = img.shape[:2]

    # Resize to model input size
    input_shape = model.input_shape[1:3]  # Get expected height and width
    img_resized = cv2.resize(img, input_shape)

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    prediction = model.predict(img_batch)[0]

    # Apply threshold to get binary mask
    binary_mask = (prediction[:, :, 0] > threshold).astype(np.uint8) * 255

    # Identify defective elements
    column_means = np.mean(binary_mask, axis=0)

    # Resize to match the number of elements
    column_means_resized = cv2.resize(column_means.reshape(1, -1), (num_elements, 1),
                                     interpolation=cv2.INTER_LINEAR)[0]

    # Columns with mean above threshold are considered defective
    defective_elements = np.where(column_means_resized > (255 * 0.3))[0].tolist()

    # Visualize the results
    if output_path:
        # Create visualization
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(img_resized)
        plt.title("Input Image")
        plt.axis('off')

        # Prediction probability
        plt.subplot(2, 2, 2)
        plt.imshow(prediction[:, :, 0], cmap='jet')
        plt.colorbar(label='Defect Probability')
        plt.title("Defect Probability Map")
        plt.axis('off')

        # Binary mask
        plt.subplot(2, 2, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Binary Segmentation Mask")
        plt.axis('off')

        # Image with defective elements highlighted
        plt.subplot(2, 2, 4)
        plt.imshow(img_resized)
        plt.title(f"Defective Elements: {defective_elements}")

        # Highlight defective elements
        img_width = img_resized.shape[1]
        element_width = img_width / num_elements

        for element_idx in defective_elements:
            x_start = element_idx * element_width
            x_end = (element_idx + 1) * element_width
            plt.axvspan(x_start, x_end, color='red', alpha=0.5)

        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Prediction saved to {output_path}")
        print(f"Detected {len(defective_elements)} defective elements: {defective_elements}")

    return img_resized, prediction[:, :, 0], defective_elements

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config
    from src.model.dataset_unified import prepare_unified_dataset
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model with element-based metrics")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--data_dir", type=str, default=os.path.join(config.DATA_DIR, "unified"),
                        help="Path to the unified data directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(config.MODEL_DIR, "evaluation_unified"),
                        help="Directory to save evaluation results")

    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model_path)

    # Prepare the dataset
    _, _, test_ds, dataset_info = prepare_unified_dataset(
        args.data_dir,
        img_size=model.input_shape[1:3],
        batch_size=config.BATCH_SIZE
    )

    # Evaluate the model
    evaluate_model_unified(model, test_ds, dataset_info, args.output_dir)