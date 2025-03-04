#!/usr/bin/env python
"""
Master script to automate the entire ultrasound segmentation pipeline.
This script sequentially runs each step of the pipeline, ensuring proper execution
and handling errors as they occur.
"""
import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import time
import threading
import shutil  # Add shutil import for file operations
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import config

# Setup logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)


def run_step_with_progress(script_name, args=None, description=None):
    """
    Run a step in the pipeline with a progress bar.

    Args:
        script_name (str): Name of the script to run
        args (list): Additional arguments to pass to the script
        description (str): Description of the step for logging

    Returns:
        bool: True if successful, False otherwise
    """
    if description:
        logging.info(f"STEP: {description}")
    else:
        logging.info(f"Running {script_name}")

    script_path = os.path.join(project_root, "scripts", script_name)

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    logging.info(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    # Create a progress bar
    pbar = tqdm(desc=f"Running {os.path.basename(script_name)}",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}')

    # Function to update progress bar
    def update_progress():
        while process.poll() is None:  # While process is running
            pbar.update(1)
            time.sleep(0.5)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Start progress thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()

        # Stream output to log file, not console
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                logging.info(f"  {line}")

        process.wait()
        pbar.close()  # Close progress bar

        # Print a summary of output
        if output_lines:
            print(f"Last output: {output_lines[-1]}")

        # Check for errors
        for line in process.stderr:
            line = line.strip()
            if line:
                logging.error(f"  {line}")
                print(f"ERROR: {line}")

        if process.returncode != 0:
            logging.error(f"Step failed with return code {process.returncode}")
            print(f"Step failed with return code {process.returncode}")
            return False

        elapsed_time = time.time() - start_time
        message = f"Step completed successfully in {elapsed_time:.2f} seconds."
        logging.info(message)
        print(message)
        return True

    except Exception as e:
        pbar.close()  # Make sure to close the progress bar
        logging.error(f"Error running {script_name}: {e}")
        print(f"Error running {script_name}: {e}")
        return False


def check_directory_not_empty(directory, description=None, file_pattern="*"):
    """
    Check if a directory exists and is not empty.

    Args:
        directory (str): Path to directory
        description (str): Description for logging
        file_pattern (str): Pattern to match files

    Returns:
        bool: True if directory exists and is not empty, False otherwise
    """
    if description:
        logging.info(f"Checking {description} directory: {directory}")
    else:
        logging.info(f"Checking directory: {directory}")

    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return False

    files = []
    for pattern in file_pattern.split(','):
        files.extend(list(Path(directory).glob(pattern)))

    if not files:
        logging.error(f"No files matching pattern '{file_pattern}' found in {directory}")
        return False

    file_count = len(files)
    logging.info(f"Directory contains {file_count} files/directories matching pattern '{file_pattern}'")

    # Log some sample filenames
    sample_files = files[:5]
    logging.info(f"Sample files: {', '.join([os.path.basename(str(f)) for f in sample_files])}")

    return True


def manually_create_sample_mask(input_dir, output_dir, num_samples=5):
    """
    Manually create sample segmentation masks based on dead elements.
    This function is used as a fallback if the mask creation script fails.

    Args:
        input_dir (str): Directory with processed PNG images and dead elements
        output_dir (str): Directory to save masks
        num_samples (int): Number of samples to process

    Returns:
        bool: True if successful, False otherwise
    """
    import numpy as np
    import matplotlib.pyplot as plt

    logging.info(f"Manually creating {num_samples} sample masks as fallback...")

    os.makedirs(output_dir, exist_ok=True)

    dead_element_files = list(Path(input_dir).glob("*_dead_elements.npy"))

    if not dead_element_files:
        logging.error(f"No dead elements files found in {input_dir}")
        return False

    # Process a few samples
    successful = 0
    for i, dead_file in enumerate(dead_element_files[:num_samples]):
        try:
            base_name = os.path.basename(str(dead_file)).replace('_dead_elements.npy', '')

            # Load dead elements data
            dead_elements = np.load(dead_file)

            # Get image data if available
            img_data_file = os.path.join(input_dir, f"{base_name}_img_data.npy")

            if os.path.exists(img_data_file):
                img_data = np.load(img_data_file)
                mask_shape = img_data.shape
            else:
                # Default shape if image data isn't available
                mask_shape = (118, 128)

            # Create mask
            mask = np.zeros(mask_shape, dtype=np.uint8)

            # Mark dead elements
            for i, is_dead in enumerate(dead_elements):
                if is_dead == 1 and i < mask_shape[1]:
                    mask[:, i] = 255

            # Save mask
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            plt.figure(figsize=(8, 8), dpi=100)
            plt.imshow(mask, cmap='binary', aspect='auto')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            logging.info(f"Created sample mask: {mask_path}")
            successful += 1

        except Exception as e:
            logging.error(f"Error creating sample mask: {e}")

    logging.info(f"Successfully created {successful} sample masks")
    return successful > 0


def attempt_fix_directory_structure():
    """
    Attempt to fix common directory structure issues.

    Returns:
        bool: True if fixed, False otherwise
    """
    logging.info("Attempting to fix directory structure...")

    try:
        # Ensure all directories exist
        for directory in [
            config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.MASK_DIR,
            config.ORGANIZED_DATA_DIR, config.AUGMENTED_DATA_DIR,
            config.UNIFIED_DATA_DIR, config.AUGMENTED_UNIFIED_DATA_DIR,
            config.MODEL_DIR, config.CHECKPOINT_DIR
        ]:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")

        # Check for mat files and copy to raw dir if needed
        mat_files_in_project = []
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file.endswith('.mat'):
                    mat_files_in_project.append(os.path.join(root, file))

        mat_files_in_raw = list(Path(config.RAW_DATA_DIR).glob("*.mat"))

        if not mat_files_in_raw and mat_files_in_project:
            logging.info(f"Found {len(mat_files_in_project)} MAT files in project directory but none in RAW_DATA_DIR")
            logging.info(f"Copying MAT files to {config.RAW_DATA_DIR}...")

            for file in mat_files_in_project[:10]:  # Copy up to 10 files as samples
                dest = os.path.join(config.RAW_DATA_DIR, os.path.basename(file))
                shutil.copy(file, dest)
                logging.info(f"Copied {file} to {dest}")

        return True

    except Exception as e:
        logging.error(f"Error fixing directory structure: {e}")
        return False


def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Run the entire ultrasound segmentation pipeline")
    parser.add_argument("--start_step", type=int, default=1,
                        help="Step to start from (1-6)")
    parser.add_argument("--end_step", type=int, default=6,
                        help="Step to end at (1-6)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (reduced to avoid memory issues)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for training")
    parser.add_argument("--model_type", choices=["resnet50", "unet"], default="resnet50",
                        help="Type of model architecture to use")
    parser.add_argument("--num_augmentations", type=int, default=3,
                        help="Number of augmentations per image")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization steps to speed up processing")
    parser.add_argument("--force", action="store_true",
                        help="Force continuation even if a step fails")
    parser.add_argument("--fix_directories", action="store_true",
                        help="Attempt to fix directory structure issues")

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("STARTING ULTRASOUND SEGMENTATION PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Start step: {args.start_step}, End step: {args.end_step}")
    logging.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Skip visualization: {args.skip_visualization}")
    logging.info(f"Force continuation: {args.force}")

    # Try to fix directory structure if requested
    if args.fix_directories:
        if not attempt_fix_directory_structure():
            logging.error("Failed to fix directory structure.")
            if not args.force:
                return 1

    # Step 1: Convert Data
    if args.start_step <= 1 <= args.end_step:
        step_args = ["--input_dir", config.RAW_DATA_DIR,
                     "--output_dir", config.PROCESSED_DATA_DIR]
        if not args.skip_visualization:
            step_args.append("--inspect")

        success = run_step_with_progress(
            "01_convert_data.py",
            args=step_args,
            description="Converting MAT files to PNG images"
        )

        if not success and not args.force:
            logging.error("Data conversion failed. Exiting.")
            return 1

        # Check processed data
        if not check_directory_not_empty(
                config.PROCESSED_DATA_DIR,
                "processed data",
                "*.png"
        ):
            if not args.force:
                logging.error("Processed data check failed. Exiting.")
                return 1

    # Step 2: Create Masks
    if args.start_step <= 2 <= args.end_step:
        step_args = ["--input_dir", config.PROCESSED_DATA_DIR,
                     "--output_dir", config.MASK_DIR]
        if not args.skip_visualization:
            step_args.append("--visualize")

        success = run_step_with_progress(
            "02_create_masks.py",
            args=step_args,
            description="Creating segmentation masks"
        )

        if not success:
            logging.warning("Mask creation script failed, attempting manual fallback...")
            fallback_success = manually_create_sample_mask(
                config.PROCESSED_DATA_DIR,
                config.MASK_DIR
            )

            if not fallback_success and not args.force:
                logging.error("Mask creation fallback failed. Exiting.")
                return 1

        # Check mask directory
        if not check_directory_not_empty(
                config.MASK_DIR,
                "mask",
                "*_mask.png"
        ):
            if not args.force:
                logging.error("Mask data check failed. Exiting.")
                return 1

    # Step 3: Organize Unified
    if args.start_step <= 3 <= args.end_step:
        step_args = ["--img_dir", config.PROCESSED_DATA_DIR,
                     "--mask_dir", config.MASK_DIR,
                     "--output_dir", config.UNIFIED_DATA_DIR,
                     "--verify"]

        success = run_step_with_progress(
            "03_organize_unified.py",
            args=step_args,
            description="Organizing data with unified approach"
        )

        if not success and not args.force:
            logging.error("Data organization failed. Exiting.")
            return 1

        # Check unified data
        if not check_directory_not_empty(
                os.path.join(config.UNIFIED_DATA_DIR, "images"),
                "unified images",
                "*.png"
        ):
            if not args.force:
                logging.error("Unified data check failed. Exiting.")
                return 1

    # Step 4: Augment Unified
    if args.start_step <= 4 <= args.end_step:
        step_args = ["--input_dir", config.UNIFIED_DATA_DIR,
                     "--output_dir", config.AUGMENTED_UNIFIED_DATA_DIR,
                     "--num_augmentations", str(args.num_augmentations)]
        if args.skip_visualization:
            step_args.append("--skip_visualization")

        success = run_step_with_progress(
            "04_augment_unified.py",
            args=step_args,
            description="Augmenting unified data"
        )

        if not success and not args.force:
            logging.error("Data augmentation failed. Exiting.")
            return 1

        # Check augmented data
        if not check_directory_not_empty(
                os.path.join(config.AUGMENTED_UNIFIED_DATA_DIR, "images"),
                "augmented images",
                "*.png"
        ):
            if not args.force:
                logging.error("Augmented data check failed. Exiting.")
                return 1

    # Step 5: Train Unified
    if args.start_step <= 5 <= args.end_step:
        step_args = ["--data_dir", config.AUGMENTED_UNIFIED_DATA_DIR,
                     "--checkpoint_dir", config.CHECKPOINT_DIR,
                     "--model_type", args.model_type,
                     "--batch_size", str(args.batch_size),
                     "--epochs", str(args.epochs),
                     "--freeze_encoder"]
        if not args.skip_visualization:
            step_args.append("--visualize_samples")

        success = run_step_with_progress(
            "05_train_unified.py",
            args=step_args,
            description="Training segmentation model"
        )

        if not success and not args.force:
            logging.error("Model training failed. Exiting.")
            return 1

        # Check model checkpoints
        if not check_directory_not_empty(
                config.CHECKPOINT_DIR,
                "model checkpoints",
                "*.h5"
        ):
            if not args.force:
                logging.error("Model checkpoint check failed. Exiting.")
                return 1

    # Step 6: Evaluate Unified
    if args.start_step <= 6 <= args.end_step:
        # Find the latest model file
        model_files = list(Path(config.CHECKPOINT_DIR).glob("*unified*.h5"))
        if not model_files:
            model_files = list(Path(config.CHECKPOINT_DIR).glob("*.h5"))

        if not model_files:
            logging.error("No model files found for evaluation.")
            if not args.force:
                return 1
        else:
            # Get the most recent model
            latest_model = max(model_files, key=os.path.getmtime)

            logging.info(f"Using latest model for evaluation: {latest_model}")

            step_args = ["--model_path", str(latest_model),
                         "--data_dir", config.UNIFIED_DATA_DIR]

            success = run_step_with_progress(
                "06_evaluate_unified.py",
                args=step_args,
                description="Evaluating trained model"
            )

            if not success and not args.force:
                logging.error("Model evaluation failed. Exiting.")
                return 1

    logging.info("=" * 80)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    return 0


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed_time = time.time() - start_time
    logging.info(f"Total pipeline execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    sys.exit(exit_code)