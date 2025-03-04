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
from tqdm import tqdm
import threading
import time
import config

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

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


def check_directory_not_empty(directory, description=None):
    """
    Check if a directory exists and is not empty.

    Args:
        directory (str): Path to directory
        description (str): Description for logging

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

    if not os.listdir(directory):
        logging.error(f"Directory is empty: {directory}")
        return False

    file_count = len(os.listdir(directory))
    logging.info(f"Directory contains {file_count} files/directories.")
    return True


def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Run the entire ultrasound segmentation pipeline")
    parser.add_argument("--start_step", type=int, default=1,
                        help="Step to start from (1-6)")
    parser.add_argument("--end_step", type=int, default=6,
                        help="Step to end at (1-6)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for training")
    parser.add_argument("--model_type", choices=["resnet50", "unet"], default="resnet50",
                        help="Type of model architecture to use")
    parser.add_argument("--num_augmentations", type=int, default=3,
                        help="Number of augmentations per image")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization steps to speed up processing")
    parser.add_argument("--skip_checks", action="store_true",
                        help="Skip directory checks between steps")
    parser.add_argument("--force", action="store_true",
                        help="Force continuation even if a step fails")

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("STARTING ULTRASOUND SEGMENTATION PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Start step: {args.start_step}, End step: {args.end_step}")
    logging.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Skip visualization: {args.skip_visualization}")
    logging.info(f"Force continuation: {args.force}")
    logging.info(f"Skip checks: {args.skip_checks}")

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

        if not args.skip_checks:
            if not check_directory_not_empty(config.PROCESSED_DATA_DIR, "processed data"):
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

        if not success and not args.force:
            logging.error("Mask creation failed. Exiting.")
            return 1

        if not args.skip_checks:
            if not check_directory_not_empty(config.MASK_DIR, "mask"):
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

        if not args.skip_checks:
            if not check_directory_not_empty(os.path.join(config.UNIFIED_DATA_DIR, "images"), "unified images"):
                if not args.force:
                    logging.error("Unified data check failed. Exiting.")
                    return 1
            if not check_directory_not_empty(os.path.join(config.UNIFIED_DATA_DIR, "masks"), "unified masks"):
                if not args.force:
                    logging.error("Unified masks check failed. Exiting.")
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

        if not args.skip_checks:
            if not check_directory_not_empty(os.path.join(config.AUGMENTED_UNIFIED_DATA_DIR, "images"),
                                             "augmented images"):
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

        if not args.skip_checks:
            if not check_directory_not_empty(config.CHECKPOINT_DIR, "model checkpoints"):
                if not args.force:
                    logging.error("Model checkpoint check failed. Exiting.")
                    return 1

    # Step 6: Evaluate Unified
    if args.start_step <= 6 <= args.end_step:
        # Find the latest model file
        model_files = [f for f in os.listdir(config.CHECKPOINT_DIR)
                       if f.endswith('.h5') and not f.endswith('_final.h5') and 'unified' in f]

        if not model_files:
            logging.error("No model files found for evaluation.")
            if not args.force:
                return 1
        else:
            # Get the most recent model
            latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(config.CHECKPOINT_DIR, f)))
            model_path = os.path.join(config.CHECKPOINT_DIR, latest_model)

            logging.info(f"Using latest model for evaluation: {model_path}")

            step_args = ["--model_path", model_path,
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