"""
Module for training the segmentation model.
"""
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard, CSVLogger


def train_model(
        model,
        train_ds,
        val_ds,
        epochs=20,
        checkpoint_dir=None,
        model_name="ultrasound_segmentation"
):
    """
    Train a segmentation model.

    Args:
        model: TensorFlow model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        model_name: Name of the model for saving

    Returns:
        model: Trained model
        history: Training history
        model_path: Path to the saved model
    """
    print(f"Starting training for {epochs} epochs...")

    # Create a timestamp for this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Checkpoint callback to save model during training
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}.h5")
        print(f"Model checkpoints will be saved to: {model_path}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_dice_coef',
            mode='max',
            verbose=1
        )
    else:
        checkpoint_callback = None
        model_path = None

    # TensorBoard callback for visualization
    log_dir = f"logs_{model_name}_{timestamp}"
    print(f"Logs will be saved to: {log_dir}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    # Learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coef',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )

    # Create a list of callbacks
    callbacks = [tensorboard_callback, early_stopping, reduce_lr]
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)

    # Define custom metrics like Dice coefficient
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

     # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            dice_coef,
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryIoU(threshold=0.5)
        ]
    )

    # Train the model with a custom training loop to handle sample weights
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the final model
    if checkpoint_dir and not os.path.exists(model_path):
        model.save(model_path)
        print(f"Model saved to {model_path}")

    return model, history, model_path

def plot_training_history(history, output_dir=None):
    """
    Plot the training history of the model.

    Args:
        history (tensorflow.keras.callbacks.History): Training history
        output_dir (str): Directory to save plots

    Returns:
        None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a figure with 2x2 subplots
    plt.figure(figsize=(16, 12))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot IoU
    plt.subplot(2, 2, 3)
    plt.plot(history.history['iou'], label='Training IoU')
    plt.plot(history.history['val_iou'], label='Validation IoU')
    plt.title('Model IoU (Intersection over Union)')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(2, 2, 4)
        plt.semilogy(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate (log scale)')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
    else:
        # If no learning rate, plot precision/recall
        plt.subplot(2, 2, 4)
        plt.plot(history.history.get('precision', []), label='Training Precision')
        plt.plot(history.history.get('val_precision', []), label='Validation Precision')
        plt.plot(history.history.get('recall', []), label='Training Recall')
        plt.plot(history.history.get('val_recall', []), label='Validation Recall')
        plt.title('Precision and Recall')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True)

    plt.tight_layout()

    # Save the plot if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        print(f"Training history plot saved to {os.path.join(output_dir, 'training_history.png')}")

    plt.close()


def continue_training(model_path, train_ds, val_ds, epochs=20, **kwargs):
    """
    Continue training from a saved model checkpoint.

    Args:
        model_path (str): Path to the saved model
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        epochs (int): Number of additional epochs
        **kwargs: Additional arguments to pass to train_model

    Returns:
        tuple: Same as train_model
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Continue training
    return train_model(model, train_ds, val_ds, epochs, **kwargs)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config
    from src.model.dataset_unified import prepare_dataset
    from src.model.network import build_unet_resnet50

    # Prepare dataset
    train_ds, val_ds, _, _ = prepare_dataset(
        config.AUGMENTED_DATA_DIR,
        img_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE
    )

    # Build model
    model = build_unet_resnet50(input_shape=(*config.IMAGE_SIZE, 3))

    # Train model
    trained_model, history, model_path = train_model(
        model,
        train_ds,
        val_ds,
        epochs=config.NUM_EPOCHS,
        checkpoint_dir=config.CHECKPOINT_DIR,
        model_name="ultrasound_segmentation"
    )