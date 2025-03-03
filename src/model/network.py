"""
Module for defining the segmentation model architecture.
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model


def build_unet_resnet50(input_shape=(256, 256, 3), freeze_encoder=True):
    """
    Build a U-Net model with ResNet50 encoder.

    Args:
        input_shape: Input image shape (height, width, channels)
        freeze_encoder: Whether to freeze the encoder weights

    Returns:
        tf.keras.Model: U-Net model with ResNet50 encoder
    """
    # Input
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Use ResNet50 as encoder (without top layers)
    resnet_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Freeze encoder if specified
    if freeze_encoder:
        for layer in resnet_base.layers:
            layer.trainable = False

    # Get skip connections from specific layers
    skips = [
        resnet_base.get_layer('conv1_relu').output,  # 128x128 (assuming 256x256 input)
        resnet_base.get_layer('conv2_block3_out').output,  # 64x64
        resnet_base.get_layer('conv3_block4_out').output,  # 32x32
        resnet_base.get_layer('conv4_block6_out').output  # 16x16
    ]

    # Get bottleneck output
    x = resnet_base.get_layer('conv5_block3_out').output  # 8x8

    # Decoder (upsampling) path
    for i in range(len(skips)):
        # Upsample current feature map
        x = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=3, strides=2, padding='same')(x)

        # Instead of using Lambda layers for resizing, use the built-in Keras resize layer
        # or calculate dimensions explicitly

        # First get the correct skip connection
        skip = skips[-(i + 1)]

        # Use the Resizing layer which doesn't need explicit output shape
        # Get the current shape of the skip connection
        skip_height = skip.shape[1]
        skip_width = skip.shape[2]

        # Resize x to match the skip connection dimensions
        if skip_height is not None and skip_width is not None:
            # If dimensions are static, use the Resizing layer
            x = tf.keras.layers.Resizing(
                height=skip_height,
                width=skip_width,
                interpolation="bilinear",
                crop_to_aspect_ratio=False
            )(x)
        else:
            # For dynamic shapes, we need a different approach
            # Use a more explicit Lambda with output_shape parameter
            def resize_to_match(inputs):
                x_tensor, skip_tensor = inputs
                target_shape = tf.shape(skip_tensor)[1:3]
                return tf.image.resize(x_tensor, target_shape)

            def compute_output_shape(input_shapes):
                return (input_shapes[0][0], None, None, input_shapes[0][3])

            x = tf.keras.layers.Lambda(
                resize_to_match,
                output_shape=compute_output_shape,
                name=f'resize_to_skip_{i}'
            )([x, skip])

        # Concatenate with skip connection
        x = tf.keras.layers.Concatenate()([x, skip])

        # Apply convolutions
        x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    # Final upsampling to original image size
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    # Output layer (binary segmentation)
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_unet(input_shape=(224, 224, 3), filters_base=32, depth=4, dropout_rate=0.1):
    """
    Build a standard U-Net model from scratch (without pretrained weights).
    This can be used as an alternative to the ResNet-50 based model.

    Args:
        input_shape (tuple): Input shape (height, width, channels)
        filters_base (int): Base number of filters
        depth (int): Depth of the U-Net
        dropout_rate (float): Dropout rate for regularization

    Returns:
        tensorflow.keras.Model: Compiled model
    """
    inputs = Input(shape=input_shape)

    # List to store skip connections
    skips = []
    x = inputs

    # Encoder
    for i in range(depth):
        x = Conv2D(filters_base * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters_base * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        skips.append(x)

        # Skip the last max pooling
        if i < depth - 1:
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(dropout_rate)(x)

    # Decoder
    for i in reversed(range(depth - 1)):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters_base * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, skips[i]])
        x = Dropout(dropout_rate)(x)
        x = Conv2D(filters_base * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters_base * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )

    return model


def get_model_summary(model):
    """
    Get a string representation of the model summary.

    Args:
        model (tensorflow.keras.Model): Model to summarize

    Returns:
        str: Model summary as a string
    """
    # Use a string buffer to capture the summary output
    import io
    summary_buffer = io.StringIO()

    # Write the summary to the buffer
    model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))

    # Get the summary as a string
    summary_string = summary_buffer.getvalue()
    summary_buffer.close()

    return summary_string


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add the project root to the path so we can import the config
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    sys.path.append(project_root)

    import config

    # Build and display model
    model = build_unet_resnet50(input_shape=(*config.IMAGE_SIZE, 3))
    model.summary()