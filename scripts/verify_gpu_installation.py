import tensorflow as tf
import os

print("\nSystem Information:")
print("-" * 50)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"GPU device name: {tf.test.gpu_device_name()}")
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")

print("\nEnvironment Variables:")
print("-" * 50)
for var in ['CUDA_HOME', 'CUDA_PATH', 'PATH']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

print("\nCUDA Libraries Check:")
print("-" * 50)
try:
    from tensorflow.python.platform import build_info
    cuda_version = build_info.build_info['cuda_version']
    cudnn_version = build_info.build_info['cudnn_version']
    print(f"CUDA version required by TensorFlow: {cuda_version}")
    print(f"cuDNN version required by TensorFlow: {cudnn_version}")
except:
    print("Could not determine CUDA/cuDNN version info")

# Perform a simple GPU operation to verify functionality
if tf.config.list_physical_devices('GPU'):
    print("\nTesting GPU computation:")
    print("-" * 50)
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result:\n{c}")
    print("GPU test successful!")
else:
    print("\nNo GPU available for testing.")