
import tensorflow as tf
print("\nTensorFlow GPU Verification:")
print("------------------------------")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"GPU device name: {tf.test.gpu_device_name()}")
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print("\nIf you see an empty list for GPU devices, make sure your NVIDIA drivers are up to date.")
