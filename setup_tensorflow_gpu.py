import os
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Setting up TensorFlow with GPU support...")

# Uninstall current TensorFlow
print("Removing existing TensorFlow installations...")
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"])

# Install TensorFlow with GPU support
print("Installing TensorFlow with GPU support...")
install_package("tensorflow==2.10.0")

# Try to install the latest CUDA and cuDNN packages
print("Installing CUDA toolkit and cuDNN...")
try:
    install_package("tensorflow-gpu==2.10.0")
except:
    print("Note: tensorflow-gpu package installation failed, but this is expected with newer TensorFlow versions")

# Create a verification script
verify_code = """
import tensorflow as tf
print("\\nTensorFlow GPU Verification:")
print("------------------------------")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"GPU device name: {tf.test.gpu_device_name()}")
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print("\\nIf you see an empty list for GPU devices, make sure your NVIDIA drivers are up to date.")
"""

with open("verify_gpu.py", "w") as f:
    f.write(verify_code)

print("\nSetup complete. Verify GPU with:")
print("python verify_gpu.py")

# Print NVIDIA driver information
try:
    print("\nChecking NVIDIA driver...")
    subprocess.call(["nvidia-smi"])
except:
    print("nvidia-smi command not found. This likely means NVIDIA drivers aren't installed or accessible.")
    print("Please ensure you have the latest NVIDIA drivers installed from: https://www.nvidia.com/Download/index.aspx")