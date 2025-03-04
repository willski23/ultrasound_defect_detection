import os
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uninstall current TensorFlow
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"])

# Install TensorFlow 2.12 with GPU support and compatible CUDA packages
print("Installing TensorFlow with GPU support...")
install_package("tensorflow==2.12.0")

# Install CUDA Toolkit compatibilities
print("Installing CUDA compatibility packages...")
install_package("nvidia-cudnn-cu11==8.6.0.163")

# Print verification info
print("\nInstallation complete. Verifying GPU recognition...")

# Create a verification script
verify_code = """
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("Is built with CUDA:", tf.test.is_built_with_cuda())
"""

with open("verify_gpu.py", "w") as f:
    f.write(verify_code)

# Run verification
print("\nRun the verification script with:")
print("python verify_gpu.py")