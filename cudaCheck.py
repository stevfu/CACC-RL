import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Check if CUDA is detected
print(torch.version.cuda)  # Check CUDA version used by PyTorch
print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled