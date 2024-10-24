import torch

# 检测GPU是否可用
if torch.cuda.is_available():
    print("GPU is available")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU instead")
