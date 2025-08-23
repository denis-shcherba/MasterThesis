import os
import torch

# See what PyTorch thinks is visible
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# Show visible GPUs and their UUIDs
for i in range(torch.cuda.device_count()):
    print(f"PyTorch GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  UUID: {torch.cuda.get_device_properties(i).uuid}")
