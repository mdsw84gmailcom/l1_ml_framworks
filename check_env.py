import sys
import platform
import torch
import pandas as pd
import sklearn

print("=== CHECK_ENV ===")
print("Python:", sys.version.split()[0])
print("OS:", platform.platform())
print()

print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print()

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

device = "cuda" if cuda_available else "cpu"
if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not available")

print("Using device:", device)

# Tensor computation
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
z = x @ y
print("Tensor matmul result:\n", z)

print("=== OK ===")
