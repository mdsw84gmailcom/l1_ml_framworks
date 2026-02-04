# l1_ml_frameworks

L1 - Robust environment management (uv)

## Purpose
This project demonstrates how to create a reproducible and isolated machine learning environment using uv, including GPU support for PyTorch.
The environment includes:
- PyTorch (with CUDA support)
- Scikit-learn
- Pandas
- Jupyter
Verificaton script (check_env.py) ensures that all components are correctly installed and working.


## System information
GPU and CUDA driver support were checked using:
```bash
nvidia-smi
```
The output shows:
- GPU: NVIDIA GeForce RTX 4070
- Driver CUDA support: CUDA Version 13.0
This indicates that the system driver supports CUDA 13 APIs and is backward compatible with earlier CUDA versions.


## Choice of PyTorch CUDA Version
PyTorch provides prebuilt binaries for specific CUDA toolkit versions. 
According to the official PyTorch installation guide, CUDA 12.6 (cu126) is one of the currently supported builds. 
Even though the system's NVIDIA driver reports CUDA 13.0 support (via nvidia-smi), NVIDIA drivers are backward compatible. This means a driver supporting CUDA 13 can run software built against earlier CUDA versions, such as CUDA 12.6.

Therefore, PyTorch was installed using the cu126 index. 
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Reference: https://pytorch.org/get-started/locally/

## How to Run the Environment check
```bash
uv sync
```
Then run the verification script:
```bash
uv run check_env.py
```
If everything is set up correctly, the script will:
- Print system and package versions
- Report CUDA / GPU availability
- Perform a tensor computation on CPU or GPU

Example of successful output:
```bash
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
Tensor matmul result: tensor([[19., 22.], [43., 50.]], device='cuda:0')
```


