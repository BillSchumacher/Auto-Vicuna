# Auto-Vicuna

Fast Chat and transformers must be installed via requirements file for now.

## Installing PyTorch and NVIDIA dependencies

First, upgrade wheel and friends.

```
pip install --upgrade setuptools pip wheel
```

For CUDA 11.X:

```
pip install nvidia-cuda-runtime-cu11 --index-url https://pypi.ngc.nvidia.com
```

For CUDA 12.x

```
pip install nvidia-cuda-runtime-cu12 --index-url https://pypi.ngc.nvidia.com
```

For PyTorch cu117
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

For PyTorch cu118
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You might need to add:
```
--upgrade --force-reinstall
```

To the PyTorch installation commands if you have a previous version of PyTorch installed.