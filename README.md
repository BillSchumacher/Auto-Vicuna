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

## LLaMA Weights

If your weights used LLaMATokenizer you may need a different version of transformers.

I have fork that should work located here: https://github.com/BillSchumacher/transformers

You should clone this repo and:
```
pip uninstall transformers
cd path/to/my/transformers/cloned/repo
python setup.py install
```

For some reason pip install -e . causes issues.

If you do that, delete __pycache__, dist and build folders and try the setup.py method instead.
