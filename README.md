# Qualcomm Environment Setup

This README provides instructions for setting up the environment required to run the project.

---

## Requirements
- [Conda](https://docs.conda.io/en/latest/)
- Python **3.10**
- CUDA **12.6** (for GPU support)

---

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n Qualcomm python=3.10
```


### 2. Install PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install ONNX Runtime (CPU & GPU)
```bash
pip install onnxruntime
pip install onnxruntime-gpu
```

### 4. Install ONNX-TensorFlow

```bash
pip install onnx-tf
```

### 5. Install TensorFlow
```bash
# Current stable release for GPU (Linux / WSL2)
pip install tensorflow[and-cuda]
```

### 6. Other necessary
```bash
pip install matplotlib
```

## Quick Start

### 1. Activate the environment:
```bash
conda activate Qualcomm
```

### 2. Verify PyTorch installation:
```bash
python -c "import torch; print(torch.__version__)"
```

### 3. Verify ONNX Runtime installation:
```bash
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

### 4. (Optional) Verify ONNX-TensorFlow:
```bash
python -c "import onnx_tf; print(onnx_tf.__version__)"
```

### 5. Test GPU Support
```bash
python test_gpu.py
```

```bash
==== PyTorch Test ====
PyTorch version: 2.8.0+cu126
CUDA available: True
CUDA device count: 4
Current device: 0
Device name: NVIDIA GeForce RTX 2080 Ti

==== TensorFlow Test ====
TensorFlow version: 2.20.0
TensorFlow detected 4 GPU(s):
 - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
 - PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')
 - PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU')
 - PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')
```