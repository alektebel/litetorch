# CNN and Diffusion Model Implementation Templates

This document describes the newly added implementation templates for Convolutional Neural Networks (CNNs) and enhanced Diffusion models in LiteTorch.

## 🎉 What's New

### 1. CNN Implementation Templates (`litetorch/nn/cnn.py`)

A comprehensive set of CNN layers, building blocks, and complete architectures have been added as implementation templates. These templates include detailed documentation, mathematical formulations, and step-by-step implementation guides.

#### Core CNN Layers

- **Conv2d**: 2D Convolutional layer with He initialization
  - Supports stride, padding, dilation, and grouped convolutions
  - Includes im2col algorithm guide for efficient convolution
  - Detailed forward/backward pass documentation

- **MaxPool2d & AvgPool2d**: Pooling layers for downsampling
  - Configurable kernel size and stride
  - Gradient flow documentation for backpropagation

- **BatchNorm2d**: Batch normalization for training stability
  - Train/eval mode support
  - Running statistics for inference
  - Learnable scale and shift parameters

- **Dropout2d**: Channel-wise dropout for CNNs
  - Drops entire feature maps for better regularization
  - Train/eval mode switching

#### CNN Building Blocks

- **VGGBlock**: Multiple conv layers + pooling
  ```python
  from litetorch.nn import VGGBlock
  block = VGGBlock(in_channels=64, out_channels=128, num_convs=2)
  ```

- **ResidualBlock**: Skip connections for ResNet
  ```python
  from litetorch.nn import ResidualBlock
  block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
  ```

- **BottleneckBlock**: Efficient residual block for deeper networks
  ```python
  from litetorch.nn import BottleneckBlock
  block = BottleneckBlock(in_channels=256, bottleneck_channels=64, 
                         out_channels=256, stride=1)
  ```

- **InceptionModule**: Multi-scale feature extraction
  ```python
  from litetorch.nn import InceptionModule
  module = InceptionModule(in_channels=192, ch1x1=64, 
                          ch3x3red=96, ch3x3=128,
                          ch5x5red=16, ch5x5=32, pool_proj=32)
  ```

- **DepthwiseSeparableConv2d**: Efficient convolution for mobile networks
  ```python
  from litetorch.nn import DepthwiseSeparableConv2d
  conv = DepthwiseSeparableConv2d(in_channels=32, out_channels=64, 
                                 kernel_size=3, padding=1)
  ```

#### Complete CNN Architectures

- **LeNet5**: Classic architecture for digit recognition (MNIST)
- **AlexNet**: Deep CNN for ImageNet classification
- **VGG**: Very deep networks with small 3x3 filters (VGG-11, 13, 16, 19)
- **ResNet**: Residual networks (ResNet-18, 34, 50, 101, 152)

Example usage:
```python
from litetorch.nn import LeNet5, ResNet

# Create LeNet5 for MNIST
model = LeNet5(num_classes=10)

# Create ResNet-18 for ImageNet
resnet18 = ResNet(block_type='basic', num_blocks=[2, 2, 2, 2], num_classes=1000)
```

### 2. Enhanced Diffusion Model Templates

#### DiffusionModel (`litetorch/generative/image_generation.py`)

Significantly enhanced with detailed implementation guides:

**Key Features:**
- Complete noise schedule precomputation (β, α, ᾱ)
- Detailed U-Net architecture guide with attention mechanisms
- Time embedding implementation (sinusoidal encoding)
- Forward diffusion with reparameterization trick
- Reverse diffusion (denoising) step-by-step guide
- Training procedure with optimizer integration
- Generation with progress tracking
- Latent space interpolation

**Mathematical Formulations:**

Forward Process (Adding Noise):
```
x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
where ε ~ N(0, I)
```

Training Objective:
```
L = ||ε - ε_θ(x_t, t)||²
```

Reverse Process (Denoising):
```
x_{t-1} ~ N(μ_θ(x_t, t), σ_t²I)
```

**Usage Example:**
```python
from litetorch.generative.image_generation import DiffusionModel
import numpy as np

# Initialize diffusion model
diffusion = DiffusionModel(
    img_shape=(64, 64, 3),  # 64x64 RGB images
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)

# Build U-Net (TODO: implement)
diffusion.build_unet(base_channels=64, channel_multipliers=(1, 2, 4, 8))

# Training step
images = np.random.randn(32, 64, 64, 3)  # batch of images
loss = diffusion.train_step(images, optimizer)

# Generate images
generated_images = diffusion.generate(num_images=16, show_progress=True)
```

#### VideoDiffusion (`litetorch/generative/video_generation.py`)

Extended diffusion to video with 3D U-Net architecture:

**Key Features:**
- 3D convolutions for processing video volumes (T, H, W, C)
- Temporal attention across frames
- Spatial attention within frames
- Autoregressive generation for long videos
- Frame conditioning and interpolation

**Architecture Components:**
- 3D U-Net with encoder-decoder structure
- Temporal attention: models frame dependencies
- Spatial attention: models spatial relationships
- Time embedding injection
- Skip connections for detail preservation

**Usage Example:**
```python
from litetorch.generative.video_generation import VideoDiffusion

# Initialize video diffusion
video_diffusion = VideoDiffusion(
    video_shape=(16, 64, 64, 3),  # 16 frames, 64x64 RGB
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)

# Build 3D U-Net
video_diffusion.build_3d_unet(
    base_channels=64,
    channel_multipliers=(1, 2, 4, 8),
    use_temporal_attention=True,
    use_spatial_attention=True
)

# Generate videos
generated_videos = video_diffusion.generate(num_videos=4, show_progress=True)

# Generate long videos autoregressively
long_video = video_diffusion.generate_autoregressive(
    num_frames=64,  # Generate 64 frames
    overlap=4,      # 4 frames overlap between chunks
    show_progress=True
)
```

## 📚 References

### CNN Architectures
- LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition" (LeNet)
- Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs" (AlexNet)
- Simonyan & Zisserman (2014): "Very Deep CNNs for Large-Scale Image Recognition" (VGG)
- He et al. (2016): "Deep Residual Learning for Image Recognition" (ResNet)
- Szegedy et al. (2015): "Going Deeper with Convolutions" (Inception)
- Howard et al. (2017): "MobileNets: Efficient CNNs for Mobile Vision Applications"
- Ioffe & Szegedy (2015): "Batch Normalization"

### Diffusion Models
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models"
- Song et al. (2020): "Score-Based Generative Modeling through SDEs"
- Nichol & Dhariwal (2021): "Improved DDPM"
- Ho et al. (2022): "Video Diffusion Models"
- Harvey et al. (2022): "Flexible Diffusion Modeling of Long Videos"
- Blattmann et al. (2023): "Align your Latents: High-Resolution Video Synthesis"

## 🧪 Testing

All templates include initialization tests. Run them with:

```bash
# Test CNN layers and architectures
python -m pytest tests/test_nn_layers.py -v

# Test specific components
python -m pytest tests/test_nn_layers.py::TestConv2d -v
python -m pytest tests/test_nn_layers.py::TestCNNBlocks -v
python -m pytest tests/test_nn_layers.py::TestCNNArchitectures -v
```

### Test Coverage

✅ Conv2d initialization
✅ MaxPool2d initialization
✅ BatchNorm2d initialization and modes
✅ VGGBlock initialization
✅ ResidualBlock initialization (with and without shortcuts)
✅ BottleneckBlock initialization
✅ InceptionModule initialization
✅ DepthwiseSeparableConv2d initialization
✅ LeNet5 architecture initialization

## 📝 Implementation Status

All templates are provided with:
- ✅ Complete class structure
- ✅ Detailed docstrings
- ✅ Mathematical formulations
- ✅ Implementation guides in comments
- ✅ References to original papers
- ⚠️ Forward/backward passes marked as TODO (for educational implementation)

The templates are designed to guide implementation while providing clear documentation of the underlying algorithms and architectures.

## 🚀 Quick Start

```python
# Import CNN components
from litetorch.nn import Conv2d, BatchNorm2d, MaxPool2d, ResidualBlock, LeNet5

# Create a convolutional layer
conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
print(f"Conv2d weight shape: {conv.weight.shape}")  # (64, 3, 3, 3)

# Create a residual block
res_block = ResidualBlock(in_channels=64, out_channels=128, stride=2)

# Create LeNet5 model
lenet = LeNet5(num_classes=10)

# Import Diffusion models
from litetorch.generative.image_generation import DiffusionModel
from litetorch.generative.video_generation import VideoDiffusion

# Create image diffusion model
diffusion = DiffusionModel(img_shape=(32, 32, 3), timesteps=1000)
print(f"Noise schedule computed: {diffusion.betas.shape}")  # (1000,)

# Create video diffusion model
video_diffusion = VideoDiffusion(video_shape=(16, 64, 64, 3), timesteps=1000)
```

## 💡 Tips for Implementation

1. **Start Simple**: Begin with implementing forward passes for basic layers (Conv2d, MaxPool2d)
2. **Test Incrementally**: Test each component before moving to more complex ones
3. **Follow the Guides**: The TODO comments provide step-by-step implementation guidance
4. **Check Shapes**: Verify tensor shapes at each step to catch bugs early
5. **Use References**: Refer to the cited papers for detailed mathematical explanations
6. **Optimize Later**: Focus on correctness first, then optimize for performance

## 📊 Files Added/Modified

### New Files
- `litetorch/nn/cnn.py` (850+ lines)

### Modified Files
- `litetorch/nn/__init__.py` (added CNN exports)
- `litetorch/generative/image_generation.py` (enhanced DiffusionModel)
- `litetorch/generative/video_generation.py` (enhanced VideoDiffusion)
- `tests/test_nn_layers.py` (added CNN tests)
- `README.md` (updated documentation)

## 🎓 Educational Value

These templates are designed for learning and understanding:
- How CNNs process images hierarchically
- How residual connections enable very deep networks
- How different architectures solve different problems
- How diffusion models generate images through iterative denoising
- How to extend 2D models to 3D for video processing

Each template includes detailed comments explaining the "why" behind architectural choices, not just the "how" of implementation.

---

**Happy Learning! ⚡**

For questions or contributions, please open an issue on GitHub.
