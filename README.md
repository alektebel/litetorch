
**Educational re-implementation of PyTorch**

The goal of this project is to help understand PyTorch's inner workings by building a minimal, from-scratch version of its most important components.

What is (re)implemented here (very simplified & tiny subset):

- Automatic differentiation (autograd) engine
- Basic tensor operations
- Simple neural network layers (Linear, ReLU, etc.)
- Optimizers (SGD, Adam, Muon)
- Loss functions (MSE, CrossEntropy, ...)
- Plans on integrating later with CUDA kernels for fast execution

**This is NOT meant to be fast or production-ready.**

It is intentionally naive, verbose and well-commented so you can follow how the backward pass, computational graph, etc. actually work under the hood.

Contributions welcome — especially clean explanations, diagrams, tests, or adding one more tiny feature with maximum clarity.

Happy learning! ⚡


=======
# litetorch
Implementation of PyTorch + Reinforcement Learning algorithms for educational purposes

## Project Structure

```
litetorch/
├── litetorch/          # Main package
│   ├── autograd/       # Automatic differentiation
│   ├── nn/             # Neural network modules
│   ├── optim/          # Optimizers
│   ├── rl/             # Reinforcement learning algorithms
│   ├── generative/     # Generative AI models (GANs, VAEs, GPT, BERT, etc.)
│   ├── gnn/            # Graph Neural Networks
│   └── bayesian/       # Bayesian inference methods (NEW!)
├── tests/              # Test files
└── benchmarks/         # Benchmark scripts
```

## TODO List - Implementations

### 1. Core Tensor Operations (litetorch/autograd/)
- [ ] **Tensor class** - Basic tensor implementation with NumPy backend
- [ ] **Autograd engine** - Automatic differentiation with computational graph
- [ ] **Backward propagation** - Gradient computation through the graph
- [ ] **Common operations** - add, sub, mul, div, matmul, pow
- [ ] **Activation functions gradients** - ReLU, Sigmoid, Tanh
- [ ] **Reduction operations** - sum, mean, max, min

### 2. Neural Network Layers (litetorch/nn/)

#### Core Layers
- [ ] **Linear** - Fully connected layer
- [ ] **Dropout** - Dropout regularization
- [ ] **ReLU** - Rectified Linear Unit activation
- [ ] **Sigmoid** - Sigmoid activation
- [ ] **Tanh** - Hyperbolic tangent activation
- [ ] **Softmax** - Softmax activation
- [ ] **Module** - Base class for all neural network modules
- [ ] **Sequential** - Sequential container for layers
- [ ] **MSELoss** - Mean Squared Error loss
- [ ] **CrossEntropyLoss** - Cross-entropy loss

#### CNN Layers (✅ Templates Added!)
- [x] **Conv2d** - 2D Convolutional layer (template with detailed implementation guide)
- [x] **MaxPool2d** - 2D Max pooling (template)
- [x] **AvgPool2d** - 2D Average pooling (template)
- [x] **BatchNorm2d** - Batch normalization (template)
- [x] **Dropout2d** - 2D Dropout for CNNs (template)

#### CNN Building Blocks (✅ Templates Added!)
- [x] **VGGBlock** - VGG-style convolutional block
- [x] **ResidualBlock** - Residual block for ResNet
- [x] **BottleneckBlock** - Bottleneck block for deeper ResNets
- [x] **InceptionModule** - Inception module from GoogLeNet
- [x] **DepthwiseSeparableConv2d** - Efficient convolution for mobile networks

#### CNN Architectures (✅ Templates Added!)
- [x] **LeNet5** - Classic CNN for digit recognition
- [x] **AlexNet** - Deep CNN for ImageNet classification
- [x] **VGG** - Very deep CNN with small filters
- [x] **ResNet** - Residual networks (ResNet-18, 34, 50, 101, 152)

### 3. Optimizers (litetorch/optim/)
- [ ] **SGD** - Stochastic Gradient Descent
- [ ] **SGD with Momentum** - SGD with momentum
- [ ] **Adam** - Adaptive Moment Estimation
- [ ] **RMSprop** - Root Mean Square Propagation
- [ ] **AdaGrad** - Adaptive Gradient Algorithm

### 4. Reinforcement Learning Algorithms (litetorch/rl/)
- [ ] **DQN** - Deep Q-Network
- [ ] **Double DQN** - Double Deep Q-Network
- [ ] **A2C** - Advantage Actor-Critic
- [ ] **PPO** - Proximal Policy Optimization
- [ ] **SAC** - Soft Actor-Critic
- [ ] **TD3** - Twin Delayed Deep Deterministic Policy Gradient
- [ ] **REINFORCE** - Policy Gradient algorithm
- [ ] **Replay Buffer** - Experience replay buffer

### 5. Generative AI Models (litetorch/generative/)

#### Image Generation
- [ ] **GAN** - Generative Adversarial Network (template)
- [ ] **DCGAN** - Deep Convolutional GAN (template)
- [ ] **StyleGAN** - Style-based GAN (template)
- [ ] **VAE** - Variational Autoencoder (template)
- [x] **Diffusion Model** - Denoising Diffusion Probabilistic Model (✅ Enhanced template with detailed implementation guide!)

#### Video Generation
- [ ] **VideoGAN** - Video generation with 3D convolutions (template)
- [ ] **VideoVAE** - Video variational autoencoder (template)
- [ ] **VideoTransformer** - Transformer for video generation (template)
- [x] **VideoDiffusion** - Diffusion model for video (✅ Enhanced template with 3D U-Net details!)
- [ ] **ConditionalVideoGenerator** - Text/image-to-video generation (template)
- [ ] **VideoGAN** - Video generation with 3D convolutions
- [ ] **VideoVAE** - Video variational autoencoder
- [ ] **VideoTransformer** - Transformer for video generation
- [ ] **VideoDiffusion** - Diffusion model for video
- [ ] **ConditionalVideoGenerator** - Text/image-to-video generation

#### Language Generation (✅ COMPLETE!)
- [x] **GPT** - Generative Pre-trained Transformer (FULLY IMPLEMENTED!)
- [x] **Attention Mechanisms** - Scaled dot-product and multi-head attention
- [x] **Transformer Blocks** - Complete decoder blocks with self-attention
- [x] **Position Encodings** - Learned and sinusoidal embeddings
- [x] **Text Generation** - Autoregressive sampling with temperature, top-k, top-p
- [ ] **Transformer** - Standard encoder-decoder transformer (template available)
- [ ] **TransformerXL** - Transformer with segment-level recurrence (template available)
- [ ] **CodeGenModel** - Code generation model (template available)
- [ ] **DialogueModel** - Conversational AI model (template available)

#### Encoder/Decoder Models
- [ ] **BERT** - Bidirectional Encoder Representations from Transformers
- [ ] **RoBERTa** - Robustly Optimized BERT
- [ ] **ELECTRA** - Efficiently Learning an Encoder that Classifies Token Replacements Accurately
- [ ] **Seq2Seq** - Sequence-to-Sequence with attention
- [ ] **T5** - Text-to-Text Transfer Transformer
- [ ] **BART** - Bidirectional and Auto-Regressive Transformer

### 6. Graph Neural Networks (litetorch/gnn/)

#### GNN Layers
- [ ] **GCNLayer** - Graph Convolutional Network layer
- [ ] **GATLayer** - Graph Attention Network layer
- [ ] **GraphSAGELayer** - GraphSAGE layer with neighborhood sampling
- [ ] **GINLayer** - Graph Isomorphism Network layer
- [ ] **EdgeConvLayer** - Edge convolution layer

#### GNN Models
- [ ] **GCN** - Graph Convolutional Network for node classification
- [ ] **GAT** - Graph Attention Network
- [ ] **GraphSAGE** - Inductive node embedding
- [ ] **GIN** - Graph Isomorphism Network for graph classification
- [ ] **GraphTransformer** - Transformer for graphs
- [ ] **MessagePassingNN** - Generic message passing framework
- [ ] **TemporalGNN** - Temporal graph neural network
- [ ] **HeterogeneousGNN** - Heterogeneous graph neural network

#### Pooling Operations
- [ ] **GlobalMeanPool** - Global mean pooling
- [ ] **GlobalMaxPool** - Global max pooling
- [ ] **GlobalSumPool** - Global sum pooling
- [ ] **GlobalAttentionPool** - Attention-based global pooling

### 7. Bayesian Inference (litetorch/bayesian/)

#### Bayesian Regression
- [x] **BayesianLinearRegression** - Bayesian linear regression with conjugate prior
- [x] **BayesianRidge** - Bayesian ridge regression with automatic relevance determination

#### Bayesian Neural Networks
- [x] **BayesianNeuralNetwork** - Bayesian neural network with variational inference
- [x] **MCDropoutBNN** - Monte Carlo Dropout for Bayesian approximation
- [x] **VariationalDense** - Variational dense layer for BNN

#### Variational Inference
- [x] **MeanFieldVariationalInference** - Mean-field variational inference (CAVI)
- [x] **BlackBoxVariationalInference** - Black box VI with reparameterization trick
- [x] **StochasticVariationalInference** - Stochastic VI for large-scale inference
- [x] **AutoEncodingVariationalBayes** - Variational Autoencoder (VAE)

#### MCMC Methods
- [x] **MetropolisHastings** - Metropolis-Hastings algorithm
- [x] **GibbsSampler** - Gibbs sampling for multivariate distributions
- [x] **HamiltonianMonteCarlo** - HMC with leapfrog integrator

## Testing

Each implementation includes test files in the `tests/` directory:
- `tests/test_tensor.py` - Tests for Tensor operations
- `tests/test_reshape_sum.py` - **NEW!** Comprehensive tests for reshape and sum operations (40 tests)
- `tests/test_autograd.py` - **NEW!** Comprehensive tests for automatic differentiation (27 tests)
- `tests/test_nn_layers.py` - Tests for neural network layers
- `tests/test_optimizers.py` - Tests for optimizers
- `tests/test_rl_algorithms.py` - Tests for RL algorithms
- `tests/test_generative.py` - Tests for generative AI models
- `tests/test_gnn.py` - Tests for graph neural networks
- `tests/test_bayesian.py` - **NEW!** Tests for Bayesian inference methods (31 tests)

### Quick Start: Testing

```bash
# Run comprehensive reshape and sum tests
python run_tests_benchmarks.py test

# Run all tensor tests
python run_tests_benchmarks.py test-all

# Run autograd tests
python -m pytest tests/test_autograd.py -v

# Run all tests
python -m pytest tests/ -v
```

See [RESHAPE_SUM_README.md](RESHAPE_SUM_README.md) and [AUTOGRAD_README.md](AUTOGRAD_README.md) for detailed documentation.

## Benchmarks

Each implementation includes benchmark scripts in the `benchmarks/` directory comparing performance against PyTorch or Stable Baselines 3:
- `benchmarks/bench_tensor_ops.py` - Benchmark tensor operations vs PyTorch
- `benchmarks/bench_reshape_sum.py` - **NEW!** Comprehensive benchmarks for reshape and sum vs PyTorch/NumPy
- `benchmarks/bench_autograd.py` - **NEW!** Comprehensive benchmarks for autograd vs PyTorch
- `benchmarks/bench_nn_layers.py` - Benchmark layers vs PyTorch
- `benchmarks/bench_optimizers.py` - Benchmark optimizers vs PyTorch
- `benchmarks/bench_rl_algorithms.py` - Benchmark RL algorithms vs Stable Baselines 3
- `benchmarks/bench_generative.py` - Benchmark generative models vs PyTorch
- `benchmarks/bench_gnn.py` - Benchmark GNN layers vs PyTorch Geometric/DGL

### Quick Start: Benchmarking

```bash
# Run full benchmarks (takes ~5 minutes)
python run_tests_benchmarks.py benchmark

# Run quick benchmarks (~30 seconds)
python run_tests_benchmarks.py benchmark-quick

# Run autograd benchmarks
python benchmarks/bench_autograd.py

# Run autograd benchmarks in quick mode
QUICK=1 python benchmarks/bench_autograd.py

# Run benchmarks without PyTorch (NumPy only)
python run_tests_benchmarks.py benchmark-no-pytorch
```

See [RESHAPE_SUM_README.md](RESHAPE_SUM_README.md) for detailed documentation.

## Installation

```bash
pip install -e .
```

## Usage

### Basic Tensor Operations

```python
from litetorch.tensor import Tensor

# Create a tensor
t = Tensor(shape=(2, 3))
t.data = [1, 2, 3, 4, 5, 6]
print(f"Shape: {t.shape}")
print(f"Data: {t.data}")

# Reshape operation
reshaped = t.reshape((3, 2))
print(f"Reshaped: {reshaped.shape}")

# Sum operation
total = t.sum()
print(f"Sum: {total}")
```

### Examples

See `examples/reshape_sum_examples.py` for comprehensive examples of reshape and sum operations.

```bash
python examples/reshape_sum_examples.py
```

## What's New

### LLM Implementation (Latest) 🚀

We've added a **complete, working implementation of GPT** (Generative Pre-trained Transformer) from scratch!

- **Complete GPT architecture** with attention mechanisms, transformer blocks, and text generation
- **24 comprehensive tests** covering all components (all passing ✅)
- **Performance benchmarks** comparing implementation efficiency
- **8 detailed examples** showing different use cases
- **Full documentation** with architecture diagrams and API reference

**Implemented Components:**
- ✅ Scaled dot-product attention
- ✅ Multi-head attention with parallel heads
- ✅ Transformer decoder blocks with residual connections
- ✅ Layer normalization and feed-forward networks
- ✅ Token and position embeddings (learned)
- ✅ Causal masking for autoregressive generation
- ✅ Complete GPT model with forward pass and loss computation
- ✅ Text generation with temperature, top-k, and top-p sampling

**Files Added:**
```
litetorch/nn/attention.py      # Attention mechanisms (340 lines)
litetorch/nn/layers.py         # Core layers (338 lines)
litetorch/nn/gpt.py           # Complete GPT model (426 lines)
tests/test_llm.py             # Comprehensive tests (506 lines, 24 tests)
benchmarks/bench_llm.py       # Performance benchmarks (377 lines)
examples/llm_examples.py      # Usage examples (326 lines, 8 examples)
LLM_README.md                 # Complete documentation (417 lines)
```

**Quick Start:**
```bash
# Run tests
python -m pytest tests/test_llm.py -v

# Run examples
python examples/llm_examples.py

# Run benchmarks
QUICK=1 python benchmarks/bench_llm.py
```

See [LLM_README.md](LLM_README.md) for complete documentation and examples.

### Autograd Tests and Benchmarks

We've added comprehensive tests and benchmarks for automatic differentiation (autograd):

- **27 comprehensive tests** covering all aspects of autograd functionality
- **Benchmarks** comparing LiteTorch autograd against PyTorch
- **Test coverage** includes:
  - Basic backward pass operations (add, sub, mul, div, power)
  - Chain rule application in computational graphs
  - Matrix multiplication gradients
  - Activation function gradients (ReLU, Sigmoid, Tanh, Softmax)
  - Gradient accumulation and zero_grad
  - Edge cases (no_grad, detach, retain_graph)
  - Neural network backward passes
- **Benchmark coverage** includes:
  - Simple backward pass timing
  - Chain of operations
  - Matrix multiplication backward
  - Activation functions backward
  - Neural network backward pass
  - Memory usage analysis
  - Gradient accumulation performance

Run tests:
```bash
python -m pytest tests/test_autograd.py -v
```

Run benchmarks:
```bash
python benchmarks/bench_autograd.py
# Or in quick mode:
QUICK=1 python benchmarks/bench_autograd.py
```

**Note:** Tests are currently marked as skipped until the autograd engine is implemented in the Tensor class. The tests define the expected API:
- `Tensor(shape, requires_grad=True)`
- `tensor.backward()`
- `tensor.grad` (gradient storage)
- `tensor.zero_grad()`

### Reshape and Sum Operations

We've added comprehensive tests and benchmarks for tensor `reshape` and `sum` operations:

- **40 comprehensive tests** covering all aspects of reshape and sum
- **Benchmarks** comparing LiteTorch against NumPy and PyTorch
- **Easy-to-run scripts** for testing and benchmarking
- **Detailed documentation** in [RESHAPE_SUM_README.md](RESHAPE_SUM_README.md)
- **Examples** demonstrating practical usage

Key features:
- ✅ Reshape: 1D↔2D↔3D↔4D↔5D transformations
- ✅ Reshape: Data integrity preservation  
- ✅ Reshape: Edge cases (single elements, large tensors, etc.)
- ✅ Sum: Without axis (fully working)
- ⚠️ Sum: With axis (partial implementation, documented issues)

**Performance:** LiteTorch is 10-1000x slower than NumPy/PyTorch (pure Python vs C/C++ backend), but offers clear educational value with readable implementation.

Run tests:
```bash
python run_tests_benchmarks.py test
```

Run benchmarks:
```bash
python run_tests_benchmarks.py benchmark-quick
```
