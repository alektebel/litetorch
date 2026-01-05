
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
│   └── gnn/            # Graph Neural Networks
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
- [ ] **Linear** - Fully connected layer
- [ ] **Conv2d** - 2D Convolutional layer
- [ ] **MaxPool2d** - 2D Max pooling
- [ ] **Dropout** - Dropout regularization
- [ ] **BatchNorm2d** - Batch normalization
- [ ] **ReLU** - Rectified Linear Unit activation
- [ ] **Sigmoid** - Sigmoid activation
- [ ] **Tanh** - Hyperbolic tangent activation
- [ ] **Softmax** - Softmax activation
- [ ] **Module** - Base class for all neural network modules
- [ ] **Sequential** - Sequential container for layers
- [ ] **MSELoss** - Mean Squared Error loss
- [ ] **CrossEntropyLoss** - Cross-entropy loss

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
- [ ] **GAN** - Generative Adversarial Network
- [ ] **DCGAN** - Deep Convolutional GAN
- [ ] **StyleGAN** - Style-based GAN
- [ ] **VAE** - Variational Autoencoder
- [ ] **Diffusion Model** - Denoising Diffusion Probabilistic Model

#### Video Generation
- [ ] **VideoGAN** - Video generation with 3D convolutions
- [ ] **VideoVAE** - Video variational autoencoder
- [ ] **VideoTransformer** - Transformer for video generation
- [ ] **VideoDiffusion** - Diffusion model for video
- [ ] **ConditionalVideoGenerator** - Text/image-to-video generation

#### Language Generation
- [ ] **GPT** - Generative Pre-trained Transformer
- [ ] **Transformer** - Standard encoder-decoder transformer
- [ ] **TransformerXL** - Transformer with segment-level recurrence
- [ ] **CodeGenModel** - Code generation model (Codex-style)
- [ ] **DialogueModel** - Conversational AI model

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

## Testing

Each implementation includes test files in the `tests/` directory:
- `tests/test_tensor.py` - Tests for Tensor operations
- `tests/test_autograd.py` - Tests for automatic differentiation
- `tests/test_nn_layers.py` - Tests for neural network layers
- `tests/test_optimizers.py` - Tests for optimizers
- `tests/test_rl_algorithms.py` - Tests for RL algorithms
- `tests/test_generative.py` - Tests for generative AI models
- `tests/test_gnn.py` - Tests for graph neural networks

## Benchmarks

Each implementation includes benchmark scripts in the `benchmarks/` directory comparing performance against PyTorch or Stable Baselines 3:
- `benchmarks/bench_tensor_ops.py` - Benchmark tensor operations vs PyTorch
- `benchmarks/bench_autograd.py` - Benchmark autograd vs PyTorch
- `benchmarks/bench_nn_layers.py` - Benchmark layers vs PyTorch
- `benchmarks/bench_optimizers.py` - Benchmark optimizers vs PyTorch
- `benchmarks/bench_rl_algorithms.py` - Benchmark RL algorithms vs Stable Baselines 3
- `benchmarks/bench_generative.py` - Benchmark generative models vs PyTorch
- `benchmarks/bench_gnn.py` - Benchmark GNN layers vs PyTorch Geometric/DGL

## Installation

```bash
pip install -e .
```

## Usage

```python
import litetorch as lt

# Example usage will be added as implementations are completed
```
