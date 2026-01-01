**Educational re-implementation of PyTorch**

The goal of this project is to help understand PyTorch's inner workings by building a minimal, from-scratch version of its most important components.

What is (re)implemented here (very simplified & tiny subset):

- Automatic differentiation (autograd) engine
- Basic tensor operations
- Simple neural network layers (Linear, ReLU, etc.)
- Optimizers (SGD, maybe Adam later)
- Loss functions (MSE, CrossEntropy, ...)
- No CUDA → everything runs on CPU/numpy-like backend

**This is NOT meant to be fast or production-ready.**

It is intentionally naive, verbose and well-commented so you can follow how the backward pass, computational graph, etc. actually work under the hood.

Contributions welcome — especially clean explanations, diagrams, tests, or adding one more tiny feature with maximum clarity.

Happy learning! ⚡

EOF
