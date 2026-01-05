# Bayesian Inference in LiteTorch

This module provides educational implementations of various Bayesian inference methods for machine learning.

## Overview

Bayesian inference treats model parameters as random variables with probability distributions rather than fixed values. This enables:
- **Uncertainty Quantification**: Measure confidence in predictions
- **Better Generalization**: Natural regularization through priors
- **Principled Model Selection**: Compare models using marginal likelihood
- **Robustness to Overfitting**: Particularly useful for small datasets

## Components

### 1. Bayesian Regression (`bayesian_regression.py`)

#### BayesianLinearRegression
Classic Bayesian linear regression with conjugate Gaussian prior.

```python
from litetorch.bayesian.bayesian_regression import BayesianLinearRegression

# Create model
model = BayesianLinearRegression(input_dim=3, alpha=1.0, beta=1.0)

# Fit on data
model.fit(X_train, y_train)

# Predict with uncertainty
predictions, std = model.predict(X_test, return_std=True)

# Sample weights from posterior
weight_samples = model.sample_weights(n_samples=100)
```

**Key Features:**
- Analytical posterior computation (no MCMC needed)
- Closed-form predictive distribution
- Conjugate prior for tractable inference

#### BayesianRidge
Automatic relevance determination through evidence maximization.

```python
from litetorch.bayesian.bayesian_regression import BayesianRidge

# Model learns optimal regularization automatically
model = BayesianRidge(input_dim=10)
model.fit(X_train, y_train)
predictions, std = model.predict(X_test, return_std=True)
```

**Key Features:**
- Automatic hyperparameter tuning
- Feature selection via ARD
- No cross-validation needed

### 2. Bayesian Neural Networks (`bayesian_nn.py`)

#### BayesianNeuralNetwork
Variational inference for neural networks.

```python
from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork

# Create BNN with specified architecture
bnn = BayesianNeuralNetwork(
    layer_sizes=[input_dim, 50, 20, output_dim],
    prior_std=1.0
)

# Make predictions with uncertainty
mean, epistemic_uncertainty = bnn.predict(X_test, n_samples=50)

# Compute KL divergence for regularization
kl = bnn.kl_divergence()
```

**Key Concepts:**
- Weight distributions instead of point estimates
- ELBO (Evidence Lower Bound) optimization
- Reparameterization trick for gradient flow
- Epistemic uncertainty quantification

#### MCDropoutBNN
Simple Bayesian approximation using dropout at test time.

```python
from litetorch.bayesian.bayesian_nn import MCDropoutBNN

# Create model with dropout
mc_bnn = MCDropoutBNN(
    layer_sizes=[input_dim, 100, 50, output_dim],
    dropout_rate=0.3
)

# Uncertainty via multiple stochastic forward passes
mean, uncertainty = mc_bnn.predict_with_uncertainty(X_test, n_samples=100)
```

**Advantages:**
- Easy to implement (just use dropout at test time)
- Works with existing architectures
- Minimal computational overhead
- Provides calibrated uncertainty estimates

#### VariationalDense
Building block for constructing custom BNNs.

```python
from litetorch.bayesian.bayesian_nn import VariationalDense

# Single variational layer
layer = VariationalDense(in_features=10, out_features=5, prior_std=1.0)
output = layer.forward(X)
kl = layer.kl_divergence()
```

### 3. Variational Inference (`variational_inference.py`)

General framework for approximate Bayesian inference.

#### MeanFieldVariationalInference
Coordinate ascent variational inference (CAVI).

```python
from litetorch.bayesian.variational_inference import MeanFieldVariationalInference

mf_vi = MeanFieldVariationalInference(n_components=10)
# Model-specific coordinate ascent updates would go here
```

**Key Assumption:**
- Factorized variational distribution: q(z) = ∏ᵢ q(zᵢ)

#### BlackBoxVariationalInference
Gradient-based VI using reparameterization.

```python
from litetorch.bayesian.variational_inference import BlackBoxVariationalInference

bb_vi = BlackBoxVariationalInference(latent_dim=20, learning_rate=0.001)
samples = bb_vi.sample_latent(n_samples=100)
```

**Key Feature:**
- Works with any model where log p(x, z) can be evaluated

#### StochasticVariationalInference
Scalable VI for large datasets using mini-batches.

```python
from litetorch.bayesian.variational_inference import StochasticVariationalInference

svi = StochasticVariationalInference(batch_size=128, learning_rate=0.01)
# Process data in mini-batches with natural gradient descent
```

**Applications:**
- Topic modeling on large corpora
- Bayesian deep learning
- Large-scale probabilistic models

#### AutoEncodingVariationalBayes (VAE)
Neural network-based variational autoencoder.

```python
from litetorch.bayesian.variational_inference import AutoEncodingVariationalBayes

vae = AutoEncodingVariationalBayes(
    input_dim=784,    # e.g., 28x28 images
    latent_dim=20,
    hidden_dim=400
)

# Encoder: q(z | x)
mu, log_var = vae.encode(x)
z = vae.reparameterize(mu, log_var)

# Decoder: p(x | z)
x_recon = vae.decode(z)

# Loss: negative ELBO
loss, recon_loss, kl_loss = vae.elbo_loss(x)
```

**Applications:**
- Generative modeling
- Representation learning
- Data compression
- Semi-supervised learning

### 4. MCMC Methods (`mcmc.py`)

Sampling-based approaches for Bayesian inference.

#### MetropolisHastings
Classic MCMC algorithm with proposal distribution.

```python
from litetorch.bayesian.mcmc import MetropolisHastings

def target_log_prob(theta):
    return -0.5 * np.sum(theta ** 2)  # Standard normal

mh = MetropolisHastings(
    target_log_prob=target_log_prob,
    proposal_std=1.0
)

samples, acceptance_rate = mh.sample(
    theta_init=np.zeros(10),
    n_samples=10000,
    burn_in=1000,
    thin=5
)
```

**Key Parameters:**
- `proposal_std`: Controls proposal step size (tune for ~50% acceptance)
- `burn_in`: Initial samples to discard
- `thin`: Keep every n-th sample to reduce correlation

#### GibbsSampler
Sample from conditional distributions iteratively.

```python
from litetorch.bayesian.mcmc import GibbsSampler

def sample_x_given_y(state):
    # Return sample from p(x | y, data)
    return ...

def sample_y_given_x(state):
    # Return sample from p(y | x, data)
    return ...

gibbs = GibbsSampler(
    conditional_samplers=[sample_x_given_y, sample_y_given_x]
)

samples = gibbs.sample(
    theta_init=[0.0, 0.0],
    n_samples=10000,
    burn_in=1000
)
```

**Advantages:**
- No tuning parameters
- Always accepts proposals (acceptance rate = 1)
- Often faster convergence

**Requirements:**
- Must know conditional distributions p(θᵢ | θ₋ᵢ, data)

#### HamiltonianMonteCarlo
Physics-inspired MCMC with gradient information.

```python
from litetorch.bayesian.mcmc import HamiltonianMonteCarlo

def target_log_prob(theta):
    return -0.5 * np.sum(theta ** 2)

def grad_log_prob(theta):
    return -theta

hmc = HamiltonianMonteCarlo(
    target_log_prob=target_log_prob,
    grad_log_prob=grad_log_prob,
    step_size=0.1,
    n_steps=20
)

samples, acceptance_rate = hmc.sample(
    theta_init=np.zeros(10),
    n_samples=10000,
    burn_in=1000
)
```

**Advantages:**
- High acceptance rates (often >90%)
- Explores parameter space efficiently
- Excellent for high-dimensional problems
- Less random walk behavior

**Requirements:**
- Requires gradient ∇ log π(θ)

## Examples

Comprehensive examples demonstrating all methods:

```bash
# Run all Bayesian inference examples
python examples/bayesian_inference_examples.py
```

Individual examples:
1. **Bayesian Linear Regression**: Uncertainty in linear models
2. **Bayesian Neural Networks**: Deep learning with uncertainty
3. **MC Dropout**: Practical Bayesian approximation
4. **Variational Inference**: Scalable approximate inference
5. **MCMC Sampling**: Gold-standard sampling methods

## Testing

Run comprehensive tests for all Bayesian methods:

```bash
# Run all Bayesian tests (31 tests)
python -m unittest tests.test_bayesian -v

# Run specific test class
python -m unittest tests.test_bayesian.TestBayesianRegression -v
python -m unittest tests.test_bayesian.TestBayesianNN -v
python -m unittest tests.test_bayesian.TestVariationalInference -v
python -m unittest tests.test_bayesian.TestMCMC -v
```

## Key Concepts

### Uncertainty Types

1. **Epistemic Uncertainty** (Model Uncertainty)
   - Uncertainty about model parameters
   - Reducible with more data
   - Captured by posterior distribution over weights
   - Example: BNN prediction variance

2. **Aleatoric Uncertainty** (Data Uncertainty)
   - Inherent noise in observations
   - Irreducible even with infinite data
   - Modeled by likelihood variance
   - Example: Observation noise σ² in regression

### ELBO (Evidence Lower Bound)

The objective function in variational inference:

```
ELBO = 𝔼_q(z)[log p(x | z)] - KL[q(z) || p(z)]
     = Likelihood term     - Complexity penalty
```

Maximizing ELBO ≈ Minimizing KL[q(z) || p(z | x)]

### Reparameterization Trick

For continuous latent variables, enable gradient flow through sampling:

```
Instead of: z ~ q(z | φ)
Use: z = μ(φ) + σ(φ) ⊙ ε, where ε ~ N(0, I)
```

This transforms stochastic node into deterministic function + external randomness.

### Conjugacy

Prior p(θ) and likelihood p(x | θ) are conjugate if posterior p(θ | x) has same form as prior.

Examples:
- Gaussian prior + Gaussian likelihood → Gaussian posterior
- Beta prior + Binomial likelihood → Beta posterior
- Dirichlet prior + Multinomial likelihood → Dirichlet posterior

## References

### Books
- Bishop (2006): "Pattern Recognition and Machine Learning"
- Murphy (2012): "Machine Learning: A Probabilistic Perspective"
- Robert & Casella (2004): "Monte Carlo Statistical Methods"
- Gelman et al. (2013): "Bayesian Data Analysis"

### Papers
- **Bayesian Neural Networks:**
  - Blundell et al. (2015): "Weight Uncertainty in Neural Networks"
  - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"

- **Variational Inference:**
  - Kingma & Welling (2014): "Auto-Encoding Variational Bayes"
  - Blei et al. (2017): "Variational Inference: A Review for Statisticians"
  - Hoffman et al. (2013): "Stochastic Variational Inference"

- **MCMC:**
  - Neal (2011): "MCMC using Hamiltonian dynamics"
  - Betancourt (2017): "A Conceptual Introduction to Hamiltonian Monte Carlo"

## Educational Notes

These implementations prioritize clarity and understanding over performance:
- Extensive documentation and comments
- Clear separation of concepts
- Educational examples
- Not optimized for production use

For production Bayesian inference, consider:
- **PyMC3** / **PyMC**: General probabilistic programming
- **Stan**: HMC-based inference with NUTS sampler
- **TensorFlow Probability**: Bayesian deep learning
- **Pyro**: Deep probabilistic programming on PyTorch
- **NumPyro**: JAX-based probabilistic programming

## Contributing

Contributions welcome! Areas for improvement:
- Additional inference algorithms (NUTS, ADVI, etc.)
- More examples and tutorials
- Visualization utilities
- Performance optimizations
- Documentation improvements

---

**Remember**: The goal is understanding, not performance! Use this code to learn how Bayesian methods work, then use production libraries for real applications.
