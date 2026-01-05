"""
Bayesian Neural Networks (BNN) implementation templates.

Bayesian Neural Networks treat neural network weights as probability distributions
rather than point estimates, enabling uncertainty quantification in predictions.
"""
import numpy as np


class BayesianNeuralNetwork:
    """
    Bayesian Neural Network with variational inference.
    
    A Bayesian Neural Network (BNN) places a prior distribution over the weights
    of a neural network. Unlike standard neural networks that learn a single set
    of weights, BNNs learn a distribution over weights, which allows them to:
        - Quantify uncertainty in predictions
        - Avoid overfitting on small datasets
        - Provide calibrated confidence estimates
    
    Training a BNN exactly requires intractable integrals, so we use
    variational inference to approximate the posterior distribution over weights.
    
    Key concepts:
        - Prior: p(w) - Initial belief about weights (e.g., Gaussian)
        - Likelihood: p(D | w) - Probability of data given weights
        - Posterior: p(w | D) - Updated belief about weights after seeing data
        - Variational distribution: q(w | theta) - Tractable approximation to posterior
    
    The variational approach optimizes the Evidence Lower Bound (ELBO):
        ELBO = E_{q(w)}[log p(D | w)] - KL[q(w) || p(w)]
        
    References:
        - Blundell et al. (2015): "Weight Uncertainty in Neural Networks"
        - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
        - Graves (2011): "Practical Variational Inference for Neural Networks"
    """
    
    def __init__(self, layer_sizes, prior_std=1.0):
        """
        Initialize Bayesian Neural Network.
        
        Args:
            layer_sizes: List of layer dimensions, e.g., [input_dim, hidden_dim, output_dim]
            prior_std: Standard deviation of the Gaussian prior over weights
        """
        self.layer_sizes = layer_sizes
        self.prior_std = prior_std
        self.n_layers = len(layer_sizes) - 1
        
        # Variational parameters (mean and log std for each weight)
        # Each layer has weight matrix W and bias b
        self.weight_means = []
        self.weight_log_stds = []
        self.bias_means = []
        self.bias_log_stds = []
        
        # Initialize variational parameters
        for i in range(self.n_layers):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            
            # Xavier/Glorot initialization for means
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.weight_means.append(
                np.random.uniform(-limit, limit, (in_dim, out_dim))
            )
            self.bias_means.append(np.zeros(out_dim))
            
            # Initialize log standard deviations (actual std = exp(log_std))
            # Start with small uncertainty
            self.weight_log_stds.append(
                np.ones((in_dim, out_dim)) * np.log(0.1)
            )
            self.bias_log_stds.append(
                np.ones(out_dim) * np.log(0.1)
            )
    
    def sample_weights(self):
        """
        Sample weights from the variational distribution q(w | theta).
        
        Uses the reparameterization trick:
            w = mu + sigma * epsilon, where epsilon ~ N(0, I)
        
        This allows gradients to flow through the sampling operation.
        
        Returns:
            weights: List of (W, b) tuples for each layer
        """
        weights = []
        
        for i in range(self.n_layers):
            # Reparameterization trick
            w_epsilon = np.random.randn(*self.weight_means[i].shape)
            b_epsilon = np.random.randn(*self.bias_means[i].shape)
            
            w_std = np.exp(self.weight_log_stds[i])
            b_std = np.exp(self.bias_log_stds[i])
            
            W = self.weight_means[i] + w_std * w_epsilon
            b = self.bias_means[i] + b_std * b_epsilon
            
            weights.append((W, b))
        
        return weights
    
    def forward(self, x, weights):
        """
        Forward pass through the network with given weights.
        
        Args:
            x: Input data, shape (batch_size, input_dim)
            weights: List of (W, b) tuples for each layer
            
        Returns:
            output: Network output, shape (batch_size, output_dim)
        """
        activation = x
        
        for i, (W, b) in enumerate(weights):
            # Linear transformation
            activation = activation @ W + b
            
            # Apply ReLU activation for hidden layers
            if i < self.n_layers - 1:
                activation = np.maximum(0, activation)
        
        return activation
    
    def kl_divergence(self):
        """
        Compute KL divergence between variational distribution and prior.
        
        KL[q(w) || p(w)] for Gaussian distributions can be computed analytically:
            KL = 0.5 * sum(log(sigma_prior^2 / sigma_q^2) + 
                           sigma_q^2 / sigma_prior^2 + 
                           (mu_q - mu_prior)^2 / sigma_prior^2 - 1)
        
        For zero-mean Gaussian prior p(w) = N(0, prior_std^2 * I):
            KL = 0.5 * sum(log(prior_std^2 / sigma_q^2) + 
                           sigma_q^2 / prior_std^2 + 
                           mu_q^2 / prior_std^2 - 1)
        
        Returns:
            kl: KL divergence value
        """
        kl = 0.0
        prior_var = self.prior_std ** 2
        
        for i in range(self.n_layers):
            # KL for weights
            w_var = np.exp(2 * self.weight_log_stds[i])
            w_kl = 0.5 * np.sum(
                np.log(prior_var / w_var) + 
                w_var / prior_var + 
                self.weight_means[i]**2 / prior_var - 
                1
            )
            
            # KL for biases
            b_var = np.exp(2 * self.bias_log_stds[i])
            b_kl = 0.5 * np.sum(
                np.log(prior_var / b_var) + 
                b_var / prior_var + 
                self.bias_means[i]**2 / prior_var - 
                1
            )
            
            kl += w_kl + b_kl
        
        return kl
    
    def elbo_loss(self, x, y, n_samples=1):
        """
        Compute the negative ELBO (loss to minimize).
        
        ELBO = E_{q(w)}[log p(y | x, w)] - KL[q(w) || p(w)]
        
        We approximate the expectation using Monte Carlo sampling.
        
        Args:
            x: Input data, shape (batch_size, input_dim)
            y: Target data, shape (batch_size, output_dim)
            n_samples: Number of weight samples for MC approximation
            
        Returns:
            loss: Negative ELBO
        """
        # Likelihood term (Monte Carlo approximation)
        log_likelihood = 0.0
        
        for _ in range(n_samples):
            weights = self.sample_weights()
            y_pred = self.forward(x, weights)
            
            # Assuming Gaussian likelihood: p(y | x, w) = N(y | f(x, w), sigma^2)
            # For simplicity, use MSE (equivalent to log likelihood with fixed sigma)
            log_likelihood += -np.mean((y - y_pred) ** 2)
        
        log_likelihood /= n_samples
        
        # KL divergence term
        kl = self.kl_divergence()
        
        # Scale KL by dataset size (important for proper balance)
        n_data = x.shape[0]
        kl_scaled = kl / n_data
        
        # Negative ELBO (we want to minimize this)
        return -log_likelihood + kl_scaled
    
    def predict(self, x, n_samples=10):
        """
        Make predictions with uncertainty quantification.
        
        Samples multiple weight configurations and averages their predictions.
        The variance across samples provides uncertainty estimates.
        
        Args:
            x: Input data, shape (batch_size, input_dim)
            n_samples: Number of forward passes with different weight samples
            
        Returns:
            mean: Mean prediction, shape (batch_size, output_dim)
            std: Standard deviation of predictions (epistemic uncertainty)
        """
        predictions = []
        
        for _ in range(n_samples):
            weights = self.sample_weights()
            y_pred = self.forward(x, weights)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Mean and std across samples
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std
    
    def train_step(self, x, y, learning_rate=0.001):
        """
        Single training step using gradient descent on negative ELBO.
        
        Note: This is a template. Full implementation would require:
            - Automatic differentiation for computing gradients
            - Proper optimization algorithm (Adam, etc.)
            - Batch processing
        
        Args:
            x: Input batch
            y: Target batch
            learning_rate: Learning rate
        """
        raise NotImplementedError(
            "Training requires automatic differentiation. "
            "See Blundell et al. (2015) for full algorithm details."
        )


class MCDropoutBNN:
    """
    Bayesian Neural Network using Monte Carlo Dropout.
    
    MC Dropout is a practical approximation to Bayesian inference in neural networks.
    By applying dropout at test time and averaging predictions over multiple forward
    passes, we can approximate sampling from the posterior distribution over weights.
    
    Key insight: Dropout training is equivalent to approximate variational inference
    with a specific variational distribution (Gal & Ghahramani, 2016).
    
    Advantages:
        - Simple to implement (just use dropout at test time)
        - Works with any existing neural network architecture
        - Provides calibrated uncertainty estimates
        - Minimal computational overhead
    
    References:
        - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
        - Srivastava et al. (2014): "Dropout: A Simple Way to Prevent Neural Networks 
          from Overfitting"
    """
    
    def __init__(self, layer_sizes, dropout_rate=0.5):
        """
        Initialize MC Dropout BNN.
        
        Args:
            layer_sizes: List of layer dimensions
            dropout_rate: Dropout probability (typical values: 0.1 to 0.5)
        """
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize weights (in practice, these would be trained)
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            
            W = np.random.uniform(-limit, limit, (in_dim, out_dim))
            b = np.zeros(out_dim)
            
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, x, training=False):
        """
        Forward pass with optional dropout.
        
        Args:
            x: Input data
            training: If True, apply dropout
            
        Returns:
            output: Network output
        """
        activation = x
        
        for i in range(self.n_layers):
            # Apply dropout (if training or doing MC sampling)
            if training:
                mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                         size=activation.shape)
                # Scale by 1/(1-p) to maintain expected value
                activation = activation * mask / (1 - self.dropout_rate)
            
            # Linear transformation
            activation = activation @ self.weights[i] + self.biases[i]
            
            # ReLU activation for hidden layers
            if i < self.n_layers - 1:
                activation = np.maximum(0, activation)
        
        return activation
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty using Monte Carlo Dropout.
        
        Performs multiple forward passes with dropout enabled, treating each
        pass as a sample from the approximate posterior over network functions.
        
        Args:
            x: Input data
            n_samples: Number of stochastic forward passes
            
        Returns:
            mean: Mean prediction (epistemic uncertainty reduced by averaging)
            std: Standard deviation (measures epistemic uncertainty)
        """
        predictions = []
        
        for _ in range(n_samples):
            # Forward pass with dropout enabled
            y_pred = self.forward(x, training=True)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Compute statistics
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std


class VariationalDense:
    """
    Variational Dense Layer for BNN.
    
    A dense (fully-connected) layer where weights are represented as distributions
    rather than point estimates. Used as building block for Bayesian Neural Networks.
    
    Each weight has:
        - Mean (mu): Expected value of the weight
        - Log standard deviation (log_sigma): Uncertainty in the weight
    
    During forward pass, weights are sampled: w = mu + sigma * epsilon
    where epsilon ~ N(0, 1) (reparameterization trick).
    """
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        """
        Initialize Variational Dense Layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            prior_std: Standard deviation of prior distribution
        """
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Variational parameters
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight_mu = np.random.uniform(-limit, limit, 
                                          (in_features, out_features))
        self.weight_log_sigma = np.ones((in_features, out_features)) * np.log(0.1)
        
        self.bias_mu = np.zeros(out_features)
        self.bias_log_sigma = np.ones(out_features) * np.log(0.1)
    
    def forward(self, x):
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Layer output
        """
        # Sample weights using reparameterization trick
        w_epsilon = np.random.randn(self.in_features, self.out_features)
        w_sigma = np.exp(self.weight_log_sigma)
        W = self.weight_mu + w_sigma * w_epsilon
        
        b_epsilon = np.random.randn(self.out_features)
        b_sigma = np.exp(self.bias_log_sigma)
        b = self.bias_mu + b_sigma * b_epsilon
        
        return x @ W + b
    
    def kl_divergence(self):
        """
        Compute KL divergence for this layer.
        
        Returns:
            kl: KL[q(w) || p(w)] for layer weights
        """
        prior_var = self.prior_std ** 2
        
        # Weight KL
        w_var = np.exp(2 * self.weight_log_sigma)
        w_kl = 0.5 * np.sum(
            np.log(prior_var / w_var) + 
            w_var / prior_var + 
            self.weight_mu**2 / prior_var - 1
        )
        
        # Bias KL
        b_var = np.exp(2 * self.bias_log_sigma)
        b_kl = 0.5 * np.sum(
            np.log(prior_var / b_var) + 
            b_var / prior_var + 
            self.bias_mu**2 / prior_var - 1
        )
        
        return w_kl + b_kl


# Example usage
if __name__ == "__main__":
    print("Bayesian Neural Network Example")
    print("=" * 50)
    
    # Create a simple BNN
    layer_sizes = [2, 10, 1]  # Input=2, Hidden=10, Output=1
    bnn = BayesianNeuralNetwork(layer_sizes=layer_sizes, prior_std=1.0)
    
    print(f"Created BNN with architecture: {layer_sizes}")
    print(f"Number of layers: {bnn.n_layers}")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    # Make predictions with uncertainty
    print("\nMaking predictions with uncertainty quantification...")
    mean_pred, std_pred = bnn.predict(X, n_samples=10)
    
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Uncertainty (std) shape: {std_pred.shape}")
    print(f"Average uncertainty: {std_pred.mean():.4f}")
    
    # MC Dropout BNN
    print("\n" + "=" * 50)
    print("MC Dropout BNN Example")
    mc_bnn = MCDropoutBNN(layer_sizes=[2, 10, 1], dropout_rate=0.5)
    
    mean_mc, std_mc = mc_bnn.predict_with_uncertainty(X, n_samples=100)
    print(f"MC Dropout - Average uncertainty: {std_mc.mean():.4f}")
