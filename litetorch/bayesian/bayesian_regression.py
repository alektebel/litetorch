"""
Bayesian Linear Regression implementation template.

Bayesian linear regression extends classical linear regression by treating
model parameters as random variables with probability distributions rather
than fixed point estimates.
"""
import numpy as np


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate prior.
    
    In Bayesian linear regression, we model the relationship:
        y = X @ w + noise, where noise ~ N(0, sigma^2)
    
    Instead of finding a single best estimate for weights w, we maintain
    a probability distribution over possible weights:
        Prior: p(w) = N(w | mu_0, Sigma_0)
        Likelihood: p(y | X, w) = N(y | X @ w, sigma^2 * I)
        Posterior: p(w | X, y) = N(w | mu_n, Sigma_n)
    
    The posterior is computed analytically using Bayes' rule, which is
    tractable when using a Gaussian prior (conjugate prior).
    
    Advantages:
        - Provides uncertainty estimates for predictions
        - Naturally handles regularization through the prior
        - Can update beliefs incrementally as new data arrives
        - Prevents overfitting on small datasets
    
    References:
        - Bishop (2006): "Pattern Recognition and Machine Learning", Chapter 3
        - Murphy (2012): "Machine Learning: A Probabilistic Perspective"
    """
    
    def __init__(self, input_dim, alpha=1.0, beta=1.0):
        """
        Initialize Bayesian Linear Regression model.
        
        Args:
            input_dim: Dimension of input features
            alpha: Precision (inverse variance) of the prior distribution over weights
                   Higher alpha = stronger regularization (narrower prior)
            beta: Precision (inverse variance) of the observation noise
                  Higher beta = less noise (more confident in observations)
        """
        self.input_dim = input_dim
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        
        # Prior distribution parameters: p(w) = N(w | mu_0, Sigma_0)
        self.prior_mean = np.zeros(input_dim)
        self.prior_cov = (1.0 / alpha) * np.eye(input_dim)
        
        # Posterior distribution parameters: p(w | X, y)
        # Initially equal to prior before observing any data
        self.posterior_mean = self.prior_mean.copy()
        self.posterior_cov = self.prior_cov.copy()
        
        # Flag to track if model has been fitted
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Update posterior distribution given observed data.
        
        Computes the posterior distribution p(w | X, y) using Bayes' rule:
            Posterior covariance: Sigma_n = (Sigma_0^-1 + beta * X^T @ X)^-1
            Posterior mean: mu_n = Sigma_n @ (Sigma_0^-1 @ mu_0 + beta * X^T @ y)
        
        Args:
            X: Input features, shape (n_samples, input_dim)
            y: Target values, shape (n_samples,) or (n_samples, 1)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Compute posterior covariance
        # Sigma_n^-1 = Sigma_0^-1 + beta * X^T @ X
        prior_precision = self.alpha * np.eye(self.input_dim)
        data_precision = self.beta * (X.T @ X)
        posterior_precision = prior_precision + data_precision
        
        # Invert to get covariance
        self.posterior_cov = np.linalg.inv(posterior_precision)
        
        # Compute posterior mean
        # mu_n = Sigma_n @ (Sigma_0^-1 @ mu_0 + beta * X^T @ y)
        prior_term = prior_precision @ self.prior_mean.reshape(-1, 1)
        data_term = self.beta * (X.T @ y)
        self.posterior_mean = (self.posterior_cov @ (prior_term + data_term)).flatten()
        
        self.is_fitted = True
        
    def predict(self, X, return_std=False):
        """
        Make predictions with uncertainty estimates.
        
        Computes predictive distribution p(y* | X*, X, y):
            Mean: y* = X* @ mu_n
            Variance: sigma_y^2 = 1/beta + X* @ Sigma_n @ X*^T
        
        Args:
            X: Input features for prediction, shape (n_samples, input_dim)
            return_std: If True, also return standard deviation of predictions
        
        Returns:
            predictions: Mean predictions, shape (n_samples,)
            std (optional): Standard deviation of predictions, shape (n_samples,)
        """
        X = np.array(X)
        
        if not self.is_fitted:
            print("Warning: Model not fitted yet, using prior for predictions")
        
        # Predictive mean: E[y* | X*] = X* @ mu_n
        predictions = X @ self.posterior_mean
        
        if return_std:
            # Predictive variance for each point
            # Var[y* | X*] = 1/beta + X* @ Sigma_n @ X*^T
            variance = (1.0 / self.beta) + np.sum(X @ self.posterior_cov * X, axis=1)
            std = np.sqrt(variance)
            return predictions, std
        
        return predictions
    
    def sample_weights(self, n_samples=1):
        """
        Sample weight vectors from the posterior distribution.
        
        This allows us to generate multiple plausible weight configurations
        and analyze the uncertainty in our model.
        
        Args:
            n_samples: Number of weight samples to draw
            
        Returns:
            weights: Sampled weight vectors, shape (n_samples, input_dim)
        """
        return np.random.multivariate_normal(
            self.posterior_mean, 
            self.posterior_cov, 
            size=n_samples
        )
    
    def get_marginal_likelihood(self, X, y):
        """
        Compute the marginal likelihood (evidence) p(y | X).
        
        The marginal likelihood integrates out the weights:
            p(y | X) = integral p(y | X, w) p(w) dw
        
        This can be computed analytically for Bayesian linear regression
        and is useful for model selection and comparison.
        
        Args:
            X: Input features, shape (n_samples, input_dim)
            y: Target values, shape (n_samples,)
            
        Returns:
            log_evidence: Log marginal likelihood
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n = X.shape[0]
        
        # This is a simplified version
        # Full implementation requires careful handling of matrix determinants
        # and normalization constants
        raise NotImplementedError(
            "Marginal likelihood computation requires careful numerical implementation. "
            "See Bishop (2006) Chapter 3.5 for details."
        )


class BayesianRidge:
    """
    Bayesian Ridge Regression with automatic relevance determination.
    
    This variant learns optimal values for hyperparameters alpha (prior precision)
    and beta (noise precision) from the data using evidence maximization
    (also called Type-II maximum likelihood or empirical Bayes).
    
    The model iteratively updates:
        1. Posterior over weights given current alpha, beta
        2. Hyperparameters alpha, beta to maximize marginal likelihood
    
    This provides automatic feature selection and appropriate regularization
    without cross-validation.
    
    References:
        - Tipping (2001): "Sparse Bayesian Learning and the Relevance Vector Machine"
        - Bishop (2006): Chapter 3.5
    """
    
    def __init__(self, input_dim, max_iter=300, tol=1e-3):
        """
        Initialize Bayesian Ridge Regression.
        
        Args:
            input_dim: Dimension of input features
            max_iter: Maximum number of iterations for hyperparameter optimization
            tol: Convergence tolerance for hyperparameters
        """
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize hyperparameters
        self.alpha = 1.0  # Prior precision
        self.beta = 1.0   # Noise precision
        
        # Model parameters (will be set during fitting)
        self.posterior_mean = None
        self.posterior_cov = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the model using evidence maximization.
        
        Alternates between:
            1. Computing posterior p(w | X, y, alpha, beta)
            2. Updating alpha and beta to maximize p(y | X, alpha, beta)
        
        Args:
            X: Input features, shape (n_samples, input_dim)
            y: Target values, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iter):
            alpha_old = self.alpha
            beta_old = self.beta
            
            # E-step: Compute posterior given current hyperparameters
            prior_precision = self.alpha * np.eye(self.input_dim)
            data_precision = self.beta * (X.T @ X)
            posterior_precision = prior_precision + data_precision
            self.posterior_cov = np.linalg.inv(posterior_precision)
            self.posterior_mean = (
                self.beta * self.posterior_cov @ X.T @ y
            ).flatten()
            
            # M-step: Update hyperparameters to maximize evidence
            # Effective number of parameters (accounts for regularization)
            gamma = self.input_dim - self.alpha * np.trace(self.posterior_cov)
            
            # Update alpha (prior precision)
            self.alpha = gamma / (self.posterior_mean @ self.posterior_mean)
            
            # Update beta (noise precision)
            residual = y - X @ self.posterior_mean.reshape(-1, 1)
            self.beta = (n_samples - gamma) / (residual.T @ residual).item()
            
            # Check convergence
            if (abs(self.alpha - alpha_old) < self.tol and 
                abs(self.beta - beta_old) < self.tol):
                break
        
        self.is_fitted = True
    
    def predict(self, X, return_std=False):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input features, shape (n_samples, input_dim)
            return_std: If True, return standard deviation
            
        Returns:
            predictions: Mean predictions
            std (optional): Standard deviation of predictions
        """
        X = np.array(X)
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = X @ self.posterior_mean
        
        if return_std:
            variance = (1.0 / self.beta) + np.sum(X @ self.posterior_cov * X, axis=1)
            std = np.sqrt(variance)
            return predictions, std
        
        return predictions


# Example usage and demonstration
if __name__ == "__main__":
    print("Bayesian Linear Regression Example")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50
    input_dim = 1
    
    # True function: y = 2x + 1 + noise
    X_train = np.random.uniform(-3, 3, (n_samples, input_dim))
    y_train = 2 * X_train.flatten() + 1 + np.random.randn(n_samples) * 0.5
    
    # Test data
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_true = 2 * X_test.flatten() + 1
    
    # Fit Bayesian Linear Regression
    print("\nFitting Bayesian Linear Regression...")
    model = BayesianLinearRegression(input_dim=input_dim, alpha=1.0, beta=4.0)
    model.fit(X_train, y_train)
    
    print(f"Posterior mean weights: {model.posterior_mean}")
    print(f"Posterior covariance diagonal: {np.diag(model.posterior_cov)}")
    
    # Make predictions with uncertainty
    y_pred, y_std = model.predict(X_test, return_std=True)
    
    print(f"\nPrediction uncertainty (mean std): {y_std.mean():.4f}")
    print(f"Prediction uncertainty (max std): {y_std.max():.4f}")
    
    # Sample weights from posterior
    weight_samples = model.sample_weights(n_samples=5)
    print(f"\nSampled weights from posterior:\n{weight_samples}")
