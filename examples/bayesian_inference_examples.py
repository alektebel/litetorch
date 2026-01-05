"""
Bayesian Inference Examples

This script demonstrates how to use the Bayesian inference templates
in LiteTorch, including Bayesian Linear Regression, Bayesian Neural Networks,
Variational Inference, and MCMC methods.
"""
import numpy as np


def example_bayesian_linear_regression():
    """
    Example: Bayesian Linear Regression
    
    Demonstrates how to perform Bayesian linear regression with uncertainty
    quantification on a simple synthetic dataset.
    """
    print("=" * 70)
    print("Example 1: Bayesian Linear Regression")
    print("=" * 70)
    
    from litetorch.bayesian.bayesian_regression import BayesianLinearRegression
    
    # Generate synthetic data: y = 3x + 2 + noise
    np.random.seed(42)
    n_samples = 30
    X_train = np.random.uniform(-5, 5, (n_samples, 1))
    y_train = 3 * X_train.flatten() + 2 + np.random.randn(n_samples) * 1.5
    
    # Create and fit model
    print("\n1. Training Bayesian Linear Regression...")
    model = BayesianLinearRegression(input_dim=1, alpha=1.0, beta=1.0)
    model.fit(X_train, y_train)
    
    print(f"   Posterior mean weight: {model.posterior_mean[0]:.4f} (true: 3.0)")
    print(f"   Weight uncertainty (std): {np.sqrt(model.posterior_cov[0, 0]):.4f}")
    
    # Make predictions with uncertainty
    print("\n2. Making predictions with uncertainty...")
    X_test = np.array([[-3.0], [0.0], [3.0]])
    predictions, std = model.predict(X_test, return_std=True)
    
    for i, (x, pred, s) in enumerate(zip(X_test.flatten(), predictions, std)):
        true_y = 3 * x + 2
        print(f"   x={x:5.1f}: prediction={pred:6.2f} ± {s:.2f}, true={true_y:6.2f}")
    
    # Sample weight distributions
    print("\n3. Sampling from posterior distribution...")
    weight_samples = model.sample_weights(n_samples=5)
    print(f"   Weight samples: {weight_samples.flatten()}")
    print()


def example_bayesian_neural_network():
    """
    Example: Bayesian Neural Network
    
    Demonstrates Bayesian Neural Networks for classification with
    uncertainty quantification.
    """
    print("=" * 70)
    print("Example 2: Bayesian Neural Network")
    print("=" * 70)
    
    from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork
    
    # Create synthetic classification data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    
    print("\n1. Creating Bayesian Neural Network...")
    layer_sizes = [2, 16, 8, 1]  # Input -> Hidden -> Hidden -> Output
    bnn = BayesianNeuralNetwork(layer_sizes=layer_sizes, prior_std=1.0)
    print(f"   Architecture: {layer_sizes}")
    
    # Make predictions with uncertainty
    print("\n2. Making predictions with uncertainty quantification...")
    X_test = np.array([[0.5, 0.5], [-0.5, -0.5], [1.0, -1.0]])
    mean_pred, std_pred = bnn.predict(X_test, n_samples=20)
    
    print(f"   Mean predictions: {mean_pred.flatten()}")
    print(f"   Prediction uncertainties: {std_pred.flatten()}")
    
    # Compute KL divergence
    print("\n3. Computing KL divergence (regularization term)...")
    kl = bnn.kl_divergence()
    print(f"   KL[q(w) || p(w)] = {kl:.4f}")
    print()


def example_mc_dropout():
    """
    Example: Monte Carlo Dropout for Bayesian Approximation
    
    Demonstrates using dropout at test time for uncertainty estimation.
    """
    print("=" * 70)
    print("Example 3: Monte Carlo Dropout")
    print("=" * 70)
    
    from litetorch.bayesian.bayesian_nn import MCDropoutBNN
    
    # Create MC Dropout BNN
    np.random.seed(42)
    print("\n1. Creating MC Dropout Network...")
    mc_bnn = MCDropoutBNN(layer_sizes=[3, 32, 16, 1], dropout_rate=0.3)
    print(f"   Dropout rate: {mc_bnn.dropout_rate}")
    
    # Generate test data
    X_test = np.random.randn(10, 3)
    
    # Predictions with uncertainty
    print("\n2. Uncertainty estimation via MC Dropout...")
    mean, std = mc_bnn.predict_with_uncertainty(X_test, n_samples=100)
    
    print(f"   Average prediction uncertainty: {std.mean():.4f}")
    print(f"   Max prediction uncertainty: {std.max():.4f}")
    print(f"   Min prediction uncertainty: {std.min():.4f}")
    print()


def example_variational_inference():
    """
    Example: Variational Inference
    
    Demonstrates different variational inference methods.
    """
    print("=" * 70)
    print("Example 4: Variational Inference")
    print("=" * 70)
    
    from litetorch.bayesian.variational_inference import (
        MeanFieldVariationalInference,
        BlackBoxVariationalInference,
        StochasticVariationalInference
    )
    
    # Mean-Field VI
    print("\n1. Mean-Field Variational Inference...")
    mf_vi = MeanFieldVariationalInference(n_components=10)
    print(f"   Number of latent components: {mf_vi.n_components}")
    print(f"   Initial variational parameters: μ={mf_vi.q_means.mean():.2f}, "
          f"log_σ={mf_vi.q_log_stds.mean():.2f}")
    
    # Black Box VI
    print("\n2. Black Box Variational Inference...")
    bb_vi = BlackBoxVariationalInference(latent_dim=5, learning_rate=0.001)
    
    # Sample from variational distribution
    samples = bb_vi.sample_latent(n_samples=100)
    print(f"   Sampled {samples.shape[0]} latent codes")
    print(f"   Sample statistics: μ={samples.mean():.4f}, σ={samples.std():.4f}")
    
    # Stochastic VI
    print("\n3. Stochastic Variational Inference...")
    svi = StochasticVariationalInference(batch_size=64, learning_rate=0.01)
    
    # Demonstrate learning rate schedule
    print("   Learning rate decay schedule:")
    for iteration in [0, 100, 500, 1000]:
        svi.iteration = iteration
        lr = svi.get_learning_rate()
        print(f"      Iteration {iteration:4d}: lr = {lr:.6f}")
    print()


def example_mcmc():
    """
    Example: MCMC Sampling
    
    Demonstrates Markov Chain Monte Carlo methods for sampling from
    complex distributions.
    """
    print("=" * 70)
    print("Example 5: MCMC Sampling")
    print("=" * 70)
    
    from litetorch.bayesian.mcmc import (
        MetropolisHastings,
        HamiltonianMonteCarlo,
        GibbsSampler
    )
    
    # Target: 2D Gaussian with correlation
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    cov_inv = np.linalg.inv(cov)
    
    def target_log_prob(theta):
        diff = theta - mean
        return -0.5 * diff @ cov_inv @ diff
    
    def grad_log_prob(theta):
        return -cov_inv @ (theta - mean)
    
    # Metropolis-Hastings
    print("\n1. Metropolis-Hastings Sampling...")
    np.random.seed(42)
    mh = MetropolisHastings(target_log_prob=target_log_prob, proposal_std=0.5)
    
    samples_mh, accept_rate = mh.sample(
        theta_init=np.array([0.0, 0.0]),
        n_samples=1000,
        burn_in=500,
        thin=2
    )
    
    print(f"   Samples generated: {len(samples_mh)}")
    print(f"   Acceptance rate: {accept_rate:.2%}")
    print(f"   Sample mean: [{samples_mh[:, 0].mean():.3f}, {samples_mh[:, 1].mean():.3f}] "
          f"(true: [1.0, 2.0])")
    
    # Hamiltonian Monte Carlo
    print("\n2. Hamiltonian Monte Carlo...")
    np.random.seed(42)
    hmc = HamiltonianMonteCarlo(
        target_log_prob=target_log_prob,
        grad_log_prob=grad_log_prob,
        step_size=0.15,
        n_steps=15
    )
    
    samples_hmc, accept_rate_hmc = hmc.sample(
        theta_init=np.array([0.0, 0.0]),
        n_samples=1000,
        burn_in=500
    )
    
    print(f"   Samples generated: {len(samples_hmc)}")
    print(f"   Acceptance rate: {accept_rate_hmc:.2%}")
    print(f"   Sample mean: [{samples_hmc[:, 0].mean():.3f}, {samples_hmc[:, 1].mean():.3f}] "
          f"(true: [1.0, 2.0])")
    
    # Gibbs Sampling
    print("\n3. Gibbs Sampling...")
    np.random.seed(42)
    
    # Conditional distributions for bivariate normal
    rho = 0.7 / np.sqrt(1.0 * 1.0)  # correlation coefficient
    
    def sample_x_given_y(state):
        y = state[1]
        mu_x = mean[0] + rho * (y - mean[1])
        sigma_x = np.sqrt(1.0 - rho**2)
        return np.random.randn() * sigma_x + mu_x
    
    def sample_y_given_x(state):
        x = state[0]
        mu_y = mean[1] + rho * (x - mean[0])
        sigma_y = np.sqrt(1.0 - rho**2)
        return np.random.randn() * sigma_y + mu_y
    
    gibbs = GibbsSampler(conditional_samplers=[sample_x_given_y, sample_y_given_x])
    samples_gibbs = gibbs.sample(
        theta_init=[0.0, 0.0],
        n_samples=1000,
        burn_in=500
    )
    samples_gibbs = np.array(samples_gibbs)
    
    print(f"   Samples generated: {len(samples_gibbs)}")
    print(f"   Sample mean: [{samples_gibbs[:, 0].mean():.3f}, {samples_gibbs[:, 1].mean():.3f}] "
          f"(true: [1.0, 2.0])")
    print()


def main():
    """Run all Bayesian inference examples."""
    print("\n" + "=" * 70)
    print("BAYESIAN INFERENCE EXAMPLES - LITETORCH")
    print("=" * 70)
    print("\nDemonstrating various Bayesian inference methods:")
    print("  1. Bayesian Linear Regression")
    print("  2. Bayesian Neural Networks")
    print("  3. Monte Carlo Dropout")
    print("  4. Variational Inference")
    print("  5. MCMC Sampling")
    print()
    
    try:
        example_bayesian_linear_regression()
        example_bayesian_neural_network()
        example_mc_dropout()
        example_variational_inference()
        example_mcmc()
        
        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
