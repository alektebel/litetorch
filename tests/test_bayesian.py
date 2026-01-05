"""
Test suite for Bayesian inference models.
"""
import unittest
import numpy as np


class TestBayesianRegression(unittest.TestCase):
    """Test Bayesian Linear Regression models."""
    
    def test_bayesian_linear_regression_initialization(self):
        """Test BayesianLinearRegression initialization."""
        from litetorch.bayesian.bayesian_regression import BayesianLinearRegression
        
        model = BayesianLinearRegression(input_dim=3, alpha=1.0, beta=1.0)
        
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.beta, 1.0)
        self.assertEqual(model.prior_mean.shape, (3,))
        self.assertEqual(model.prior_cov.shape, (3, 3))
        self.assertFalse(model.is_fitted)
    
    def test_bayesian_linear_regression_fit(self):
        """Test BayesianLinearRegression fitting."""
        from litetorch.bayesian.bayesian_regression import BayesianLinearRegression
        
        # Simple 1D regression problem
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2*x
        
        model = BayesianLinearRegression(input_dim=1, alpha=1.0, beta=1.0)
        model.fit(X, y)
        
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.posterior_mean.shape, (1,))
        self.assertEqual(model.posterior_cov.shape, (1, 1))
        
        # Check that posterior mean is close to true weight (2.0)
        self.assertAlmostEqual(model.posterior_mean[0], 2.0, delta=1.0)
    
    def test_bayesian_linear_regression_predict(self):
        """Test BayesianLinearRegression prediction."""
        from litetorch.bayesian.bayesian_regression import BayesianLinearRegression
        
        np.random.seed(42)
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([2.0, 4.0, 6.0])
        
        model = BayesianLinearRegression(input_dim=1)
        model.fit(X_train, y_train)
        
        X_test = np.array([[4.0], [5.0]])
        predictions = model.predict(X_test)
        
        self.assertEqual(predictions.shape, (2,))
    
    def test_bayesian_linear_regression_predict_with_uncertainty(self):
        """Test BayesianLinearRegression prediction with uncertainty."""
        from litetorch.bayesian.bayesian_regression import BayesianLinearRegression
        
        np.random.seed(42)
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([2.0, 4.0, 6.0])
        
        model = BayesianLinearRegression(input_dim=1)
        model.fit(X_train, y_train)
        
        X_test = np.array([[4.0], [5.0]])
        predictions, std = model.predict(X_test, return_std=True)
        
        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(np.all(std > 0))  # Uncertainty should be positive
    
    def test_bayesian_linear_regression_sample_weights(self):
        """Test weight sampling from posterior."""
        from litetorch.bayesian.bayesian_regression import BayesianLinearRegression
        
        model = BayesianLinearRegression(input_dim=2)
        samples = model.sample_weights(n_samples=10)
        
        self.assertEqual(samples.shape, (10, 2))
    
    def test_bayesian_ridge_initialization(self):
        """Test BayesianRidge initialization."""
        from litetorch.bayesian.bayesian_regression import BayesianRidge
        
        model = BayesianRidge(input_dim=5, max_iter=300, tol=1e-3)
        
        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.max_iter, 300)
        self.assertEqual(model.tol, 1e-3)
        self.assertFalse(model.is_fitted)
    
    def test_bayesian_ridge_fit(self):
        """Test BayesianRidge fitting with evidence maximization."""
        from litetorch.bayesian.bayesian_regression import BayesianRidge
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.1
        
        model = BayesianRidge(input_dim=3)
        model.fit(X, y)
        
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.posterior_mean)
        self.assertIsNotNone(model.posterior_cov)


class TestBayesianNN(unittest.TestCase):
    """Test Bayesian Neural Network models."""
    
    def test_bayesian_nn_initialization(self):
        """Test BayesianNeuralNetwork initialization."""
        from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork
        
        layer_sizes = [5, 10, 2]
        bnn = BayesianNeuralNetwork(layer_sizes=layer_sizes, prior_std=1.0)
        
        self.assertEqual(bnn.layer_sizes, layer_sizes)
        self.assertEqual(bnn.n_layers, 2)
        self.assertEqual(bnn.prior_std, 1.0)
        self.assertEqual(len(bnn.weight_means), 2)
        self.assertEqual(len(bnn.bias_means), 2)
    
    def test_bayesian_nn_sample_weights(self):
        """Test weight sampling from variational distribution."""
        from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork
        
        bnn = BayesianNeuralNetwork(layer_sizes=[3, 5, 2])
        weights = bnn.sample_weights()
        
        self.assertEqual(len(weights), 2)  # 2 layers
        self.assertEqual(weights[0][0].shape, (3, 5))  # First layer weights
        self.assertEqual(weights[0][1].shape, (5,))    # First layer biases
        self.assertEqual(weights[1][0].shape, (5, 2))  # Second layer weights
        self.assertEqual(weights[1][1].shape, (2,))    # Second layer biases
    
    def test_bayesian_nn_forward(self):
        """Test forward pass through BNN."""
        from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork
        
        np.random.seed(42)
        bnn = BayesianNeuralNetwork(layer_sizes=[2, 4, 1])
        
        X = np.random.randn(10, 2)
        weights = bnn.sample_weights()
        output = bnn.forward(X, weights)
        
        self.assertEqual(output.shape, (10, 1))
    
    def test_bayesian_nn_kl_divergence(self):
        """Test KL divergence computation."""
        from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork
        
        bnn = BayesianNeuralNetwork(layer_sizes=[2, 3, 1])
        kl = bnn.kl_divergence()
        
        self.assertIsInstance(kl, float)
        self.assertGreaterEqual(kl, 0.0)  # KL divergence is non-negative
    
    def test_bayesian_nn_predict(self):
        """Test BNN prediction with uncertainty."""
        from litetorch.bayesian.bayesian_nn import BayesianNeuralNetwork
        
        np.random.seed(42)
        bnn = BayesianNeuralNetwork(layer_sizes=[2, 5, 1])
        
        X = np.random.randn(5, 2)
        mean, std = bnn.predict(X, n_samples=10)
        
        self.assertEqual(mean.shape, (5, 1))
        self.assertEqual(std.shape, (5, 1))
        self.assertTrue(np.all(std >= 0))  # Uncertainty should be non-negative
    
    def test_mc_dropout_bnn_initialization(self):
        """Test MCDropoutBNN initialization."""
        from litetorch.bayesian.bayesian_nn import MCDropoutBNN
        
        layer_sizes = [10, 20, 5]
        mc_bnn = MCDropoutBNN(layer_sizes=layer_sizes, dropout_rate=0.5)
        
        self.assertEqual(mc_bnn.layer_sizes, layer_sizes)
        self.assertEqual(mc_bnn.dropout_rate, 0.5)
        self.assertEqual(mc_bnn.n_layers, 2)
    
    def test_mc_dropout_forward(self):
        """Test MC Dropout forward pass."""
        from litetorch.bayesian.bayesian_nn import MCDropoutBNN
        
        np.random.seed(42)
        mc_bnn = MCDropoutBNN(layer_sizes=[3, 5, 2], dropout_rate=0.3)
        
        X = np.random.randn(10, 3)
        
        # Forward without dropout
        output_no_dropout = mc_bnn.forward(X, training=False)
        self.assertEqual(output_no_dropout.shape, (10, 2))
        
        # Forward with dropout
        output_with_dropout = mc_bnn.forward(X, training=True)
        self.assertEqual(output_with_dropout.shape, (10, 2))
    
    def test_mc_dropout_predict_with_uncertainty(self):
        """Test MC Dropout prediction with uncertainty."""
        from litetorch.bayesian.bayesian_nn import MCDropoutBNN
        
        np.random.seed(42)
        mc_bnn = MCDropoutBNN(layer_sizes=[2, 8, 1], dropout_rate=0.5)
        
        X = np.random.randn(5, 2)
        mean, std = mc_bnn.predict_with_uncertainty(X, n_samples=50)
        
        self.assertEqual(mean.shape, (5, 1))
        self.assertEqual(std.shape, (5, 1))
        self.assertTrue(np.all(std >= 0))
    
    def test_variational_dense_layer(self):
        """Test VariationalDense layer."""
        from litetorch.bayesian.bayesian_nn import VariationalDense
        
        layer = VariationalDense(in_features=5, out_features=3, prior_std=1.0)
        
        self.assertEqual(layer.in_features, 5)
        self.assertEqual(layer.out_features, 3)
        self.assertEqual(layer.weight_mu.shape, (5, 3))
        self.assertEqual(layer.bias_mu.shape, (3,))
        
        # Test forward pass
        np.random.seed(42)
        X = np.random.randn(10, 5)
        output = layer.forward(X)
        self.assertEqual(output.shape, (10, 3))
        
        # Test KL divergence
        kl = layer.kl_divergence()
        self.assertIsInstance(kl, float)
        self.assertGreaterEqual(kl, 0.0)


class TestVariationalInference(unittest.TestCase):
    """Test Variational Inference methods."""
    
    def test_mean_field_vi_initialization(self):
        """Test MeanFieldVariationalInference initialization."""
        from litetorch.bayesian.variational_inference import MeanFieldVariationalInference
        
        mf_vi = MeanFieldVariationalInference(n_components=5)
        
        self.assertEqual(mf_vi.n_components, 5)
        self.assertEqual(mf_vi.q_means.shape, (5,))
        self.assertEqual(mf_vi.q_log_stds.shape, (5,))
    
    def test_black_box_vi_initialization(self):
        """Test BlackBoxVariationalInference initialization."""
        from litetorch.bayesian.variational_inference import BlackBoxVariationalInference
        
        bb_vi = BlackBoxVariationalInference(latent_dim=10, learning_rate=0.001)
        
        self.assertEqual(bb_vi.latent_dim, 10)
        self.assertEqual(bb_vi.learning_rate, 0.001)
        self.assertEqual(bb_vi.q_mean.shape, (10,))
    
    def test_black_box_vi_sample_latent(self):
        """Test sampling from variational distribution."""
        from litetorch.bayesian.variational_inference import BlackBoxVariationalInference
        
        np.random.seed(42)
        bb_vi = BlackBoxVariationalInference(latent_dim=5)
        
        samples = bb_vi.sample_latent(n_samples=10)
        
        self.assertEqual(samples.shape, (10, 5))
    
    def test_stochastic_vi_initialization(self):
        """Test StochasticVariationalInference initialization."""
        from litetorch.bayesian.variational_inference import StochasticVariationalInference
        
        svi = StochasticVariationalInference(batch_size=100, learning_rate=0.01)
        
        self.assertEqual(svi.batch_size, 100)
        self.assertEqual(svi.learning_rate, 0.01)
        self.assertEqual(svi.iteration, 0)
    
    def test_stochastic_vi_learning_rate_decay(self):
        """Test learning rate decay schedule."""
        from litetorch.bayesian.variational_inference import StochasticVariationalInference
        
        svi = StochasticVariationalInference()
        
        # Test learning rate decay
        rates = []
        for i in range(5):
            svi.iteration = i * 100
            rates.append(svi.get_learning_rate())
        
        # Learning rate should decrease over iterations
        self.assertTrue(rates[0] > rates[-1])
    
    def test_vae_initialization(self):
        """Test VAE initialization."""
        from litetorch.bayesian.variational_inference import AutoEncodingVariationalBayes
        
        vae = AutoEncodingVariationalBayes(
            input_dim=784,
            latent_dim=20,
            hidden_dim=400
        )
        
        self.assertEqual(vae.input_dim, 784)
        self.assertEqual(vae.latent_dim, 20)
        self.assertEqual(vae.hidden_dim, 400)
        self.assertIsNotNone(vae.encoder_weights)
        self.assertIsNotNone(vae.decoder_weights)
    
    def test_vae_reparameterize(self):
        """Test VAE reparameterization trick."""
        from litetorch.bayesian.variational_inference import AutoEncodingVariationalBayes
        
        np.random.seed(42)
        vae = AutoEncodingVariationalBayes(input_dim=10, latent_dim=5)
        
        mu = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        log_var = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        
        z = vae.reparameterize(mu, log_var)
        
        self.assertEqual(z.shape, (1, 5))


class TestMCMC(unittest.TestCase):
    """Test MCMC sampling methods."""
    
    def test_metropolis_hastings_initialization(self):
        """Test MetropolisHastings initialization."""
        from litetorch.bayesian.mcmc import MetropolisHastings
        
        def target_log_prob(theta):
            return -0.5 * np.sum(theta ** 2)
        
        mh = MetropolisHastings(target_log_prob=target_log_prob, proposal_std=1.0)
        
        self.assertEqual(mh.proposal_std, 1.0)
        self.assertEqual(mh.n_accepted, 0)
        self.assertEqual(mh.n_total, 0)
    
    def test_metropolis_hastings_propose(self):
        """Test MH proposal generation."""
        from litetorch.bayesian.mcmc import MetropolisHastings
        
        def target_log_prob(theta):
            return -0.5 * np.sum(theta ** 2)
        
        np.random.seed(42)
        mh = MetropolisHastings(target_log_prob=target_log_prob, proposal_std=0.5)
        
        theta_current = np.array([1.0, 2.0])
        theta_proposed = mh.propose(theta_current)
        
        self.assertEqual(theta_proposed.shape, theta_current.shape)
        # Proposal should be different from current (with high probability)
        self.assertFalse(np.allclose(theta_current, theta_proposed))
    
    def test_metropolis_hastings_sample(self):
        """Test MH sampling."""
        from litetorch.bayesian.mcmc import MetropolisHastings
        
        # Target: standard normal
        def target_log_prob(theta):
            return -0.5 * np.sum(theta ** 2)
        
        np.random.seed(42)
        mh = MetropolisHastings(target_log_prob=target_log_prob, proposal_std=1.0)
        
        theta_init = np.array([0.0])
        samples, accept_rate = mh.sample(
            theta_init=theta_init,
            n_samples=100,
            burn_in=50,
            thin=1
        )
        
        self.assertEqual(samples.shape, (100, 1))
        self.assertTrue(0.0 <= accept_rate <= 1.0)
    
    def test_gibbs_sampler_initialization(self):
        """Test GibbsSampler initialization."""
        from litetorch.bayesian.mcmc import GibbsSampler
        
        def sample_x(state):
            return np.random.randn()
        
        def sample_y(state):
            return np.random.randn()
        
        gibbs = GibbsSampler(conditional_samplers=[sample_x, sample_y])
        
        self.assertEqual(gibbs.n_vars, 2)
    
    def test_gibbs_sampler_sample(self):
        """Test Gibbs sampling."""
        from litetorch.bayesian.mcmc import GibbsSampler
        
        np.random.seed(42)
        
        def sample_x(state):
            return np.random.randn()
        
        def sample_y(state):
            return np.random.randn()
        
        gibbs = GibbsSampler(conditional_samplers=[sample_x, sample_y])
        
        samples = gibbs.sample(
            theta_init=[0.0, 0.0],
            n_samples=50,
            burn_in=20
        )
        
        self.assertEqual(len(samples), 50)
    
    def test_hmc_initialization(self):
        """Test HamiltonianMonteCarlo initialization."""
        from litetorch.bayesian.mcmc import HamiltonianMonteCarlo
        
        def target_log_prob(theta):
            return -0.5 * np.sum(theta ** 2)
        
        def grad_log_prob(theta):
            return -theta
        
        hmc = HamiltonianMonteCarlo(
            target_log_prob=target_log_prob,
            grad_log_prob=grad_log_prob,
            step_size=0.1,
            n_steps=10
        )
        
        self.assertEqual(hmc.step_size, 0.1)
        self.assertEqual(hmc.n_steps, 10)
    
    def test_hmc_hamiltonian(self):
        """Test Hamiltonian computation."""
        from litetorch.bayesian.mcmc import HamiltonianMonteCarlo
        
        def target_log_prob(theta):
            return -0.5 * np.sum(theta ** 2)
        
        def grad_log_prob(theta):
            return -theta
        
        hmc = HamiltonianMonteCarlo(
            target_log_prob=target_log_prob,
            grad_log_prob=grad_log_prob
        )
        
        theta = np.array([0.0])
        momentum = np.array([0.0])
        
        H = hmc.hamiltonian(theta, momentum)
        
        self.assertIsInstance(H, (float, np.floating))
    
    def test_hmc_sample(self):
        """Test HMC sampling."""
        from litetorch.bayesian.mcmc import HamiltonianMonteCarlo
        
        # Target: standard normal
        def target_log_prob(theta):
            return -0.5 * np.sum(theta ** 2)
        
        def grad_log_prob(theta):
            return -theta
        
        np.random.seed(42)
        hmc = HamiltonianMonteCarlo(
            target_log_prob=target_log_prob,
            grad_log_prob=grad_log_prob,
            step_size=0.15,
            n_steps=20
        )
        
        theta_init = np.array([0.0])
        samples, accept_rate = hmc.sample(
            theta_init=theta_init,
            n_samples=50,
            burn_in=20
        )
        
        self.assertEqual(samples.shape, (50, 1))
        self.assertTrue(0.0 <= accept_rate <= 1.0)


if __name__ == '__main__':
    unittest.main()
