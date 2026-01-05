"""
Markov Chain Monte Carlo (MCMC) methods for Bayesian inference.

MCMC methods generate samples from complex probability distributions
that are difficult to sample from directly, enabling Bayesian inference
in models where analytical solutions are intractable.
"""
import numpy as np


class MetropolisHastings:
    """
    Metropolis-Hastings algorithm for MCMC sampling.
    
    The Metropolis-Hastings algorithm is a general MCMC method that constructs
    a Markov chain whose stationary distribution is the target distribution π(θ).
    
    Algorithm:
        1. Start with initial state θ_0
        2. For t = 1, 2, ...:
            a. Propose new state θ* ~ q(θ* | θ_t)
            b. Compute acceptance ratio: α = min(1, [π(θ*) q(θ_t | θ*)] / [π(θ_t) q(θ* | θ_t)])
            c. Accept θ* with probability α:
               - If accepted: θ_{t+1} = θ*
               - If rejected: θ_{t+1} = θ_t
    
    The proposal distribution q(θ* | θ_t) can be:
        - Symmetric (q(θ* | θ_t) = q(θ_t | θ*)) → simplifies to Metropolis algorithm
        - Random walk: θ* = θ_t + ε, where ε ~ N(0, Σ)
        - Independent: q(θ* | θ_t) = q(θ*) (doesn't depend on current state)
    
    Properties:
        - Guarantees convergence to target distribution (under mild conditions)
        - Simple to implement
        - Can be slow to converge for high-dimensional problems
        - Sensitive to proposal distribution choice
    
    References:
        - Metropolis et al. (1953): "Equation of State Calculations by Fast Computing Machines"
        - Hastings (1970): "Monte Carlo Sampling Methods Using Markov Chains"
        - Robert & Casella (2004): "Monte Carlo Statistical Methods"
    """
    
    def __init__(self, target_log_prob, proposal_std=1.0):
        """
        Initialize Metropolis-Hastings sampler.
        
        Args:
            target_log_prob: Function that computes log π(θ) (log target density)
            proposal_std: Standard deviation for Gaussian random walk proposal
        """
        self.target_log_prob = target_log_prob
        self.proposal_std = proposal_std
        
        # Statistics
        self.n_accepted = 0
        self.n_total = 0
    
    def propose(self, theta_current):
        """
        Generate proposal using Gaussian random walk.
        
        θ* = θ_t + ε, where ε ~ N(0, proposal_std^2 * I)
        
        Args:
            theta_current: Current state
            
        Returns:
            theta_proposed: Proposed state
        """
        return theta_current + np.random.randn(*theta_current.shape) * self.proposal_std
    
    def acceptance_prob(self, theta_current, theta_proposed):
        """
        Compute acceptance probability.
        
        For symmetric proposal (random walk):
            α = min(1, π(θ*) / π(θ_t))
        
        Args:
            theta_current: Current state
            theta_proposed: Proposed state
            
        Returns:
            acceptance_prob: Probability of accepting proposal
        """
        # Compute log probabilities
        log_prob_current = self.target_log_prob(theta_current)
        log_prob_proposed = self.target_log_prob(theta_proposed)
        
        # Log acceptance ratio
        log_alpha = log_prob_proposed - log_prob_current
        
        # Acceptance probability (in log space for numerical stability)
        return min(1.0, np.exp(log_alpha))
    
    def sample(self, theta_init, n_samples, burn_in=1000, thin=1):
        """
        Generate samples from target distribution.
        
        Args:
            theta_init: Initial state
            n_samples: Number of samples to generate (after burn-in and thinning)
            burn_in: Number of initial samples to discard
            thin: Keep every thin-th sample (reduces correlation)
            
        Returns:
            samples: Array of samples, shape (n_samples, dim)
            acceptance_rate: Fraction of proposals accepted
        """
        theta_current = np.array(theta_init, dtype=float)
        samples = []
        
        self.n_accepted = 0
        self.n_total = 0
        
        total_iterations = burn_in + n_samples * thin
        
        for i in range(total_iterations):
            # Propose new state
            theta_proposed = self.propose(theta_current)
            
            # Compute acceptance probability
            alpha = self.acceptance_prob(theta_current, theta_proposed)
            
            # Accept/reject
            if np.random.rand() < alpha:
                theta_current = theta_proposed
                self.n_accepted += 1
            
            self.n_total += 1
            
            # Store sample (after burn-in, with thinning)
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(theta_current.copy())
        
        acceptance_rate = self.n_accepted / self.n_total
        
        return np.array(samples), acceptance_rate


class GibbsSampler:
    """
    Gibbs Sampling for multivariate distributions.
    
    Gibbs sampling is a special case of MCMC that samples from conditional
    distributions one variable at a time. It's particularly useful when
    the conditional distributions are easy to sample from, even if the
    joint distribution is complex.
    
    Algorithm:
        For iteration t:
            θ_1^{t+1} ~ p(θ_1 | θ_2^t, θ_3^t, ..., θ_n^t, data)
            θ_2^{t+1} ~ p(θ_2 | θ_1^{t+1}, θ_3^t, ..., θ_n^t, data)
            ...
            θ_n^{t+1} ~ p(θ_n | θ_1^{t+1}, θ_2^{t+1}, ..., θ_{n-1}^{t+1}, data)
    
    Advantages:
        - No tuning parameters (unlike M-H)
        - Always accepts proposals (acceptance rate = 1)
        - Often faster convergence for certain problems
    
    Disadvantages:
        - Requires knowing conditional distributions
        - Can be slow if variables are highly correlated
        - Not applicable to all models
    
    Applications:
        - Bayesian linear regression
        - Topic modeling (LDA)
        - Hierarchical models
        - Image processing (Ising model, etc.)
    
    References:
        - Geman & Geman (1984): "Stochastic Relaxation, Gibbs Distributions"
        - Casella & George (1992): "Explaining the Gibbs Sampler"
    """
    
    def __init__(self, conditional_samplers):
        """
        Initialize Gibbs sampler.
        
        Args:
            conditional_samplers: List of functions, one per variable
                Each function takes current state and returns sample from conditional
        """
        self.conditional_samplers = conditional_samplers
        self.n_vars = len(conditional_samplers)
    
    def sample(self, theta_init, n_samples, burn_in=1000):
        """
        Generate samples using Gibbs sampling.
        
        Args:
            theta_init: Initial state (list or array)
            n_samples: Number of samples to generate
            burn_in: Number of initial samples to discard
            
        Returns:
            samples: Array of samples
        """
        theta_current = list(theta_init)
        samples = []
        
        total_iterations = burn_in + n_samples
        
        for i in range(total_iterations):
            # Update each variable from its conditional distribution
            for j in range(self.n_vars):
                theta_current[j] = self.conditional_samplers[j](theta_current)
            
            # Store sample (after burn-in)
            if i >= burn_in:
                samples.append([x.copy() if isinstance(x, np.ndarray) else x 
                               for x in theta_current])
        
        return samples


class HamiltonianMonteCarlo:
    """
    Hamiltonian Monte Carlo (HMC) sampling.
    
    HMC uses ideas from physics to propose distant states while maintaining
    high acceptance rates. It augments the target distribution with auxiliary
    momentum variables and simulates Hamiltonian dynamics.
    
    Key idea: Treat sampling as simulating a particle moving on a potential
    energy surface, where:
        - Position = parameters θ
        - Potential energy U(θ) = -log π(θ)
        - Kinetic energy K(p) = p^T M^{-1} p / 2
        - Hamiltonian H(θ, p) = U(θ) + K(p)
    
    Algorithm:
        1. Sample momentum p ~ N(0, M)
        2. Simulate Hamiltonian dynamics for L steps using leapfrog integrator:
           p_{t+ε/2} = p_t - (ε/2) ∇U(θ_t)
           θ_{t+ε} = θ_t + ε M^{-1} p_{t+ε/2}
           p_{t+ε} = p_{t+ε/2} - (ε/2) ∇U(θ_{t+ε})
        3. Accept/reject final state using Metropolis criterion
    
    Advantages:
        - Explores parameter space efficiently
        - High acceptance rates (often > 90%)
        - Good for high-dimensional problems
        - Less random walk behavior
    
    Disadvantages:
        - Requires gradient of log probability
        - Needs tuning of step size ε and number of steps L
        - More complex to implement
    
    HMC is the foundation for modern probabilistic programming systems
    like Stan and PyMC3's NUTS sampler.
    
    References:
        - Neal (2011): "MCMC using Hamiltonian dynamics"
        - Betancourt (2017): "A Conceptual Introduction to Hamiltonian Monte Carlo"
    """
    
    def __init__(self, target_log_prob, grad_log_prob, step_size=0.1, n_steps=10):
        """
        Initialize HMC sampler.
        
        Args:
            target_log_prob: Function that computes log π(θ)
            grad_log_prob: Function that computes ∇ log π(θ)
            step_size: Step size ε for leapfrog integrator
            n_steps: Number of leapfrog steps L
        """
        self.target_log_prob = target_log_prob
        self.grad_log_prob = grad_log_prob
        self.step_size = step_size
        self.n_steps = n_steps
        
        self.n_accepted = 0
        self.n_total = 0
    
    def leapfrog(self, theta, momentum):
        """
        Leapfrog integration for Hamiltonian dynamics.
        
        Args:
            theta: Current position
            momentum: Current momentum
            
        Returns:
            theta_new: New position
            momentum_new: New momentum
        """
        # Half step for momentum
        momentum = momentum + 0.5 * self.step_size * self.grad_log_prob(theta)
        
        # Full steps for position and momentum
        for i in range(self.n_steps):
            theta = theta + self.step_size * momentum
            
            if i < self.n_steps - 1:
                momentum = momentum + self.step_size * self.grad_log_prob(theta)
        
        # Half step for momentum at end
        momentum = momentum + 0.5 * self.step_size * self.grad_log_prob(theta)
        
        # Negate momentum for reversibility
        momentum = -momentum
        
        return theta, momentum
    
    def hamiltonian(self, theta, momentum):
        """
        Compute Hamiltonian (total energy).
        
        H(θ, p) = -log π(θ) + 0.5 * p^T p
        
        Args:
            theta: Position
            momentum: Momentum
            
        Returns:
            energy: Hamiltonian value
        """
        potential_energy = -self.target_log_prob(theta)
        kinetic_energy = 0.5 * np.sum(momentum ** 2)
        return potential_energy + kinetic_energy
    
    def sample(self, theta_init, n_samples, burn_in=1000):
        """
        Generate samples using HMC.
        
        Args:
            theta_init: Initial position
            n_samples: Number of samples to generate
            burn_in: Number of initial samples to discard
            
        Returns:
            samples: Array of samples
            acceptance_rate: Fraction of proposals accepted
        """
        theta_current = np.array(theta_init, dtype=float)
        samples = []
        
        self.n_accepted = 0
        self.n_total = 0
        
        total_iterations = burn_in + n_samples
        
        for i in range(total_iterations):
            # Sample momentum
            momentum_current = np.random.randn(*theta_current.shape)
            
            # Compute current Hamiltonian
            H_current = self.hamiltonian(theta_current, momentum_current)
            
            # Leapfrog integration
            theta_proposed, momentum_proposed = self.leapfrog(
                theta_current.copy(), 
                momentum_current.copy()
            )
            
            # Compute proposed Hamiltonian
            H_proposed = self.hamiltonian(theta_proposed, momentum_proposed)
            
            # Accept/reject (Metropolis step)
            log_accept_prob = H_current - H_proposed
            
            if np.log(np.random.rand()) < log_accept_prob:
                theta_current = theta_proposed
                self.n_accepted += 1
            
            self.n_total += 1
            
            # Store sample (after burn-in)
            if i >= burn_in:
                samples.append(theta_current.copy())
        
        acceptance_rate = self.n_accepted / self.n_total
        
        return np.array(samples), acceptance_rate


# Example usage
if __name__ == "__main__":
    print("MCMC Methods Example")
    print("=" * 50)
    
    # Target distribution: Standard normal N(0, 1)
    def target_log_prob(theta):
        return -0.5 * np.sum(theta ** 2)
    
    def grad_log_prob(theta):
        return -theta
    
    # Example 1: Metropolis-Hastings
    print("\n1. Metropolis-Hastings Sampling")
    print("-" * 50)
    
    mh_sampler = MetropolisHastings(
        target_log_prob=target_log_prob,
        proposal_std=1.0
    )
    
    theta_init = np.array([5.0])  # Start far from mode
    samples_mh, accept_rate = mh_sampler.sample(
        theta_init=theta_init,
        n_samples=1000,
        burn_in=500,
        thin=2
    )
    
    print(f"Generated {len(samples_mh)} samples")
    print(f"Acceptance rate: {accept_rate:.2%}")
    print(f"Sample mean: {samples_mh.mean():.4f} (true: 0.0)")
    print(f"Sample std: {samples_mh.std():.4f} (true: 1.0)")
    
    # Example 2: Hamiltonian Monte Carlo
    print("\n2. Hamiltonian Monte Carlo")
    print("-" * 50)
    
    hmc_sampler = HamiltonianMonteCarlo(
        target_log_prob=target_log_prob,
        grad_log_prob=grad_log_prob,
        step_size=0.1,
        n_steps=20
    )
    
    samples_hmc, accept_rate_hmc = hmc_sampler.sample(
        theta_init=theta_init,
        n_samples=1000,
        burn_in=500
    )
    
    print(f"Generated {len(samples_hmc)} samples")
    print(f"Acceptance rate: {accept_rate_hmc:.2%}")
    print(f"Sample mean: {samples_hmc.mean():.4f} (true: 0.0)")
    print(f"Sample std: {samples_hmc.std():.4f} (true: 1.0)")
    
    # Example 3: Gibbs Sampling
    print("\n3. Gibbs Sampling (Bivariate Normal)")
    print("-" * 50)
    
    # Bivariate normal with correlation
    rho = 0.8  # Correlation
    
    def sample_x_given_y(state):
        """Sample x | y from bivariate normal."""
        y = state[1]
        mu = rho * y
        sigma = np.sqrt(1 - rho**2)
        return np.random.randn() * sigma + mu
    
    def sample_y_given_x(state):
        """Sample y | x from bivariate normal."""
        x = state[0]
        mu = rho * x
        sigma = np.sqrt(1 - rho**2)
        return np.random.randn() * sigma + mu
    
    gibbs_sampler = GibbsSampler(
        conditional_samplers=[sample_x_given_y, sample_y_given_x]
    )
    
    samples_gibbs = gibbs_sampler.sample(
        theta_init=[0.0, 0.0],
        n_samples=1000,
        burn_in=500
    )
    
    samples_gibbs = np.array(samples_gibbs)
    print(f"Generated {len(samples_gibbs)} samples")
    print(f"Sample correlation: {np.corrcoef(samples_gibbs.T)[0, 1]:.4f} (true: {rho})")
