"""
Benchmark RL algorithms against Stable Baselines 3.

This script compares the performance of reinforcement learning algorithms
between LiteTorch and Stable Baselines 3.
"""
import time
import numpy as np

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("Gymnasium/Gym not available. Install with: pip install gymnasium")

try:
    from stable_baselines3 import DQN, A2C, PPO, SAC, TD3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable Baselines 3 not available. Install with: pip install stable-baselines3")


def benchmark_algorithm(name, litetorch_fn, sb3_fn, timesteps=10000):
    """Benchmark a reinforcement learning algorithm."""
    # Benchmark LiteTorch
    print(f"Training {name} with LiteTorch...")
    start = time.time()
    litetorch_reward = litetorch_fn(timesteps)
    litetorch_time = time.time() - start
    
    # Benchmark Stable Baselines 3
    if SB3_AVAILABLE:
        print(f"Training {name} with Stable Baselines 3...")
        start = time.time()
        sb3_reward = sb3_fn(timesteps)
        sb3_time = time.time() - start
        
        print(f"{name}:")
        print(f"  LiteTorch:")
        print(f"    Time:   {litetorch_time:.2f}s")
        print(f"    Reward: {litetorch_reward:.2f}")
        print(f"  Stable Baselines 3:")
        print(f"    Time:   {sb3_time:.2f}s")
        print(f"    Reward: {sb3_reward:.2f}")
    else:
        print(f"{name}:")
        print(f"  LiteTorch:")
        print(f"    Time:   {litetorch_time:.2f}s")
        print(f"    Reward: {litetorch_reward:.2f}")
    print()


def benchmark_dqn():
    """Benchmark DQN algorithm."""
    # TODO: Implement after DQN is created
    # Example structure:
    # import litetorch as lt
    # 
    # def litetorch_train(timesteps):
    #     env = gym.make("CartPole-v1")
    #     agent = lt.rl.DQN(env)
    #     return agent.train(timesteps)
    # 
    # def sb3_train(timesteps):
    #     env = gym.make("CartPole-v1")
    #     model = DQN("MlpPolicy", env, verbose=0)
    #     model.learn(total_timesteps=timesteps)
    #     # Evaluate
    #     rewards = []
    #     for _ in range(10):
    #         obs, _ = env.reset()
    #         terminated = False
    #         truncated = False
    #         total_reward = 0
    #         while not (terminated or truncated):
    #             action, _ = model.predict(obs)
    #             obs, reward, terminated, truncated, _ = env.step(action)
    #             total_reward += reward
    #         rewards.append(total_reward)
    #     return np.mean(rewards)
    # 
    # benchmark_algorithm("DQN", litetorch_train, sb3_train)
    print("DQN benchmark - TODO: Implement after DQN algorithm")


def benchmark_a2c():
    """Benchmark A2C algorithm."""
    # TODO: Implement after A2C is created
    print("A2C benchmark - TODO: Implement after A2C algorithm")


def benchmark_ppo():
    """Benchmark PPO algorithm."""
    # TODO: Implement after PPO is created
    print("PPO benchmark - TODO: Implement after PPO algorithm")


def benchmark_sac():
    """Benchmark SAC algorithm."""
    # TODO: Implement after SAC is created
    print("SAC benchmark - TODO: Implement after SAC algorithm")


def benchmark_td3():
    """Benchmark TD3 algorithm."""
    # TODO: Implement after TD3 is created
    print("TD3 benchmark - TODO: Implement after TD3 algorithm")


if __name__ == "__main__":
    print("=" * 50)
    print("RL Algorithms Benchmark")
    print("=" * 50)
    print()
    
    if not GYM_AVAILABLE:
        print("Gymnasium/Gym is required for RL benchmarks.")
        print("Please install with: pip install gymnasium")
        exit(1)
    
    benchmark_dqn()
    benchmark_a2c()
    benchmark_ppo()
    benchmark_sac()
    benchmark_td3()
    
    print("Benchmarks complete!")
    print("\nNote: Benchmarks will be fully implemented after RL algorithms are created.")
