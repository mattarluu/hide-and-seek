"""
mappo_trainer.py

MAPPO training algorithm with:
- PPO clipped objective
- Generalized Advantage Estimation (GAE)
- Centralized critic, decentralized actors
- Mini-batch updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class RolloutBuffer:
    """
    Buffer for storing rollout experience from multiple agents.
    """
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = {"hider": [], "seeker": []}
        self.rewards = {"hider": [], "seeker": []}
        self.values = []
        self.log_probs = {"hider": [], "seeker": []}
        self.dones = []
        self.size = 0
    
    def add(self, obs, actions, rewards, value, log_probs, done):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observation dict
            actions: Dict of actions for each agent
            rewards: Dict of rewards for each agent
            value: Centralized value estimate
            log_probs: Dict of log probabilities for each agent
            done: Episode done flag
        """
        self.observations.append(obs)
        self.actions["hider"].append(actions["hider"])
        self.actions["seeker"].append(actions["seeker"])
        self.rewards["hider"].append(rewards["hider"])
        self.rewards["seeker"].append(rewards["seeker"])
        self.values.append(value)
        self.log_probs["hider"].append(log_probs["hider"])
        self.log_probs["seeker"].append(log_probs["seeker"])
        self.dones.append(done)
        self.size += 1
    
    def get_batch(self):
        """Get all data as batches."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "values": self.values,
            "log_probs": self.log_probs,
            "dones": self.dones
        }
    
    def is_full(self):
        """Check if buffer is full."""
        return self.size >= self.max_size


class MAPPOTrainer:
    """
    MAPPO trainer with PPO updates.
    """
    def __init__(
        self,
        agent,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device='cuda'
    ):
        """
        Initialize MAPPO trainer.
        
        Args:
            agent: MAPPOAgent instance
            env: Environment instance
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Mini-batch size
            device: Device to use
        """
        self.agent = agent
        self.env = env
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Optimizer for all networks
        self.optimizer = optim.Adam(agent.get_all_parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training statistics
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": []
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        # Reverse iterate to compute GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_done = dones[t]
            else:
                next_val = values[t + 1]
                next_done = dones[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self, next_obs):
        """
        Perform PPO update using collected rollouts.
        
        Args:
            next_obs: Next observation for bootstrapping value
        """
        # Get data from buffer
        data = self.buffer.get_batch()
        
        # Compute next value for GAE
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs, self.env).item()
        
        # Compute advantages and returns for each agent
        advantages = {}
        returns = {}
        
        for agent_key in ["hider", "seeker"]:
            # Check if seeker was active
            if agent_key == "seeker" and data["observations"][0]["seeker"]["state"][0] < 0:
                # Seeker not active, use zeros
                advantages[agent_key] = [0.0] * len(data["rewards"][agent_key])
                returns[agent_key] = [0.0] * len(data["rewards"][agent_key])
            else:
                adv, ret = self.compute_gae(
                    data["rewards"][agent_key],
                    data["values"],
                    data["dones"],
                    next_value
                )
                advantages[agent_key] = adv
                returns[agent_key] = ret
        
        # Normalize advantages per agent
        for agent_key in ["hider", "seeker"]:
            adv_array = np.array(advantages[agent_key])
            if adv_array.std() > 1e-8:
                advantages[agent_key] = (adv_array - adv_array.mean()) / (adv_array.std() + 1e-8)
            else:
                advantages[agent_key] = adv_array
        
        # Convert to tensors
        returns_tensor = {
            "hider": torch.FloatTensor(returns["hider"]).to(self.device),
            "seeker": torch.FloatTensor(returns["seeker"]).to(self.device)
        }
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = list(range(len(data["observations"])))
            random.shuffle(indices)
            
            epoch_stats = {
                "policy_loss": [],
                "value_loss": [],
                "entropy": [],
                "total_loss": [],
                "approx_kl": []
            }
            
            for start_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                policy_losses = []
                value_losses = []
                entropies = []
                approx_kls = []
                
                for idx in batch_indices:
                    obs = data["observations"][idx]
                    
                    # Get current policy and value
                    hider_features = self.agent.get_features(obs, "hider", self.env)
                    
                    # Handle inactive seeker
                    if obs["seeker"]["state"][0] < 0:
                        seeker_features = torch.zeros_like(hider_features)
                        seeker_active = False
                    else:
                        seeker_features = self.agent.get_features(obs, "seeker", self.env)
                        seeker_active = True
                    
                    current_value = self.agent.critic(hider_features, seeker_features)
                    
                    # Policy loss for each agent
                    for agent_key in ["hider", "seeker"]:
                        if agent_key == "seeker" and not seeker_active:
                            continue
                        
                        features = hider_features if agent_key == "hider" else seeker_features
                        actor = self.agent.hider_actor if agent_key == "hider" else self.agent.seeker_actor
                        
                        # Current policy
                        action_probs = actor(features)
                        dist = torch.distributions.Categorical(action_probs)
                        
                        action = data["actions"][agent_key][idx]
                        current_log_prob = dist.log_prob(torch.tensor(action).to(self.device))
                        entropy = dist.entropy()
                        
                        # Old log prob
                        old_log_prob = data["log_probs"][agent_key][idx]
                        
                        # Ratio and clipped objective
                        ratio = torch.exp(current_log_prob - old_log_prob)
                        adv = torch.FloatTensor([advantages[agent_key][idx]]).to(self.device)
                        
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                        policy_loss = -torch.min(surr1, surr2)
                        
                        policy_losses.append(policy_loss)
                        entropies.append(entropy)
                        
                        # Approximate KL divergence
                        approx_kl = (old_log_prob - current_log_prob).mean()
                        approx_kls.append(approx_kl.item())
                    
                    # Value loss (shared for both agents)
                    # Use combined return (average of hider and seeker returns)
                    target_return = (returns_tensor["hider"][idx] + returns_tensor["seeker"][idx]) / 2
                    value_loss = nn.MSELoss()(current_value, target_return)
                    value_losses.append(value_loss)
                
                # Compute total loss
                if len(policy_losses) > 0:
                    policy_loss = torch.stack(policy_losses).mean()
                    value_loss = torch.stack(value_losses).mean()
                    entropy = torch.stack(entropies).mean()
                    
                    total_loss = (
                        policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy
                    )
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.get_all_parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    # Track statistics
                    epoch_stats["policy_loss"].append(policy_loss.item())
                    epoch_stats["value_loss"].append(value_loss.item())
                    epoch_stats["entropy"].append(entropy.item())
                    epoch_stats["total_loss"].append(total_loss.item())
                    if len(approx_kls) > 0:
                        epoch_stats["approx_kl"].append(np.mean(approx_kls))
            
            # Average epoch statistics
            for key in epoch_stats:
                if len(epoch_stats[key]) > 0:
                    self.stats[key].append(np.mean(epoch_stats[key]))
        
        # Clear buffer after update
        self.buffer.clear()
        
        return {k: np.mean(v[-self.n_epochs:]) if len(v) > 0 else 0.0 
                for k, v in self.stats.items()}