"""
mappo_trainer.py
MAPPO training algorithm optimized for batched processing (Vectorized).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class RolloutBuffer:
    """Buffer for storing rollout experience from multiple agents."""
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
        """Add a transition to the buffer."""
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

class MAPPOTrainer:
    """MAPPO trainer with PPO updates (Vectorized)."""
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
        
        self.optimizer = optim.Adam(agent.get_all_parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        
        self.stats = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "total_loss": [], "approx_kl": []
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_done = dones[t]
            else:
                next_val = values[t + 1]
                next_done = dones[t]
            
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, next_obs):
        """Perform PPO update using collected rollouts (Fully Vectorized)."""
        data = self.buffer.get_batch()
        
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs, self.env).item()
        
        # 1. Compute GAE
        advantages = {}
        returns = {}
        
        for agent_key in ["hider", "seeker"]:
            # Basic check for seeker existence in first frame (heuristic)
            if agent_key == "seeker" and data["observations"][0]["seeker"]["state"][0] < 0:
                advantages[agent_key] = [0.0] * len(data["rewards"][agent_key])
                returns[agent_key] = [0.0] * len(data["rewards"][agent_key])
            else:
                adv, ret = self.compute_gae(
                    data["rewards"][agent_key], data["values"], data["dones"], next_value
                )
                advantages[agent_key] = adv
                returns[agent_key] = ret
        
        # Normalize advantages
        for agent_key in ["hider", "seeker"]:
            adv_array = np.array(advantages[agent_key])
            if adv_array.std() > 1e-8:
                advantages[agent_key] = (adv_array - adv_array.mean()) / (adv_array.std() + 1e-8)
            else:
                advantages[agent_key] = adv_array
        
        # Prepare full batch tensors
        returns_tensor = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device) 
            for k, v in returns.items()
        }
        advantages_tensor = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device) 
            for k, v in advantages.items()
        }
        
        old_log_probs_tensor = {}
        for k, v in data["log_probs"].items():
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                old_log_probs_tensor[k] = torch.stack([lp.to(self.device) for lp in v]).detach()
            else:
                old_log_probs_tensor[k] = torch.tensor(v, device=self.device)

        # 2. Optimization Loop (Batched)
        indices = list(range(len(data["observations"])))
        
        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            
            epoch_stats = {
                "policy_loss": [], "value_loss": [], "entropy": [],
                "total_loss": [], "approx_kl": []
            }
            
            for start_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # --- Prepare Batch Data ---
                # Slicing lists is fast enough; processing them in network must be batched
                batch_obs = [data["observations"][i] for i in batch_indices]
                
                b_act_hider = torch.tensor([data["actions"]["hider"][i] for i in batch_indices], device=self.device)
                b_act_seeker = torch.tensor([data["actions"]["seeker"][i] for i in batch_indices], device=self.device)
                
                # Get slices of tensors
                # Convert batch_indices to tensor for indexing if needed, or simple list slice if contiguous 
                # (but shuffling makes it non-contiguous, so list indexing works for tensors in PyTorch)
                b_ret_hider = returns_tensor["hider"][batch_indices]
                b_ret_seeker = returns_tensor["seeker"][batch_indices]
                b_adv_hider = advantages_tensor["hider"][batch_indices]
                b_adv_seeker = advantages_tensor["seeker"][batch_indices]
                b_old_lp_hider = old_log_probs_tensor["hider"][batch_indices]
                b_old_lp_seeker = old_log_probs_tensor["seeker"][batch_indices]

                # --- Forward Passes (VECTORIZED) ---
                # NOTE: feature_extractor now handles list of dicts -> Batch Tensor
                h_features = self.agent.hider_feature_extractor(batch_obs, "hider", self.env)
                s_features = self.agent.seeker_feature_extractor(batch_obs, "seeker", self.env)
                
                # Mask inactive seekers
                s_active = torch.tensor(
                    [o["seeker"]["state"][0] >= 0 for o in batch_obs], 
                    device=self.device, dtype=torch.bool
                )
                # Zero out features for dead seekers
                s_features = s_features * s_active.unsqueeze(1).float()

                # Critic
                current_values = self.agent.critic(h_features, s_features).squeeze()
                target_values = (b_ret_hider + b_ret_seeker) / 2
                value_loss = (current_values - target_values).pow(2).mean()
                
                # --- Actor: Hider ---
                h_probs = self.agent.hider_actor(h_features)
                h_dist = torch.distributions.Categorical(h_probs)
                h_log_probs = h_dist.log_prob(b_act_hider)
                h_entropy = h_dist.entropy().mean()
                
                ratio_h = torch.exp(h_log_probs - b_old_lp_hider)
                surr1_h = ratio_h * b_adv_hider
                surr2_h = torch.clamp(ratio_h, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv_hider
                policy_loss_h = -torch.min(surr1_h, surr2_h).mean()
                
                approx_kl_h = (b_old_lp_hider - h_log_probs).mean().item()
                
                # --- Actor: Seeker ---
                s_probs = self.agent.seeker_actor(s_features)
                s_dist = torch.distributions.Categorical(s_probs)
                s_log_probs = s_dist.log_prob(b_act_seeker)
                s_entropy = s_dist.entropy()
                
                ratio_s = torch.exp(s_log_probs - b_old_lp_seeker)
                surr1_s = ratio_s * b_adv_seeker
                surr2_s = torch.clamp(ratio_s, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv_seeker
                policy_loss_s_raw = -torch.min(surr1_s, surr2_s)
                
                # Only apply seeker loss if active
                if s_active.any():
                    policy_loss_s = (policy_loss_s_raw * s_active.float()).sum() / (s_active.sum() + 1e-8)
                    s_entropy_mean = (s_entropy * s_active.float()).sum() / (s_active.sum() + 1e-8)
                    approx_kl_s = ((b_old_lp_seeker - s_log_probs) * s_active.float()).sum() / (s_active.sum() + 1e-8)
                    approx_kl_s = approx_kl_s.item()
                else:
                    policy_loss_s = torch.tensor(0.0, device=self.device)
                    s_entropy_mean = torch.tensor(0.0, device=self.device)
                    approx_kl_s = 0.0

                # --- Total Loss & Update ---
                policy_loss = policy_loss_h + policy_loss_s
                entropy = h_entropy + s_entropy_mean
                
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.get_all_parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Stats
                epoch_stats["policy_loss"].append(policy_loss.item())
                epoch_stats["value_loss"].append(value_loss.item())
                epoch_stats["entropy"].append(entropy.item())
                epoch_stats["total_loss"].append(total_loss.item())
                epoch_stats["approx_kl"].append((approx_kl_h + approx_kl_s) / 2)

            # Average epoch stats
            for key in epoch_stats:
                if len(epoch_stats[key]) > 0:
                    self.stats[key].append(np.mean(epoch_stats[key]))
        
        self.buffer.clear()
        return {k: np.mean(v[-self.n_epochs:]) if len(v) > 0 else 0.0 for k, v in self.stats.items()}