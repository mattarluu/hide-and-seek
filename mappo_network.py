"""
mappo_network.py

Neural network architectures for MAPPO in hide and seek.
Separate networks for hider and seeker with shared feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class SpatialFeatureExtractor(nn.Module):
    """
    Extracts spatial features from the environment.
    Processes position, visible cells, and object locations.
    """
    def __init__(self, grid_size=20, feature_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        
        # Spatial embedding for agent position
        self.pos_embed = nn.Linear(4, 64)  # (x, y, direction, z)
        
        # Convolutional layers for spatial awareness
        # Input: 5 channels (self, other_agent, blocks, ramp, walls)
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate conv output size: 20 -> 10 -> 5 -> 2 (after 3 pools)
        conv_output_size = 64 * 2 * 2  # 256
        
        # Combine spatial and positional features
        self.fc_combine = nn.Linear(64 + conv_output_size, feature_dim)
        
    def create_spatial_grid(self, obs, agent_key, env):
        batch_size = 1
        device = self.pos_embed.weight.device

        grid = torch.zeros(batch_size, 5, self.grid_size, self.grid_size, device=device)

        # Channel 0: Current agent position
        x, y, _, z = obs[agent_key]["state"]
        if x >= 0:
            grid[0, 0, int(y), int(x)] = 1.0 + z * 0.5

        # Channel 1: Other agent
        other_key = "seeker" if agent_key == "hider" else "hider"
        ox, oy, _, oz = obs[other_key]["state"]
        if ox >= 0:
            grid[0, 1, int(oy), int(ox)] = 1.0 + oz * 0.5

        # Channel 2: Blocks
        for block in env.blocks:
            bx, by = block.position
            val = 1.5 if block.locked else 1.0
            grid[0, 2, by, bx] = val

        # Channel 3: Ramp
        if env.ramp:
            rx, ry = env.ramp.position
            grid[0, 3, ry, rx] = 1.0

        # Channel 4: Walls
        for wx, wy in env.room.wall_cells:
            grid[0, 4, wy, wx] = 1.0

        return grid
    
    def forward(self, obs, agent_key, env):
        # Position encoding
        device = self.pos_embed.weight.device
        state = torch.as_tensor(obs[agent_key]["state"], dtype=torch.float32, device=device)

        pos_features = F.relu(self.pos_embed(state))

        # Spatial grid encoding
        spatial_grid = self.create_spatial_grid(obs, agent_key, env)

        # Conv layers
        x = F.relu(self.conv1(spatial_grid))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        spatial_features = x.view(x.size(0), -1)

        # Combine
        combined = torch.cat([pos_features.unsqueeze(0), spatial_features], dim=1)
        features = F.relu(self.fc_combine(combined))

        return features.squeeze(0)


class ActorNetwork(nn.Module):
    """
    Actor network for MAPPO.
    Outputs action probabilities.
    """
    def __init__(self, feature_dim=128, action_dim=10, hidden_dim=256):
        super().__init__()
        
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, features):
        """
        Args:
            features: Output from feature extractor
            
        Returns:
            action_probs: Probability distribution over actions
        """
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        
        return F.softmax(action_logits, dim=-1)
    
    def get_action(self, features, deterministic=False):
        """
        Sample an action from the policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            entropy: Entropy of the distribution
        """
        action_probs = self.forward(features)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized critic for MAPPO.
    Takes global state (all agents) and outputs value estimate.
    """
    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()
        
        # Critic sees features from both agents
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, hider_features, seeker_features):
        """
        Args:
            hider_features: Features from hider
            seeker_features: Features from seeker
            
        Returns:
            value: State value estimate
        """
        # Concatenate features from both agents (centralized critic)
        combined = torch.cat([hider_features, seeker_features], dim=-1)
        
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        
        return value.squeeze(-1)


class MAPPOAgent:
    """
    Complete MAPPO agent combining feature extractor, actor, and critic.
    """
    def __init__(self, grid_size=20, action_dim=10, device='cuda'):
        self.device = device
        
        # Networks
        self.hider_feature_extractor = SpatialFeatureExtractor(grid_size).to(device)
        self.seeker_feature_extractor = SpatialFeatureExtractor(grid_size).to(device)
        
        self.hider_actor = ActorNetwork(action_dim=action_dim).to(device)
        self.seeker_actor = ActorNetwork(action_dim=action_dim).to(device)
        
        self.critic = CriticNetwork().to(device)
        
    def get_features(self, obs, agent_key, env):
        if agent_key == "hider":
            feats = self.hider_feature_extractor(obs, agent_key, env)
        else:
            feats = self.seeker_feature_extractor(obs, agent_key, env)

        # Ensure correct device
        return feats.to(self.device)
    
    def get_action(self, obs, agent_key, env, deterministic=False):
        """Get action for a specific agent."""
        features = self.get_features(obs, agent_key, env)
        
        if agent_key == "hider":
            return self.hider_actor.get_action(features, deterministic)
        else:
            return self.seeker_actor.get_action(features, deterministic)
    
    def get_value(self, obs, env):
        """Get centralized value estimate."""
        hider_features = self.get_features(obs, "hider", env)
        
        # Handle inactive seeker
        if obs["seeker"]["state"][0] < 0:
            seeker_features = torch.zeros_like(hider_features)
        else:
            seeker_features = self.get_features(obs, "seeker", env)
        
        return self.critic(hider_features, seeker_features)
    
    def get_all_parameters(self):
        """Get all trainable parameters."""
        params = []
        params.extend(list(self.hider_feature_extractor.parameters()))
        params.extend(list(self.seeker_feature_extractor.parameters()))
        params.extend(list(self.hider_actor.parameters()))
        params.extend(list(self.seeker_actor.parameters()))
        params.extend(list(self.critic.parameters()))
        return params


if __name__ == "__main__":
    # Test the network
    agent = MAPPOAgent()
    print("MAPPO Agent initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in agent.get_all_parameters())}")