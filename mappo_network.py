"""
mappo_network.py

Neural network architectures for MAPPO in hide and seek.
Optimized for batch processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class SpatialFeatureExtractor(nn.Module):
    """
    Extracts spatial features from the environment.
    Optimized: Caches static maps and uses vectorized operations.
    """
    def __init__(self, grid_size=20, feature_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        
        # Cache for static elements (walls, ramps)
        self.static_grid_cache = None
        
        self.pos_embed = nn.Linear(4, 64)
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 20 -> 10 -> 5 -> 2
        conv_output_size = 64 * 2 * 2
        self.fc_combine = nn.Linear(64 + conv_output_size, feature_dim)

    def _build_static_cache(self, env, device):
        """Builds the wall and ramp channels once."""
        cache = torch.zeros(5, self.grid_size, self.grid_size, device=device)
        
        # Channel 3: Ramp
        if env.ramp:
            rx, ry = env.ramp.position
            cache[3, ry, rx] = 1.0
            
        # Channel 4: Walls
        # Usamos listas de python para extraer coordenadas y asignarlas de golpe
        if env.room.wall_cells:
            wx = [w[0] for w in env.room.wall_cells]
            wy = [w[1] for w in env.room.wall_cells]
            cache[4, wy, wx] = 1.0
            
        self.static_grid_cache = cache

    def process_batch(self, obs_list, agent_key, env):
        """
        Process a list of observations into a batch tensor efficiently.
        HIGH PERFORMANCE VERSION.
        """
        device = next(self.parameters()).device
        batch_size = len(obs_list)
        
        # 1. Initialize Cache if needed
        if self.static_grid_cache is None:
            self._build_static_cache(env, device)
            
        # 2. Fast State Extraction (numpy -> tensor is faster than loop of tensors)
        # Extract states into a numpy array first to avoid loop overhead
        raw_states = np.array([o[agent_key]["state"] for o in obs_list], dtype=np.float32)
        states = torch.tensor(raw_states, device=device)

        # 3. Clone static grid (Walls/Ramps are already there)
        # Expansion: (1, 5, H, W) -> (B, 5, H, W) without copying memory excessively
        grids = self.static_grid_cache.unsqueeze(0).repeat(batch_size, 1, 1, 1).clone()
        
        # 4. Vectorized Scatter for Agents (No loops!)
        batch_idx = torch.arange(batch_size, device=device)
        
        # Current Agent (Channel 0)
        curr_x = states[:, 0].long()
        curr_y = states[:, 1].long()
        curr_z = states[:, 3]
        valid_curr = curr_x >= 0
        
        if valid_curr.any():
            # Vectorized assignment
            grids[batch_idx[valid_curr], 0, curr_y[valid_curr], curr_x[valid_curr]] = 1.0 + curr_z[valid_curr] * 0.5
            
        # Other Agent (Channel 1)
        other_key = "seeker" if agent_key == "hider" else "hider"
        other_raw = np.array([o[other_key]["state"] for o in obs_list], dtype=np.float32)
        other_states = torch.tensor(other_raw, device=device)
        
        other_x = other_states[:, 0].long()
        other_y = other_states[:, 1].long()
        other_z = other_states[:, 3]
        valid_other = other_x >= 0
        
        if valid_other.any():
            grids[batch_idx[valid_other], 1, other_y[valid_other], other_x[valid_other]] = 1.0 + other_z[valid_other] * 0.5

        # 5. Blocks (Channel 2) - Still needs a loop as block count varies, but minimized
        # If blocks are few (1-3), this loop is negligible compared to walls
        for i, obs in enumerate(obs_list):
            # Assuming logic from original: pulling from env.blocks is risky if parallel envs diverge.
            # Ideally obs should contain block info. Assuming env.blocks matches for now.
            for block in env.blocks:
                bx, by = block.position
                val = 1.5 if block.locked else 1.0
                grids[i, 2, by, bx] = val
        
        return states, grids
    
    def forward(self, inputs):
        state_tensor, grid_tensor = inputs
        pos_features = F.relu(self.pos_embed(state_tensor))
        
        x = F.relu(self.conv1(grid_tensor))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        spatial_features = x.view(x.size(0), -1)
        combined = torch.cat([pos_features, spatial_features], dim=1)
        features = F.relu(self.fc_combine(combined))
        return features


class ActorNetwork(nn.Module):
    def __init__(self, feature_dim=128, action_dim=10, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        return F.softmax(action_logits, dim=-1)
    
    def get_action(self, features, deterministic=False):
        # Handle batch or single
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        action_probs = self.forward(features)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # If single input, return scalars
        if features.size(0) == 1:
            return action.item(), log_prob, entropy
        
        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, hider_features, seeker_features):
        combined = torch.cat([hider_features, seeker_features], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value.squeeze(-1)


class MAPPOAgent:
    def __init__(self, grid_size=20, action_dim=10, device='cuda'):
        self.device = device
        self.hider_feature_extractor = SpatialFeatureExtractor(grid_size).to(device)
        self.seeker_feature_extractor = SpatialFeatureExtractor(grid_size).to(device)
        self.hider_actor = ActorNetwork(action_dim=action_dim).to(device)
        self.seeker_actor = ActorNetwork(action_dim=action_dim).to(device)
        self.critic = CriticNetwork().to(device)
        
    def get_features(self, obs, agent_key, env):
        # Legacy/Single support
        extractor = self.hider_feature_extractor if agent_key == "hider" else self.seeker_feature_extractor
        inputs = extractor.process_batch([obs], agent_key, env)
        return extractor(inputs) # Returns batch 1
    
    def get_action(self, obs, agent_key, env, deterministic=False):
        features = self.get_features(obs, agent_key, env)
        actor = self.hider_actor if agent_key == "hider" else self.seeker_actor
        return actor.get_action(features.squeeze(0), deterministic)
    
    def get_value(self, obs, env):
        hider_feat = self.get_features(obs, "hider", env)
        if obs["seeker"]["state"][0] < 0:
            seeker_feat = torch.zeros_like(hider_feat).to(self.device)
        else:
            seeker_feat = self.get_features(obs, "seeker", env)
        return self.critic(hider_feat, seeker_feat)

    def get_all_parameters(self):
        params = []
        params.extend(list(self.hider_feature_extractor.parameters()))
        params.extend(list(self.seeker_feature_extractor.parameters()))
        params.extend(list(self.hider_actor.parameters()))
        params.extend(list(self.seeker_actor.parameters()))
        params.extend(list(self.critic.parameters()))
        return params