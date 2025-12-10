"""
mappo_network.py - VERSIÃ“N OPTIMIZADA (BATCHING)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, grid_size=20, feature_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        # Reducción de dimensionalidad para posición (x, y, vx, vy)
        self.pos_embed = nn.Linear(4, 64)
        
        # CNN para el grid
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calcular tamaño de salida de la CNN
        # 20 -> 10 -> 5 -> 2 (aprox dependiendo del padding/pool exacto)
        # Con 3 pools de 2x2 sobre 20x20:
        # P1: 20->10, P2: 10->5, P3: 5->2.
        # Output final: 64 canales * 2 * 2
        conv_output_size = 64 * 2 * 2
        
        self.fc_combine = nn.Linear(64 + conv_output_size, feature_dim)
    
    def process_batch_obs(self, obs_list, agent_key, env):
        """Versión VECTORIZADA y optimizada."""
        device = next(self.parameters()).device
        batch_size = len(obs_list)
        
        # 1. Pre-alocar el grid
        grids = np.zeros((batch_size, 5, self.grid_size, self.grid_size), dtype=np.float32)
        
        # 2. Canales estáticos (Muros y bloques)
        static_channels = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        for block in env.blocks:
            bx, by = int(block.position[0]), int(block.position[1])
            val = 1.5 if block.locked else 1.0
            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                static_channels[0, by, bx] = val
        
        if env.ramp:
            rx, ry = int(env.ramp.position[0]), int(env.ramp.position[1])
            if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                static_channels[1, ry, rx] = 1.0
                
        for wx, wy in env.room.wall_cells:
            if 0 <= wx < self.grid_size and 0 <= wy < self.grid_size:
                static_channels[2, int(wy), int(wx)] = 1.0

        # Copiar canales estáticos a todo el batch
        grids[:, 2:] = static_channels

        # 3. Procesamiento de agentes
        other_key = "seeker" if agent_key == "hider" else "hider"
        
        # Extraer estados con numpy (mucho más rápido que loops)
        all_states = np.array([o[agent_key]["state"] for o in obs_list], dtype=np.float32)
        other_states = np.array([o[other_key]["state"] for o in obs_list], dtype=np.float32)
        
        batch_idx = np.arange(batch_size)
        
        # --- Agente Principal ---
        x, y = all_states[:, 0], all_states[:, 1]
        z = all_states[:, 3]
        ix, iy = x.astype(int), y.astype(int)
        
        mask = (x >= 0) & (ix >= 0) & (ix < self.grid_size) & (iy >= 0) & (iy < self.grid_size)
        if mask.any():
            grids[batch_idx[mask], 0, iy[mask], ix[mask]] = 1.0 + z[mask] * 0.5
            
        # --- Oponente ---
        ox, oy = other_states[:, 0], other_states[:, 1]
        oz = other_states[:, 3]
        iox, ioy = ox.astype(int), oy.astype(int)
        
        mask_o = (ox >= 0) & (iox >= 0) & (iox < self.grid_size) & (ioy >= 0) & (ioy < self.grid_size)
        if mask_o.any():
            grids[batch_idx[mask_o], 1, ioy[mask_o], iox[mask_o]] = 1.0 + oz[mask_o] * 0.5

        return torch.tensor(grids, device=device), torch.tensor(all_states, device=device)
    
    def forward(self, obs_input, agent_key, env):
        # Esta es la función que faltaba
        if isinstance(obs_input, list):
            spatial_grid, state = self.process_batch_obs(obs_input, agent_key, env)
        else:
            spatial_grid, state = self.process_batch_obs([obs_input], agent_key, env)
        
        # Procesar características de posición
        pos_features = F.relu(self.pos_embed(state))
        
        # Procesar Grid con CNN
        x = F.relu(self.conv1(spatial_grid))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Aplanar y combinar
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
        action_probs = self.forward(features)
        dist = Categorical(action_probs)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
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
        if agent_key == "hider":
            return self.hider_feature_extractor(obs, agent_key, env)
        else:
            return self.seeker_feature_extractor(obs, agent_key, env)
    
    def get_action(self, obs, agent_key, env, deterministic=False):
        features = self.get_features(obs, agent_key, env)
        if agent_key == "hider":
            return self.hider_actor.get_action(features, deterministic)
        else:
            return self.seeker_actor.get_action(features, deterministic)
    
    def get_value(self, obs, env):
        hider_features = self.get_features(obs, "hider", env)
        if isinstance(obs, list):
            seeker_active_mask = torch.tensor(
                [o["seeker"]["state"][0] >= 0 for o in obs], 
                device=self.device, dtype=torch.float32
            ).unsqueeze(1)
            seeker_features = self.get_features(obs, "seeker", env)
            seeker_features = seeker_features * seeker_active_mask
        else:
            if obs["seeker"]["state"][0] < 0:
                seeker_features = torch.zeros_like(hider_features).to(self.device)
            else:
                seeker_features = self.get_features(obs, "seeker", env)
        return self.critic(hider_features, seeker_features)
    
    def get_all_parameters(self):
        params = []
        params.extend(list(self.hider_feature_extractor.parameters()))
        params.extend(list(self.seeker_feature_extractor.parameters()))
        params.extend(list(self.hider_actor.parameters()))
        params.extend(list(self.seeker_actor.parameters()))
        params.extend(list(self.critic.parameters()))
        return params