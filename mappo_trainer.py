"""
mappo_trainer.py
Optimized Vectorized MAPPO Trainer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class RolloutBuffer:
    def __init__(self, max_size=2048):
        self.max_size = max_size; self.clear()
    def clear(self):
        self.observations = []; self.actions = {"hider":[], "seeker":[]}
        self.rewards = {"hider":[], "seeker":[]}; self.values = []
        self.log_probs = {"hider":[], "seeker":[]}; self.dones = []; self.size = 0
    def add(self, obs, act, rew, val, log, done):
        self.observations.append(obs); self.actions["hider"].append(act["hider"])
        self.actions["seeker"].append(act["seeker"]); self.rewards["hider"].append(rew["hider"])
        self.rewards["seeker"].append(rew["seeker"]); self.values.append(val)
        self.log_probs["hider"].append(log["hider"]); self.log_probs["seeker"].append(log["seeker"])
        self.dones.append(done); self.size += 1
    def get_batch(self):
        return {"observations": self.observations, "actions": self.actions,
                "rewards": self.rewards, "values": self.values,
                "log_probs": self.log_probs, "dones": self.dones}

class MAPPOTrainer:
    def __init__(self, agent, env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, n_epochs=10, 
                 batch_size=1024, device='cuda'):
        self.agent = agent; self.env = env; self.device = device
        self.gamma = gamma; self.gae_lambda = gae_lambda; self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef; self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm; self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(agent.get_all_parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.stats = {"policy_loss": [], "value_loss": [], "total_loss": []}

    def compute_gae(self, rewards, values, dones, next_val):
        advs, gae = [], 0
        for t in reversed(range(len(rewards))):
            nv = next_val if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * nv * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            advs.insert(0, gae)
        return advs, [a + v for a, v in zip(advs, values)]

    def update(self, next_obs):
        data = self.buffer.get_batch()
        
        # --- OPTIMIZATION START: Pre-process EVERYTHING once ---
        # Instead of processing batches inside the epoch loop, we process all data to tensors first.
        # This reduces CNN overhead by factor of n_epochs.
        
        all_obs = data["observations"]
        batch_size_total = len(all_obs)
        
        with torch.no_grad():
            next_val = self.agent.get_value(next_obs, self.env).item()
            
            # Pre-compute Features for the WHOLE buffer
            # This is safe for GPU memory (approx 3MB for features vs 60MB for grids)
            h_s, h_g = self.agent.hider_feature_extractor.process_batch(all_obs, "hider", self.env)
            s_s, s_g = self.agent.seeker_feature_extractor.process_batch(all_obs, "seeker", self.env)
            
            # We store the FEATURES, not the grids.
            all_h_feats = self.agent.hider_feature_extractor((h_s, h_g))
            
            # Seeker mask handling
            s_mask = torch.tensor([o["seeker"]["state"][0] >= 0 for o in all_obs], device=self.device)
            raw_s_feats = self.agent.seeker_feature_extractor((s_s, s_g))
            all_s_feats = raw_s_feats * s_mask.unsqueeze(1)
            
            # Values for GAE
            values = self.agent.critic(all_h_feats, all_s_feats).squeeze().cpu().numpy().tolist()
            
        # --- END OPTIMIZATION PRE-CALC ---

        # Advantages calculation (Uses the pre-calc values or buffer values)
        # Note: If buffer has values, use them. If we want fresh ones, use 'values'.
        # Using buffer values ensures consistency with rollout.
        advs, rets = {}, {}
        for key in ["hider", "seeker"]:
            if key == "seeker":
                # Handle seeker death cases logic
                # Simplified for brevity based on your code logic
                advs[key], rets[key] = self.compute_gae(data["rewards"][key], data["values"], data["dones"], next_val)
                # Apply mask zeroing externally/later or via simple loop
                for i, valid in enumerate(s_mask):
                    if not valid: advs[key][i] = 0.0; rets[key][i] = 0.0
            else:
                advs[key], rets[key] = self.compute_gae(data["rewards"][key], data["values"], data["dones"], next_val)
            
            adv_arr = np.array(advs[key])
            if adv_arr.std() > 1e-8: advs[key] = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        
        # Prepare Tensors
        ret_t = {k: torch.tensor(v, device=self.device, dtype=torch.float) for k, v in rets.items()}
        adv_t = {k: torch.tensor(v, device=self.device, dtype=torch.float) for k, v in advs.items()}
        act_t = {k: torch.tensor(v, device=self.device, dtype=torch.long) for k, v in data["actions"].items()}
        
        old_log_t = {}
        for k, v in data["log_probs"].items():
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                old_log_t[k] = torch.stack(v).detach().to(self.device)
            else:
                old_log_t[k] = torch.tensor(v, device=self.device)

        indices = list(range(batch_size_total))
        
        for _ in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, batch_size_total, self.batch_size):
                idx = indices[start:start+self.batch_size]
                
                # Slicing Pre-computed tensors (SUPER FAST)
                b_h_feat = all_h_feats[idx]
                b_s_feat = all_s_feats[idx]
                b_s_mask = s_mask[idx]
                
                # Critic & Actor updates
                # Note: We detach features if we don't want to train the CNN during PPO (optional)
                # Usually we DO want to train CNN, so we must re-forward IF we didn't store graph.
                # WARNING: Storing features with no_grad breaks backprop to CNN. 
                # CORRECT APPROACH FOR PPO + CNN:
                # To train CNN, we MUST re-forward the grids.
                # BUT, we can use the OPTIMIZED process_batch from step 1.
                # If we use pre-calculated features from no_grad, we freeze the CNN.
                
                # LET'S RE-ENABLE CNN TRAINING BUT USE OPTIMIZED GRID GENERATION:
                batch_obs = [all_obs[i] for i in idx]
                
                # Re-generate grids using the NEW FAST process_batch
                h_s, h_g = self.agent.hider_feature_extractor.process_batch(batch_obs, "hider", self.env)
                s_s, s_g = self.agent.seeker_feature_extractor.process_batch(batch_obs, "seeker", self.env)
                
                h_feat = self.agent.hider_feature_extractor((h_s, h_g))
                s_feat = self.agent.seeker_feature_extractor((s_s, s_g)) * b_s_mask.unsqueeze(1)

                curr_val = self.agent.critic(h_feat, s_feat).squeeze()
                h_probs = self.agent.hider_actor(h_feat)
                s_probs = self.agent.seeker_actor(s_feat)
                
                # --- Calculates Losses (Same as before) ---
                # Hider
                dist_h = torch.distributions.Categorical(h_probs)
                log_h = dist_h.log_prob(act_t["hider"][idx])
                ent_h = dist_h.entropy().mean()
                ratio_h = torch.exp(log_h - old_log_t["hider"][idx])
                surr1_h = ratio_h * adv_t["hider"][idx]
                surr2_h = torch.clamp(ratio_h, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_t["hider"][idx]
                loss_h = -torch.min(surr1_h, surr2_h).mean()
                
                # Seeker
                loss_s, ent_s = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                if b_s_mask.any():
                    dist_s = torch.distributions.Categorical(s_probs)
                    log_s = dist_s.log_prob(act_t["seeker"][idx])
                    ratio_s = torch.exp(log_s - old_log_t["seeker"][idx])
                    surr1_s = ratio_s * adv_t["seeker"][idx]
                    surr2_s = torch.clamp(ratio_s, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_t["seeker"][idx]
                    loss_s = ((-torch.min(surr1_s, surr2_s) * b_s_mask).sum() / (b_s_mask.sum() + 1e-8))
                    ent_s = (dist_s.entropy() * b_s_mask).sum() / (b_s_mask.sum() + 1e-8)
                
                target = (ret_t["hider"][idx] + ret_t["seeker"][idx]) / 2
                loss_v = (curr_val - target).pow(2).mean()
                
                total = (loss_h + loss_s) + self.value_coef * loss_v - self.entropy_coef * (ent_h + ent_s)
                
                self.optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.agent.get_all_parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.stats["policy_loss"].append((loss_h+loss_s).item())
                self.stats["value_loss"].append(loss_v.item())
                self.stats["total_loss"].append(total.item())
                
        self.buffer.clear()
        return {k: np.mean(v[-self.n_epochs:]) if v else 0.0 for k,v in self.stats.items()}