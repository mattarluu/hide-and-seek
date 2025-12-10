"""
train_mappo.py
Main training script.
Fixed: Removed invalid 'verbose' argument.
"""
import os
import sys
import torch
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.hide_and_seek_env import HideAndSeekEnv
from mappo_network import MAPPOAgent
from mappo_trainer import MAPPOTrainer, RolloutBuffer
from parallel_env import make_parallel_env

class CurriculumScheduler:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage_idx = 0
        self.steps_in_stage = 0
    def get_current_stage(self): return self.stages[self.current_stage_idx]
    def should_advance(self): return self.steps_in_stage >= self.stages[self.current_stage_idx]['duration_steps']
    def advance(self):
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.steps_in_stage = 0
            return True
        return False
    def step(self): self.steps_in_stage += 1

class HideAndSeekTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
        self.curriculum = self._create_curriculum()
        
        print(f"Creating {args.n_envs} parallel environments...")
        # CORRECCIÓN: Quitamos verbose=False porque tu entorno no lo soporta
        self.envs = make_parallel_env(lambda: HideAndSeekEnv(), n_envs=args.n_envs)
        
        self.eval_env = HideAndSeekEnv()
        print("Initializing MAPPO agent...")
        self.agent = MAPPOAgent(grid_size=20, action_dim=10, device=self.device)
        print("Initializing MAPPO trainer...")
        self.trainer = MAPPOTrainer(
            agent=self.agent, env=self.eval_env, lr=args.lr, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm, n_epochs=args.n_epochs,
            batch_size=args.batch_size, device=self.device
        )
        self.total_steps = 0
        
    def _create_curriculum(self):
        return CurriculumScheduler([
            {'name': 'Stage 1', 'duration_steps': 100000, 'config': {'max_episode_steps': 50}},
            {'name': 'Stage 2', 'duration_steps': 200000, 'config': {'max_episode_steps': 75}},
            {'name': 'Stage 3', 'duration_steps': float('inf'), 'config': {'max_episode_steps': 100}}
        ])
    
    def collect_rollouts(self, n_steps):
        buffer = RolloutBuffer(max_size=n_steps * self.args.n_envs)
        obs_list = self.envs.reset()
        episode_rewards = {k: [0.0]*self.args.n_envs for k in ["hider", "seeker"]}
        episode_lengths = [0] * self.args.n_envs
        completed = {"hider": [], "seeker": [], "lengths": []}
        
        for _ in range(n_steps):
            with torch.no_grad():
                h_s, h_g = self.agent.hider_feature_extractor.process_batch(obs_list, "hider", self.eval_env)
                s_s, s_g = self.agent.seeker_feature_extractor.process_batch(obs_list, "seeker", self.eval_env)
                h_feat = self.agent.hider_feature_extractor((h_s, h_g))
                s_feat = self.agent.seeker_feature_extractor((s_s, s_g))
                
                h_act_probs = self.agent.hider_actor(h_feat)
                s_act_probs = self.agent.seeker_actor(s_feat)
                
                h_dist = torch.distributions.Categorical(h_act_probs)
                s_dist = torch.distributions.Categorical(s_act_probs)
                h_actions = h_dist.sample()
                s_actions = s_dist.sample()
                h_log = h_dist.log_prob(h_actions)
                s_log = s_dist.log_prob(s_actions)
                
                seeker_mask = torch.tensor([o["seeker"]["state"][0] >= 0 for o in obs_list], device=self.device)
                s_feat_masked = s_feat * seeker_mask.unsqueeze(1)
                values = self.agent.critic(h_feat, s_feat_masked)
                
            actions_list = []
            log_probs_list = []
            values_list = values.cpu().numpy().tolist()
            
            for i in range(self.args.n_envs):
                actions_list.append({
                    "hider": h_actions[i].item(),
                    "seeker": s_actions[i].item() if seeker_mask[i] else 0
                })
                log_probs_list.append({
                    "hider": h_log[i],
                    "seeker": s_log[i] if seeker_mask[i] else torch.tensor(0.0, device=self.device)
                })
            
            next_obs, dones, rewards = self.envs.step(actions_list)
            
            for i in range(self.args.n_envs):
                buffer.add(obs_list[i], actions_list[i], rewards[i], values_list[i], log_probs_list[i], dones[i])
                episode_rewards["hider"][i] += rewards[i]["hider"]
                episode_rewards["seeker"][i] += rewards[i]["seeker"]
                episode_lengths[i] += 1
                if dones[i]:
                    completed["hider"].append(episode_rewards["hider"][i])
                    completed["seeker"].append(episode_rewards["seeker"][i])
                    completed["lengths"].append(episode_lengths[i])
                    episode_rewards["hider"][i] = 0.0
                    episode_rewards["seeker"][i] = 0.0
                    episode_lengths[i] = 0
            
            obs_list = next_obs
            self.total_steps += self.args.n_envs
        return buffer, completed, obs_list[0]

    def train(self):
        print(f"\n{'='*60}\nStarting MAPPO Training\nDevice: {self.device}\nEnvs: {self.args.n_envs}\n{'='*60}\n")
        pbar = tqdm(total=self.args.total_timesteps)
        update_num = 0
        
        while self.total_steps < self.args.total_timesteps:
            if self.curriculum.should_advance() and self.curriculum.advance():
                print(f"\n>>> Stage: {self.curriculum.get_current_stage()['name']}")
                self.writer.add_scalar("Config/Stage", self.curriculum.current_stage_idx, self.total_steps)
            
            buffer, completed, last_obs = self.collect_rollouts(self.args.rollout_steps)
            self.trainer.buffer = buffer
            stats = self.trainer.update(last_obs)
            
            for k, v in stats.items(): self.writer.add_scalar(f"Loss/{k}", v, self.total_steps)
            if completed["hider"]:
                self.writer.add_scalar("Reward/Hider", np.mean(completed["hider"]), self.total_steps)
                self.writer.add_scalar("Reward/Seeker", np.mean(completed["seeker"]), self.total_steps)
            
            update_num += 1
            self.curriculum.step()
            
            if update_num % self.args.eval_freq == 0:
                # Simple eval log
                pass 
                
            if update_num % self.args.save_freq == 0:
                self.save_checkpoint(update_num)
            
            pbar.update(self.args.rollout_steps * self.args.n_envs)
            
        pbar.close()
        self.save_checkpoint('final')
        self.writer.close()
        self.envs.close()

    def save_checkpoint(self, ep):
        path = os.path.join(self.args.save_dir, f"ckpt_{ep}.pt")
        torch.save({'agent': self.agent.get_all_parameters()}, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=32)
    parser.add_argument('--total-timesteps', type=int, default=5000000)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--eval-freq', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--save-freq', type=int, default=100)
    return parser.parse_args()

if __name__ == "__main__":
    trainer = HideAndSeekTrainer(parse_args())
    trainer.train()