#!/usr/bin/env python3
"""
TRAIN_MAPPO_FINAL.py
Versión completamente corregida y optimizada del entrenamiento MAPPO.

Correcciones aplicadas:
1. Workers paralelos con señalización 'ready' para evitar deadlocks
2. Timeouts apropiados en comunicación entre procesos
3. Manejo robusto de errores
4. Conversión correcta de tensores a tipos nativos de Python
5. Progress bar informativa
"""
import os
import sys
import torch
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.hide_and_seek_env import HideAndSeekEnv
from mappo_network import MAPPOAgent
from mappo_trainer import MAPPOTrainer, RolloutBuffer
from parallel_env import make_parallel_env

def make_env():
    """Función auxiliar para crear el entorno, necesaria para multiprocessing."""
    return HideAndSeekEnv()

class CurriculumScheduler:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage_idx = 0
        self.steps_in_stage = 0
    
    def get_current_stage(self):
        return self.stages[self.current_stage_idx]
    
    def should_advance(self):
        return self.steps_in_stage >= self.stages[self.current_stage_idx]['duration_steps']
    
    def advance(self):
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.steps_in_stage = 0
            return True
        return False
    
    def step(self):
        self.steps_in_stage += 1


class HideAndSeekTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
        
        self.curriculum = self._create_curriculum()
        
        print(f"\nCreating {args.n_envs} parallel environments...")
        self.envs = make_parallel_env(make_env, n_envs=args.n_envs)
        
        self.eval_env = HideAndSeekEnv()
        
        print("Initializing MAPPO agent...")
        self.agent = MAPPOAgent(
            grid_size=20,
            action_dim=10,
            device=self.device
        )
        
        print("Initializing MAPPO trainer...")
        self.trainer = MAPPOTrainer(
            agent=self.agent,
            env=self.eval_env,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=self.device
        )
        
        self.total_steps = 0
    
    def _create_curriculum(self):
        stages = [
            {'name': 'Stage 1: Basic Navigation', 'duration_steps': 100000, 'config': {'max_episode_steps': 50}},
            {'name': 'Stage 2: Intermediate Tactics', 'duration_steps': 200000, 'config': {'max_episode_steps': 75}},
            {'name': 'Stage 3: Advanced Play', 'duration_steps': float('inf'), 'config': {'max_episode_steps': 100}}
        ]
        return CurriculumScheduler(stages)
    
    def collect_rollouts(self, n_steps):
        """
        Recolecta experiencias de los entornos paralelos.
        CRÍTICO: Maneja correctamente la conversión de tensores a tipos nativos de Python.
        """
        buffer = RolloutBuffer(max_size=n_steps * self.args.n_envs)
        obs_list = self.envs.reset()
        
        episode_rewards = {agent: [0.0] * self.args.n_envs for agent in ["hider", "seeker"]}
        episode_lengths = [0] * self.args.n_envs
        completed_episodes = {"hider": [], "seeker": [], "lengths": []}
        
        for step in range(n_steps):
            print(f"\rCollecting step {step+1}/{n_steps}", end="", flush=True)
            # Obtener acciones en batch (con gradientes desactivados)
            with torch.no_grad():
                h_actions, h_log_probs, _ = self.agent.get_action(obs_list, "hider", self.eval_env)
                s_actions, s_log_probs, _ = self.agent.get_action(obs_list, "seeker", self.eval_env)
                values = self.agent.get_value(obs_list, self.eval_env)
            
            # CRÍTICO: Convertir a numpy y luego a int nativo de Python
            # Los workers esperan tipos nativos de Python, no numpy types
            h_actions_np = h_actions.cpu().numpy()
            s_actions_np = s_actions.cpu().numpy()
            
            # Preparar acciones para enviar a los workers
            actions_list = []
            for i in range(self.args.n_envs):
                seeker_active = obs_list[i]["seeker"]["state"][0] >= 0
                
                # IMPORTANTE: Usar item() o int() para convertir a tipo nativo
                act = {
                    "hider": int(h_actions_np[i]),
                    "seeker": int(s_actions_np[i]) if seeker_active else 0
                }
                actions_list.append(act)
            
            # Ejecutar step - aquí es donde puede haber deadlock si los datos no son correctos
            next_obs_list, dones, rewards_list = self.envs.step(actions_list)
            
            # Guardar experiencias en el buffer
            for i in range(self.args.n_envs):
                seeker_active = obs_list[i]["seeker"]["state"][0] >= 0
                
                stored_actions = {
                    "hider": int(h_actions_np[i]),
                    "seeker": int(s_actions_np[i]) if seeker_active else 0
                }
                
                stored_log_probs = {
                    "hider": h_log_probs[i],
                    "seeker": s_log_probs[i] if seeker_active else torch.tensor(0.0, device=self.device)
                }
                
                buffer.add(
                    obs=obs_list[i],
                    actions=stored_actions,
                    rewards=rewards_list[i],
                    value=values[i].item(),
                    log_probs=stored_log_probs,
                    done=dones[i]
                )
                
                episode_rewards["hider"][i] += rewards_list[i]["hider"]
                episode_rewards["seeker"][i] += rewards_list[i]["seeker"]
                episode_lengths[i] += 1
                
                if dones[i]:
                    completed_episodes["hider"].append(episode_rewards["hider"][i])
                    completed_episodes["seeker"].append(episode_rewards["seeker"][i])
                    completed_episodes["lengths"].append(episode_lengths[i])
                    
                    episode_rewards["hider"][i] = 0.0
                    episode_rewards["seeker"][i] = 0.0
                    episode_lengths[i] = 0
            
            obs_list = next_obs_list
            self.total_steps += self.args.n_envs
        
        return buffer, completed_episodes, obs_list[0]
    
    def save_checkpoint(self, update_num):
        checkpoint_path = os.path.join(self.args.save_dir, f"checkpoint_{update_num}.pt")
        torch.save({
            'update_num': update_num,
            'total_steps': self.total_steps,
            'hider_feature': self.agent.hider_feature_extractor.state_dict(),
            'seeker_feature': self.agent.seeker_feature_extractor.state_dict(),
            'hider_actor': self.agent.hider_actor.state_dict(),
            'seeker_actor': self.agent.seeker_actor.state_dict(),
            'critic': self.agent.critic.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
        }, checkpoint_path)
        return checkpoint_path
    
    def train(self):
        print(f"\n{'='*70}")
        print(f" MAPPO Training - Hide and Seek")
        print(f"{'='*70}")
        print(f" Device:          {self.device}")
        print(f" Parallel Envs:   {self.args.n_envs}")
        print(f" Total Steps:     {self.args.total_timesteps:,}")
        print(f" Rollout Steps:   {self.args.rollout_steps}")
        print(f" Batch Size:      {self.args.batch_size}")
        print(f" Learning Rate:   {self.args.lr}")
        print(f" Save Dir:        {self.args.save_dir}")
        print(f" Log Dir:         {self.args.log_dir}")
        print(f"{'='*70}\n")
        
        update_num = 0
        pbar = tqdm(
            total=self.args.total_timesteps,
            desc="Training",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        try:
            while self.total_steps < self.args.total_timesteps:
                # Curriculum management
                stage = self.curriculum.get_current_stage()
                if self.curriculum.should_advance():
                    if self.curriculum.advance():
                        new_stage = self.curriculum.get_current_stage()
                        tqdm.write(f"\n>>> Curriculum: {new_stage['name']}")
                        self.writer.add_scalar("Config/Stage", self.curriculum.current_stage_idx, self.total_steps)
                
                # Collect rollouts
                buffer, completed_episodes, last_obs = self.collect_rollouts(self.args.rollout_steps)
                
                # Update policy
                self.trainer.buffer = buffer
                update_stats = self.trainer.update(last_obs)
                
                # TensorBoard logging
                for key, value in update_stats.items():
                    self.writer.add_scalar(f"Loss/{key}", value, self.total_steps)
                
                # Episode statistics
                if len(completed_episodes["hider"]) > 0:
                    h_mean = np.mean(completed_episodes["hider"])
                    s_mean = np.mean(completed_episodes["seeker"])
                    len_mean = np.mean(completed_episodes["lengths"])
                    
                    self.writer.add_scalar("Reward/Hider_Mean", h_mean, self.total_steps)
                    self.writer.add_scalar("Reward/Seeker_Mean", s_mean, self.total_steps)
                    self.writer.add_scalar("Episode/Length", len_mean, self.total_steps)
                    
                    pbar.set_postfix(
                        upd=update_num,
                        H=f'{h_mean:+.1f}',
                        S=f'{s_mean:+.1f}',
                        L=f'{len_mean:.0f}',
                        eps=len(completed_episodes["hider"])
                    )
                
                update_num += 1
                self.curriculum.step()
                
                # Save checkpoint
                if update_num % self.args.save_freq == 0:
                    ckpt_path = self.save_checkpoint(update_num)
                    tqdm.write(f"Checkpoint saved: {ckpt_path}")
                
                pbar.update(self.args.rollout_steps * self.args.n_envs)
            
            pbar.close()
            final_ckpt = self.save_checkpoint('final')
            print(f"\n✓ Training completed successfully!")
            print(f"✓ Final checkpoint: {final_ckpt}\n")
            
        except KeyboardInterrupt:
            pbar.close()
            print("\n\n⚠️ Training interrupted by user")
            ckpt = self.save_checkpoint(f'interrupted_{update_num}')
            print(f"✓ Checkpoint saved: {ckpt}")
        except Exception as e:
            pbar.close()
            print(f"\n\n❌ Training failed: {e}")
            traceback.print_exc()
            try:
                ckpt = self.save_checkpoint(f'error_{update_num}')
                print(f"✓ Emergency checkpoint: {ckpt}")
            except:
                print("✗ Could not save emergency checkpoint")
        finally:
            print("\nCleaning up...")
            self.writer.close()
            if hasattr(self.envs, 'close'):
                self.envs.close()
            print("✓ Cleanup complete\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MAPPO for Multi-Agent Hide and Seek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment
    parser.add_argument('--n-envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000, help='Total training timesteps')
    parser.add_argument('--rollout-steps', type=int, default=128, help='Steps per rollout per environment')
    
    # Algorithm
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--n-epochs', type=int, default=4, help='Number of epochs per update')
    parser.add_argument('--batch-size', type=int, default=512, help='Mini-batch size')
    
    # Logging & Saving
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--save-freq', type=int, default=200, help='Checkpoint save frequency (updates)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    args = parse_args()
    
    print("\n" + "="*70)
    print(" MAPPO Hide-and-Seek Training")
    print("="*70)
    print(f" Configuration:")
    for arg, value in sorted(vars(args).items()):
        print(f"   {arg:20s} = {value}")
    print("="*70 + "\n")
    
    trainer = HideAndSeekTrainer(args)
    trainer.train()