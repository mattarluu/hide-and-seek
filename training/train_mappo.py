"""
train_mappo.py

Main training script for MAPPO hide and seek with:
- Parallel environments
- Curriculum learning
- Self-play
- MLflow experiment tracking
- Model checkpointing
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.hide_and_seek_env import HideAndSeekEnv
from mappo_network import MAPPOAgent
from mappo_trainer import MAPPOTrainer, RolloutBuffer
from parallel_env import make_parallel_env


class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    """
    def __init__(self, stages):
        """
        Args:
            stages: List of dicts with curriculum stages
                   Each dict has: {'name', 'duration_steps', 'config'}
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.steps_in_stage = 0
        
    def get_current_stage(self):
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def should_advance(self):
        """Check if should move to next stage."""
        current = self.stages[self.current_stage_idx]
        return self.steps_in_stage >= current['duration_steps']
    
    def advance(self):
        """Move to next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.steps_in_stage = 0
            return True
        return False
    
    def step(self):
        """Increment step counter."""
        self.steps_in_stage += 1


class HideAndSeekTrainer:
    """
    Complete training pipeline for hide and seek with MAPPO.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Curriculum stages
        self.curriculum = self._create_curriculum()
        
        # Create environments
        print(f"Creating {args.n_envs} parallel environments...")
        self.envs = make_parallel_env(
            lambda: HideAndSeekEnv(),
            n_envs=args.n_envs
        )
        
        # Create single environment for agent initialization
        self.eval_env = HideAndSeekEnv()
        
        # Initialize agent
        print("Initializing MAPPO agent...")
        self.agent = MAPPOAgent(
            grid_size=20,
            action_dim=10,
            device=self.device
        )
        
        # Initialize trainer
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
        
        # Training state
        self.total_steps = 0
        self.episode_rewards = {"hider": [], "seeker": []}
        self.episode_lengths = []
        
    def _create_curriculum(self):
        """
        Create curriculum learning stages.
        
        Stage 1: Simple environment (few obstacles, short episodes)
        Stage 2: Medium complexity (more obstacles, longer episodes)
        Stage 3: Full complexity (all features enabled)
        """
        stages = [
            {
                'name': 'Stage 1: Basic Navigation',
                'duration_steps': 100000,
                'config': {
                    'max_episode_steps': 50,
                    'seeker_delay': 20,
                    'description': 'Learn basic movement and object interaction'
                }
            },
            {
                'name': 'Stage 2: Intermediate Tactics',
                'duration_steps': 200000,
                'config': {
                    'max_episode_steps': 75,
                    'seeker_delay': 15,
                    'description': 'Develop hiding strategies and seeking tactics'
                }
            },
            {
                'name': 'Stage 3: Advanced Play',
                'duration_steps': float('inf'),
                'config': {
                    'max_episode_steps': 100,
                    'seeker_delay': 10,
                    'description': 'Full complexity with all mechanics'
                }
            }
        ]
        return CurriculumScheduler(stages)
    
    def collect_rollouts(self, n_steps):
        """
        Collect rollouts from parallel environments.
        
        Args:
            n_steps: Number of steps to collect per environment
            
        Returns:
            buffer: Filled rollout buffer
        """
        buffer = RolloutBuffer(max_size=n_steps * self.args.n_envs)
        
        # Reset environments
        obs_list = self.envs.reset()
        
        episode_rewards = {agent: [0.0] * self.args.n_envs for agent in ["hider", "seeker"]}
        episode_lengths = [0] * self.args.n_envs
        
        completed_episodes = {"hider": [], "seeker": [], "lengths": []}
        
        for step in range(n_steps):
            actions_list = []
            log_probs_list = []
            values_list = []
            
            # Get actions for all environments
            for env_idx, obs in enumerate(obs_list):
                actions = {}
                log_probs = {}
                
                # Hider action
                action, log_prob, _ = self.agent.get_action(obs, "hider", self.eval_env)
                actions["hider"] = action
                log_probs["hider"] = log_prob
                
                # Seeker action (if active)
                if obs["seeker"]["state"][0] >= 0:
                    action, log_prob, _ = self.agent.get_action(obs, "seeker", self.eval_env)
                    actions["seeker"] = action
                    log_probs["seeker"] = log_prob
                else:
                    actions["seeker"] = 0  # Dummy action
                    log_probs["seeker"] = torch.tensor(0.0)
                
                # Get value estimate
                value = self.agent.get_value(obs, self.eval_env)
                
                actions_list.append(actions)
                log_probs_list.append(log_probs)
                values_list.append(value.item())
            
            # Step all environments
            next_obs_list, dones, rewards_list = self.envs.step(actions_list)
            
            # Store transitions
            for env_idx in range(self.args.n_envs):
                buffer.add(
                    obs=obs_list[env_idx],
                    actions=actions_list[env_idx],
                    rewards=rewards_list[env_idx],
                    value=values_list[env_idx],
                    log_probs=log_probs_list[env_idx],
                    done=dones[env_idx]
                )
                
                # Track episode statistics
                episode_rewards["hider"][env_idx] += rewards_list[env_idx]["hider"]
                episode_rewards["seeker"][env_idx] += rewards_list[env_idx]["seeker"]
                episode_lengths[env_idx] += 1
                
                # Episode completed
                if dones[env_idx]:
                    completed_episodes["hider"].append(episode_rewards["hider"][env_idx])
                    completed_episodes["seeker"].append(episode_rewards["seeker"][env_idx])
                    completed_episodes["lengths"].append(episode_lengths[env_idx])
                    
                    episode_rewards["hider"][env_idx] = 0.0
                    episode_rewards["seeker"][env_idx] = 0.0
                    episode_lengths[env_idx] = 0
            
            obs_list = next_obs_list
            self.total_steps += self.args.n_envs
        
        return buffer, completed_episodes, obs_list[0]
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            metrics: Dict of evaluation metrics
        """
        eval_rewards = {"hider": [], "seeker": []}
        eval_lengths = []
        
        for ep in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = {"hider": 0.0, "seeker": 0.0}
            ep_length = 0
            
            while not done:
                # Get deterministic actions
                hider_action, _, _ = self.agent.get_action(
                    obs, "hider", self.eval_env, deterministic=True
                )
                
                if obs["seeker"]["state"][0] >= 0:
                    seeker_action, _, _ = self.agent.get_action(
                        obs, "seeker", self.eval_env, deterministic=True
                    )
                else:
                    seeker_action = 0
                
                actions = {"hider": hider_action, "seeker": seeker_action}
                obs, done, rewards = self.eval_env.step(actions)
                
                ep_reward["hider"] += rewards["hider"]
                ep_reward["seeker"] += rewards["seeker"]
                ep_length += 1
            
            eval_rewards["hider"].append(ep_reward["hider"])
            eval_rewards["seeker"].append(ep_reward["seeker"])
            eval_lengths.append(ep_length)
        
        return {
            "eval/hider_reward_mean": np.mean(eval_rewards["hider"]),
            "eval/seeker_reward_mean": np.mean(eval_rewards["seeker"]),
            "eval/episode_length_mean": np.mean(eval_lengths),
            "eval/hider_reward_std": np.std(eval_rewards["hider"]),
            "eval/seeker_reward_std": np.std(eval_rewards["seeker"])
        }
    
    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.args.save_dir,
            f"checkpoint_episode_{episode}.pt"
        )
        
        torch.save({
            'episode': episode,
            'total_steps': self.total_steps,
            'hider_feature_extractor': self.agent.hider_feature_extractor.state_dict(),
            'seeker_feature_extractor': self.agent.seeker_feature_extractor.state_dict(),
            'hider_actor': self.agent.hider_actor.state_dict(),
            'seeker_actor': self.agent.seeker_actor.state_dict(),
            'critic': self.agent.critic.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def train(self):
        """Main training loop."""
        # Create metrics log file
        metrics_file = os.path.join(self.args.log_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        print(f"\n{'='*60}")
        print(f"Starting MAPPO Training")
        print(f"Device: {self.device}")
        print(f"Parallel Envs: {self.args.n_envs}")
        print(f"Total Timesteps: {self.args.total_timesteps}")
        print(f"Metrics log: {metrics_file}")
        print(f"{'='*60}\n")

        # Write CSV header
        with open(metrics_file, 'w') as f:
            f.write("step,hider_reward_mean,seeker_reward_mean,episode_length_mean,n_episodes,"
                   "policy_loss,value_loss,entropy,total_loss,approx_kl,"
                   "eval_hider_reward_mean,eval_seeker_reward_mean,eval_episode_length_mean,curriculum_stage\n")
        
            update_num = 0
            pbar = tqdm(total=self.args.total_timesteps, desc="Training")
            
            while self.total_steps < self.args.total_timesteps:
                # Check curriculum advancement
                stage = self.curriculum.get_current_stage()
                if self.curriculum.should_advance():
                    if self.curriculum.advance():
                        new_stage = self.curriculum.get_current_stage()
                        print(f"\n🎓 Advancing to {new_stage['name']}")
                
                # Collect rollouts
                buffer, completed_episodes, last_obs = self.collect_rollouts(
                    self.args.rollout_steps
                )
                
                # Prepare metrics for logging
                metrics = {
                    'step': self.total_steps,
                    'hider_reward_mean': 0.0,
                    'seeker_reward_mean': 0.0,
                    'episode_length_mean': 0.0,
                    'n_episodes': 0,
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0,
                    'total_loss': 0.0,
                    'approx_kl': 0.0,
                    'eval_hider_reward_mean': 0.0,
                    'eval_seeker_reward_mean': 0.0,
                    'eval_episode_length_mean': 0.0,
                    'curriculum_stage': self.curriculum.current_stage_idx
                }
                
                # Log training episodes
                if len(completed_episodes["hider"]) > 0:
                    metrics['hider_reward_mean'] = np.mean(completed_episodes["hider"])
                    metrics['seeker_reward_mean'] = np.mean(completed_episodes["seeker"])
                    metrics['episode_length_mean'] = np.mean(completed_episodes["lengths"])
                    metrics['n_episodes'] = len(completed_episodes["hider"])
                
                # Update policy using collected buffer
                self.trainer.buffer = buffer
                update_stats = self.trainer.update(last_obs)
                
                # Update metrics with training stats
                metrics['policy_loss'] = update_stats.get('policy_loss', 0.0)
                metrics['value_loss'] = update_stats.get('value_loss', 0.0)
                metrics['entropy'] = update_stats.get('entropy', 0.0)
                metrics['total_loss'] = update_stats.get('total_loss', 0.0)
                metrics['approx_kl'] = update_stats.get('approx_kl', 0.0)
                
                update_num += 1
                self.curriculum.step()
                
                # Periodic evaluation
                if update_num % self.args.eval_freq == 0:
                    eval_metrics = self.evaluate(n_episodes=self.args.eval_episodes)
                    
                    metrics['eval_hider_reward_mean'] = eval_metrics['eval/hider_reward_mean']
                    metrics['eval_seeker_reward_mean'] = eval_metrics['eval/seeker_reward_mean']
                    metrics['eval_episode_length_mean'] = eval_metrics['eval/episode_length_mean']
                    
                    print(f"\n📊 Evaluation at step {self.total_steps}:")
                    print(f"  Hider Reward: {eval_metrics['eval/hider_reward_mean']:.2f} "
                          f"± {eval_metrics['eval/hider_reward_std']:.2f}")
                    print(f"  Seeker Reward: {eval_metrics['eval/seeker_reward_mean']:.2f} "
                          f"± {eval_metrics['eval/seeker_reward_std']:.2f}")
                    print(f"  Episode Length: {eval_metrics['eval/episode_length_mean']:.1f}")
                
                # Write metrics to CSV
                with open(metrics_file, 'a') as f:
                    f.write(f"{metrics['step']},{metrics['hider_reward_mean']:.4f},"
                           f"{metrics['seeker_reward_mean']:.4f},{metrics['episode_length_mean']:.4f},"
                           f"{metrics['n_episodes']},{metrics['policy_loss']:.6f},"
                           f"{metrics['value_loss']:.6f},{metrics['entropy']:.6f},"
                           f"{metrics['total_loss']:.6f},{metrics['approx_kl']:.6f},"
                           f"{metrics['eval_hider_reward_mean']:.4f},"
                           f"{metrics['eval_seeker_reward_mean']:.4f},"
                           f"{metrics['eval_episode_length_mean']:.4f},"
                           f"{metrics['curriculum_stage']}\n")
                
                # Save checkpoint
                if update_num % self.args.save_freq == 0:
                    self.save_checkpoint(update_num)
                
                pbar.update(self.args.rollout_steps * self.args.n_envs)
                pbar.set_postfix({
                    'stage': stage['name'][:15],
                    'updates': update_num
                })
            
            pbar.close()
            
            # Final evaluation and checkpoint
            print("\n🏁 Training complete! Running final evaluation...")
            final_metrics = self.evaluate(n_episodes=50)
            
            # Save final metrics
            final_results = {
                'final_hider_reward': final_metrics['eval/hider_reward_mean'],
                'final_seeker_reward': final_metrics['eval/seeker_reward_mean'],
                'final_episode_length': final_metrics['eval/episode_length_mean'],
                'total_steps': self.total_steps
            }
            
            results_file = os.path.join(self.args.log_dir, 'final_results.json')
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            final_checkpoint = self.save_checkpoint('final')
            
            print(f"\n{'='*60}")
            print("Final Results:")
            print(f"  Hider Reward: {final_metrics['eval/hider_reward_mean']:.2f}")
            print(f"  Seeker Reward: {final_metrics['eval/seeker_reward_mean']:.2f}")
            print(f"  Episode Length: {final_metrics['eval/episode_length_mean']:.1f}")
            print(f"  Results saved to: {results_file}")
            print(f"{'='*60}\n")
        
        self.envs.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MAPPO for Hide and Seek")
    
    # Environment
    parser.add_argument('--n-envs', type=int, default=16,
                       help='Number of parallel environments')
    
    # Training
    parser.add_argument('--total-timesteps', type=int, default=5_000_000,
                       help='Total training timesteps')
    parser.add_argument('--rollout-steps', type=int, default=128,
                       help='Steps per rollout per environment')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clipping epsilon')
    parser.add_argument('--value-coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Max gradient norm')
    parser.add_argument('--n-epochs', type=int, default=4,
                       help='Number of epochs per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Mini-batch size')
    
    # Evaluation
    parser.add_argument('--eval-freq', type=int, default=10,
                       help='Evaluation frequency (in updates)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    # Logging
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Checkpoint save frequency (in updates)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = HideAndSeekTrainer(args)
    trainer.train()