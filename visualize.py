"""
visualize.py

Visualization utilities for hide and seek training.
Integrates with existing utils/visualization.py for rendering.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.hide_and_seek_env import HideAndSeekEnv
from mappo_network import MAPPOAgent

# Import existing visualization utilities
try:
    from utils.visualization import render_environment, draw_grid, draw_room, draw_objects, draw_agent
except ImportError:
    print("Warning: utils.visualization not found. Using built-in rendering.")


def render_episode(env, agent, checkpoint_path=None, save_path=None, deterministic=True, use_utils_render=True):
    """
    Render and save an episode as an animation.
    
    Args:
        env: Environment instance
        agent: MAPPOAgent instance
        checkpoint_path: Path to trained checkpoint (optional)
        save_path: Path to save animation (optional)
        deterministic: Use deterministic actions
        use_utils_render: Try to use utils/visualization.py if available
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.hider_feature_extractor.load_state_dict(checkpoint['hider_feature_extractor'])
        agent.seeker_feature_extractor.load_state_dict(checkpoint['seeker_feature_extractor'])
        agent.hider_actor.load_state_dict(checkpoint['hider_actor'])
        agent.seeker_actor.load_state_dict(checkpoint['seeker_actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    
    # Collect episode trajectory
    obs = env.reset()
    done = False
    trajectory = []
    
    agent.hider_feature_extractor.eval()
    agent.seeker_feature_extractor.eval()
    agent.hider_actor.eval()
    agent.seeker_actor.eval()
    
    with torch.no_grad():
        while not done:
            # Store current state (save full environment state)
            trajectory.append({
                'env_state': env,
                'obs': obs
            })
            
            # Get actions
            hider_action, _, _ = agent.get_action(obs, "hider", env, deterministic=deterministic)
            
            if obs["seeker"]["state"][0] >= 0:
                seeker_action, _, _ = agent.get_action(obs, "seeker", env, deterministic=deterministic)
            else:
                seeker_action = 0
            
            actions = {"hider": hider_action, "seeker": seeker_action}
            obs, done, rewards = env.step(actions)
    
    # Try to use existing visualization utilities
    if use_utils_render:
        try:
            from utils.visualization import render_environment
            fig, ax = plt.subplots(figsize=(10, 10))
            
            def init():
                return []
            
            def update(frame):
                # Set environment state from trajectory
                # Note: This is a simplified approach - ideally we'd save/restore full state
                render_environment(ax, env)
                ax.set_title(f"Hide and Seek - Step {frame}/{len(trajectory)-1}")
                return []
            
            anim = FuncAnimation(fig, update, init_func=init, frames=len(trajectory),
                                interval=200, blit=True, repeat=True)
            
            if save_path:
                print(f"Saving animation to {save_path}...")
                writer = PillowWriter(fps=5)
                anim.save(save_path, writer=writer)
                print(f"Animation saved!")
            else:
                plt.show()
            
            plt.close()
            return
            
        except Exception as e:
            print(f"Could not use utils.visualization: {e}")
            print("Falling back to built-in rendering...")
    
    # Fallback: Use built-in simple rendering
    _render_episode_builtin(env, trajectory, save_path)

def _render_episode_builtin(env, trajectory, save_path=None):
    """
    Built-in rendering when utils.visualization is not available.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def init():
        ax.clear()
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title("Hide and Seek - MAPPO")
        ax.grid(True, alpha=0.3)
        return []
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Note: trajectory just contains obs, not full environment state
        # This is a limitation - for full rendering we'd need to save more state
        obs = trajectory[frame]['obs']
        
        # Draw room walls
        for wall in env.room.wall_cells:
            rect = patches.Rectangle(wall, 1, 1, facecolor='gray', edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        # Draw blocks (current positions)
        for block in env.blocks:
            block_pos = block.position
            rect = patches.Rectangle(block_pos, 1, 1, facecolor='brown', edgecolor='black')
            ax.add_patch(rect)
            ax.text(block_pos[0] + 0.5, block_pos[1] + 0.5, 'B', 
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Draw ramp
        if env.ramp:
            ramp_pos = env.ramp.position
            triangle = patches.Polygon(
                [(ramp_pos[0], ramp_pos[1] + 1),
                 (ramp_pos[0] + 1, ramp_pos[1] + 1),
                 (ramp_pos[0] + 1, ramp_pos[1])],
                facecolor='yellow', edgecolor='black'
            )
            ax.add_patch(triangle)
            ax.text(ramp_pos[0] + 0.7, ramp_pos[1] + 0.7, 'R',
                   ha='center', va='center', color='black', fontweight='bold', fontsize=8)
        
        # Draw hider
        hider_state = obs['hider']['state']
        if hider_state[0] >= 0:
            hider_pos = (hider_state[0], hider_state[1])
            hider_z = hider_state[3]
            hider_color = 'blue' if hider_z == 0 else 'darkblue'
            hider_circle = patches.Circle(
                (hider_pos[0] + 0.5, hider_pos[1] + 0.5),
                0.3, facecolor=hider_color, edgecolor='black', linewidth=2
            )
            ax.add_patch(hider_circle)
            ax.text(hider_pos[0] + 0.5, hider_pos[1] + 0.5, 'H',
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Draw seeker (if active)
        seeker_state = obs['seeker']['state']
        if seeker_state[0] >= 0:
            seeker_pos = (seeker_state[0], seeker_state[1])
            seeker_z = seeker_state[3]
            seeker_color = 'red' if seeker_z == 0 else 'darkred'
            seeker_circle = patches.Circle(
                (seeker_pos[0] + 0.5, seeker_pos[1] + 0.5),
                0.3, facecolor=seeker_color, edgecolor='black', linewidth=2
            )
            ax.add_patch(seeker_circle)
            ax.text(seeker_pos[0] + 0.5, seeker_pos[1] + 0.5, 'S',
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='blue', label='Hider (z=0)'),
            patches.Patch(facecolor='darkblue', label='Hider (z=1)'),
            patches.Patch(facecolor='red', label='Seeker (z=0)'),
            patches.Patch(facecolor='darkred', label='Seeker (z=1)'),
            patches.Patch(facecolor='brown', label='Block'),
            patches.Patch(facecolor='yellow', label='Ramp'),
            patches.Patch(facecolor='gray', label='Wall')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.set_title(f"Hide and Seek - Step {frame}/{len(trajectory)-1}")
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(trajectory),
                        interval=200, blit=True, repeat=True)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        writer = PillowWriter(fps=5)
        anim.save(save_path, writer=writer)
        print(f"Animation saved!")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training curves from CSV logs.
    
    Args:
        log_dir: Path to logs directory containing CSV files
        save_path: Path to save plot (optional)
    """
    import pandas as pd
    import glob
    
    # Find most recent metrics CSV file
    csv_files = glob.glob(os.path.join(log_dir, 'metrics_*.csv'))
    if not csv_files:
        print(f"No metrics CSV files found in {log_dir}")
        return
    
    csv_file = max(csv_files, key=os.path.getctime)
    print(f"Loading metrics from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MAPPO Training Progress', fontsize=16)
    
    # Subplot 1: Rewards
    if 'hider_reward_mean' in df.columns and 'seeker_reward_mean' in df.columns:
        axes[0, 0].plot(df['step'], df['hider_reward_mean'], label='Hider', color='blue', alpha=0.6)
        axes[0, 0].plot(df['step'], df['seeker_reward_mean'], label='Seeker', color='red', alpha=0.6)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Policy Loss
    if 'policy_loss' in df.columns:
        axes[0, 1].plot(df['step'], df['policy_loss'], color='green', alpha=0.6)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Value Loss
    if 'value_loss' in df.columns:
        axes[0, 2].plot(df['step'], df['value_loss'], color='orange', alpha=0.6)
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Value Loss')
        axes[0, 2].set_title('Value Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Subplot 4: Episode Length
    if 'episode_length_mean' in df.columns:
        axes[1, 0].plot(df['step'], df['episode_length_mean'], color='purple', alpha=0.6)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 5: Entropy
    if 'entropy' in df.columns:
        axes[1, 1].plot(df['step'], df['entropy'], color='brown', alpha=0.6)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Subplot 6: Evaluation Rewards
    if 'eval_hider_reward_mean' in df.columns and 'eval_seeker_reward_mean' in df.columns:
        # Filter out zero values (not evaluated yet)
        eval_df = df[df['eval_hider_reward_mean'] != 0]
        if len(eval_df) > 0:
            axes[1, 2].plot(eval_df['step'], eval_df['eval_hider_reward_mean'], 
                          label='Hider', color='blue', marker='o')
            axes[1, 2].plot(eval_df['step'], eval_df['eval_seeker_reward_mean'], 
                          label='Seeker', color='red', marker='o')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Eval Reward')
            axes[1, 2].set_title('Evaluation Rewards')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize hide and seek training")
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    # Episode rendering
    render_parser = subparsers.add_parser('episode', help='Render an episode')
    render_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to checkpoint')
    render_parser.add_argument('--save', type=str, default=None,
                              help='Path to save animation (e.g., episode.gif)')
    render_parser.add_argument('--stochastic', action='store_true',
                              help='Use stochastic policy')
    
    # Training curves
    curves_parser = subparsers.add_parser('curves', help='Plot training curves')
    curves_parser.add_argument('--log-dir', type=str, default='./logs',
                              help='Path to logs directory')
    curves_parser.add_argument('--save', type=str, default=None,
                              help='Path to save plot')
    
    args = parser.parse_args()
    
    if args.command == 'episode':
        print("Initializing environment and agent...")
        env = HideAndSeekEnv()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = MAPPOAgent(grid_size=20, action_dim=10, device=device)
        
        print("Rendering episode...")
        render_episode(
            env, agent,
            checkpoint_path=args.checkpoint,
            save_path=args.save,
            deterministic=not args.stochastic
        )
    
    elif args.command == 'curves':
        print("Plotting training curves...")
        plot_training_curves(args.log_dir, args.save)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()