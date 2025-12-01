"""
evaluate_mappo.py

Evaluate trained MAPPO agents and optionally render episodes.
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.hide_and_seek_env import HideAndSeekEnv
from mappo_network import MAPPOAgent


def load_checkpoint(agent, checkpoint_path):
    """
    Load trained model from checkpoint.
    
    Args:
        agent: MAPPOAgent instance
        checkpoint_path: Path to checkpoint file
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    agent.hider_feature_extractor.load_state_dict(checkpoint['hider_feature_extractor'])
    agent.seeker_feature_extractor.load_state_dict(checkpoint['seeker_feature_extractor'])
    agent.hider_actor.load_state_dict(checkpoint['hider_actor'])
    agent.seeker_actor.load_state_dict(checkpoint['seeker_actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    
    print(f"✓ Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    print(f"  Total steps: {checkpoint.get('total_steps', 'unknown')}")
    
    return checkpoint


def evaluate_agent(agent, env, n_episodes=100, deterministic=True, verbose=False):
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained MAPPOAgent
        env: Environment instance
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        verbose: Print episode details
        
    Returns:
        results: Dict with evaluation statistics
    """
    agent.hider_feature_extractor.eval()
    agent.seeker_feature_extractor.eval()
    agent.hider_actor.eval()
    agent.seeker_actor.eval()
    agent.critic.eval()
    
    episode_rewards = {"hider": [], "seeker": []}
    episode_lengths = []
    hider_found_count = 0
    hider_in_room_steps = []
    
    with torch.no_grad():
        for ep in tqdm(range(n_episodes), desc="Evaluating"):
            obs = env.reset()
            done = False
            ep_reward = {"hider": 0.0, "seeker": 0.0}
            ep_length = 0
            in_room_steps = 0
            found = False
            
            while not done:
                # Get actions
                hider_action, _, _ = agent.get_action(
                    obs, "hider", env, deterministic=deterministic
                )
                
                if obs["seeker"]["state"][0] >= 0:
                    seeker_action, _, _ = agent.get_action(
                        obs, "seeker", env, deterministic=deterministic
                    )
                else:
                    seeker_action = 0
                
                actions = {"hider": hider_action, "seeker": seeker_action}
                obs, done, rewards = env.step(actions)
                
                ep_reward["hider"] += rewards["hider"]
                ep_reward["seeker"] += rewards["seeker"]
                ep_length += 1
                
                # Track if hider is in room
                hider_pos = obs["hider"]["state"][:2]
                if env.room.is_inside(hider_pos):
                    in_room_steps += 1
                
                # Track if seeker found hider
                if rewards["seeker"] > 0 and not found:
                    found = True
                    hider_found_count += 1
            
            episode_rewards["hider"].append(ep_reward["hider"])
            episode_rewards["seeker"].append(ep_reward["seeker"])
            episode_lengths.append(ep_length)
            hider_in_room_steps.append(in_room_steps)
            
            if verbose and ep < 5:
                print(f"\nEpisode {ep + 1}:")
                print(f"  Hider Reward: {ep_reward['hider']:.2f}")
                print(f"  Seeker Reward: {ep_reward['seeker']:.2f}")
                print(f"  Length: {ep_length}")
                print(f"  Hider in Room: {in_room_steps}/{ep_length} steps")
                print(f"  Found: {'Yes' if found else 'No'}")
    
    results = {
        "n_episodes": n_episodes,
        "hider_reward_mean": np.mean(episode_rewards["hider"]),
        "hider_reward_std": np.std(episode_rewards["hider"]),
        "hider_reward_min": np.min(episode_rewards["hider"]),
        "hider_reward_max": np.max(episode_rewards["hider"]),
        "seeker_reward_mean": np.mean(episode_rewards["seeker"]),
        "seeker_reward_std": np.std(episode_rewards["seeker"]),
        "seeker_reward_min": np.min(episode_rewards["seeker"]),
        "seeker_reward_max": np.max(episode_rewards["seeker"]),
        "episode_length_mean": np.mean(episode_lengths),
        "episode_length_std": np.std(episode_lengths),
        "hider_found_rate": hider_found_count / n_episodes,
        "hider_in_room_mean": np.mean(hider_in_room_steps),
        "hider_in_room_ratio": np.mean(hider_in_room_steps) / np.mean(episode_lengths)
    }
    
    return results


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nEpisodes: {results['n_episodes']}")
    
    print(f"\n📦 HIDER Performance:")
    print(f"  Reward: {results['hider_reward_mean']:.2f} ± {results['hider_reward_std']:.2f}")
    print(f"  Range: [{results['hider_reward_min']:.2f}, {results['hider_reward_max']:.2f}]")
    print(f"  In-Room Ratio: {results['hider_in_room_ratio']:.1%}")
    
    print(f"\n🔍 SEEKER Performance:")
    print(f"  Reward: {results['seeker_reward_mean']:.2f} ± {results['seeker_reward_std']:.2f}")
    print(f"  Range: [{results['seeker_reward_min']:.2f}, {results['seeker_reward_max']:.2f}]")
    print(f"  Find Rate: {results['hider_found_rate']:.1%}")
    
    print(f"\n📊 EPISODE Statistics:")
    print(f"  Length: {results['episode_length_mean']:.1f} ± {results['episode_length_std']:.1f}")
    
    print("\n" + "="*60)


def compare_checkpoints(agent, env, checkpoint_paths, n_episodes=50):
    """
    Compare multiple checkpoints.
    
    Args:
        agent: MAPPOAgent instance
        env: Environment instance
        checkpoint_paths: List of checkpoint paths
        n_episodes: Episodes per checkpoint
    """
    comparison_results = []
    
    for ckpt_path in checkpoint_paths:
        print(f"\n{'='*60}")
        print(f"Evaluating: {os.path.basename(ckpt_path)}")
        print(f"{'='*60}")
        
        checkpoint = load_checkpoint(agent, ckpt_path)
        results = evaluate_agent(agent, env, n_episodes=n_episodes)
        
        results['checkpoint'] = os.path.basename(ckpt_path)
        results['episode'] = checkpoint.get('episode', 'unknown')
        results['total_steps'] = checkpoint.get('total_steps', 'unknown')
        
        print_results(results)
        comparison_results.append(results)
    
    # Print comparison table
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON")
    print("="*80)
    print(f"{'Checkpoint':<30} {'Hider Reward':<15} {'Seeker Reward':<15} {'Find Rate':<10}")
    print("-"*80)
    
    for res in comparison_results:
        print(f"{res['checkpoint']:<30} "
              f"{res['hider_reward_mean']:>6.2f} ± {res['hider_reward_std']:<5.2f} "
              f"{res['seeker_reward_mean']:>6.2f} ± {res['seeker_reward_std']:<5.2f} "
              f"{res['hider_found_rate']:>8.1%}")
    
    print("="*80 + "\n")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO agents")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (or directory for comparison)')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions (default: False)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed episode information')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple checkpoints in directory')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize environment and agent
    print("Initializing environment and agent...")
    env = HideAndSeekEnv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = MAPPOAgent(grid_size=20, action_dim=10, device=device)
    
    if args.compare:
        # Compare multiple checkpoints
        if os.path.isdir(args.checkpoint):
            checkpoint_paths = [
                os.path.join(args.checkpoint, f)
                for f in sorted(os.listdir(args.checkpoint))
                if f.endswith('.pt')
            ]
        else:
            print("Error: --compare requires a directory path")
            return
        
        results = compare_checkpoints(agent, env, checkpoint_paths, args.n_episodes)
        
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.save_results}")
    
    else:
        # Evaluate single checkpoint
        load_checkpoint(agent, args.checkpoint)
        results = evaluate_agent(
            agent, env,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            verbose=args.verbose
        )
        
        print_results(results)
        
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()