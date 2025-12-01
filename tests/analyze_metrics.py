"""
analyze_metrics.py

Simple script to analyze training metrics from CSV logs.
Shows basic statistics and creates summary plots.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse


def load_latest_metrics(log_dir='./logs'):
    """Load the most recent metrics CSV file."""
    csv_files = glob.glob(os.path.join(log_dir, 'metrics_*.csv'))
    
    if not csv_files:
        print(f"❌ No metrics files found in {log_dir}")
        print("Make sure training has started and created log files.")
        return None
    
    # Get most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📊 Loading: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"✅ Loaded {len(df)} training updates")
    return df


def print_summary(df):
    """Print summary statistics."""
    if df is None or len(df) == 0:
        return
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Latest values
    latest = df.iloc[-1]
    print(f"\n📈 Current Progress:")
    print(f"  Step: {int(latest['step']):,}")
    print(f"  Curriculum Stage: {int(latest['curriculum_stage'])}")
    
    print(f"\n🎮 Episode Performance (latest):")
    if latest['n_episodes'] > 0:
        print(f"  Hider Reward: {latest['hider_reward_mean']:.2f}")
        print(f"  Seeker Reward: {latest['seeker_reward_mean']:.2f}")
        print(f"  Episode Length: {latest['episode_length_mean']:.1f}")
    
    print(f"\n🎓 Training Metrics (latest):")
    print(f"  Policy Loss: {latest['policy_loss']:.6f}")
    print(f"  Value Loss: {latest['value_loss']:.6f}")
    print(f"  Entropy: {latest['entropy']:.6f}")
    
    # Evaluation results (if available)
    eval_df = df[df['eval_hider_reward_mean'] != 0]
    if len(eval_df) > 0:
        latest_eval = eval_df.iloc[-1]
        print(f"\n⭐ Evaluation (latest):")
        print(f"  Hider Reward: {latest_eval['eval_hider_reward_mean']:.2f}")
        print(f"  Seeker Reward: {latest_eval['eval_seeker_reward_mean']:.2f}")
        print(f"  Episode Length: {latest_eval['eval_episode_length_mean']:.1f}")
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    reward_df = df[df['n_episodes'] > 0]
    if len(reward_df) > 0:
        print(f"  Hider Reward (mean): {reward_df['hider_reward_mean'].mean():.2f} ± {reward_df['hider_reward_mean'].std():.2f}")
        print(f"  Seeker Reward (mean): {reward_df['seeker_reward_mean'].mean():.2f} ± {reward_df['seeker_reward_mean'].std():.2f}")
    
    print("="*60 + "\n")


def plot_rewards(df, save_path=None):
    """Plot training rewards over time."""
    if df is None or len(df) == 0:
        return
    
    # Filter to only rows with episode data
    plot_df = df[df['n_episodes'] > 0].copy()
    
    if len(plot_df) == 0:
        print("No episode data to plot yet")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training rewards
    ax1.plot(plot_df['step'], plot_df['hider_reward_mean'], 
             label='Hider', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(plot_df['step'], plot_df['seeker_reward_mean'], 
             label='Seeker', color='red', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Evaluation rewards (if available)
    eval_df = df[df['eval_hider_reward_mean'] != 0]
    if len(eval_df) > 0:
        ax2.plot(eval_df['step'], eval_df['eval_hider_reward_mean'], 
                label='Hider', color='blue', marker='o', markersize=6)
        ax2.plot(eval_df['step'], eval_df['eval_seeker_reward_mean'], 
                label='Seeker', color='red', marker='s', markersize=6)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Evaluation Reward', fontsize=12)
        ax2.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No evaluation data yet', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Rewards plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_losses(df, save_path=None):
    """Plot training losses over time."""
    if df is None or len(df) == 0:
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Policy Loss
    ax1.plot(df['step'], df['policy_loss'], color='green', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Policy Loss', fontsize=11)
    ax1.set_title('Policy Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Value Loss
    ax2.plot(df['step'], df['value_loss'], color='orange', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('Value Loss', fontsize=11)
    ax2.set_title('Value Loss', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Entropy
    ax3.plot(df['step'], df['entropy'], color='purple', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Entropy', fontsize=11)
    ax3.set_title('Policy Entropy', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Losses plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze MAPPO training metrics")
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory containing metrics CSV files')
    parser.add_argument('--save-rewards', type=str, default=None,
                       help='Save rewards plot to file')
    parser.add_argument('--save-losses', type=str, default=None,
                       help='Save losses plot to file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plotting, just show summary')
    
    args = parser.parse_args()
    
    # Load metrics
    df = load_latest_metrics(args.log_dir)
    
    if df is None:
        return
    
    # Print summary
    print_summary(df)
    
    # Create plots
    if not args.no_plots:
        print("📊 Generating plots...")
        plot_rewards(df, args.save_rewards)
        plot_losses(df, args.save_losses)
        print("✅ Done!")


if __name__ == "__main__":
    main()