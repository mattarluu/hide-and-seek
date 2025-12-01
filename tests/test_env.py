import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.hide_and_seek_env import HideAndSeekEnv
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from training.rl_agent import DQNAgent
from utils.visualization import render_environment
from utils.logger import log_info

# Test parameters
num_test_episodes = 5  # Number of episodes to visualize (reduced since they're longer now)
# No max_steps limit - episodes run until done (capture or 200 steps)
output_video_path = "test_video.mp4"  # Output video file
output_gif_path = "test_animation.gif"  # Output GIF file

# Initialize environment
env = HideAndSeekEnv()
state_dim = 4  # (x, y, direction, z)
agent_action_dim = 10  # 10 actions now

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Testing on device: {device}")

# Initialize DQN agents for testing (without any further training)
seeker_agent = DQNAgent(state_dim, agent_action_dim, device=device)
hider_agent = DQNAgent(state_dim, agent_action_dim, device=device)

# Load pre-trained models (ensure these are the final weights you want to test)
seeker_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\seeker_dqn_model.pth')
hider_agent.load_model(r'C:\Users\User\Projects\NPUA\multi_agent_hide_and_seek\training\models\hider_dqn_model.pth')

# Move networks to device
seeker_agent.q_network.to(device)
hider_agent.q_network.to(device)

# Do not update target networks or call train_step() in test mode.
# In test mode, we use a greedy policy (i.e. ε = 0) and do not perform any learning.

fig, ax = plt.subplots(figsize=(10, 10))
frames = []

for episode in range(num_test_episodes):
    state = env.reset()
    # Convert states; note: seeker state may be dummy until activated.
    seeker_state = torch.tensor(state['seeker']['state'], dtype=torch.float32).to(device)
    hider_state = torch.tensor(state['hider']['state'], dtype=torch.float32).to(device)

    total_reward_seeker = 0
    total_reward_hider = 0

    log_info(f"Test Episode {episode + 1} started.")

    step = 0
    done = False
    while not done:
        step += 1
        
        # Select actions using a greedy policy (ε=0)
        if env.seeker_active:
            action_seeker = seeker_agent.select_action(seeker_state.cpu().numpy())
        else:
            action_seeker = None  # No action if seeker not yet active

        action_hider = hider_agent.select_action(hider_state.cpu().numpy())

        actions = {"seeker": action_seeker, "hider": action_hider}

        next_state, done, step_rewards = env.step(actions)

        # Render the environment and capture an RGB frame.
        render_environment(ax, env)
        plt.pause(0.01)  # 100 FPS - Very fast visualization
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        frame = frame.reshape((height, width, 4))
        # Convert ARGB to RGB by dropping the alpha channel.
        frame = frame[:, :, 1:4]
        frames.append(frame)

        next_seeker_state = torch.tensor(next_state['seeker']['state'], dtype=torch.float32).to(device)
        next_hider_state = torch.tensor(next_state['hider']['state'], dtype=torch.float32).to(device)

        # Use step_rewards
        reward_seeker = step_rewards.get("seeker", 0.0)
        reward_hider = step_rewards.get("hider", 0.0)
        total_reward_seeker += reward_seeker
        total_reward_hider += reward_hider

        seeker_state = next_seeker_state
        hider_state = next_hider_state

    # Determine winner and victory type
    if total_reward_seeker > total_reward_hider:
        winner = "SEEKER"
        if step < env.max_steps:
            victory_type = "CAPTURE"
        else:
            victory_type = "VISIBILITY"
    else:
        winner = "HIDER"
        victory_type = "EVASION"
    
    log_info(f"Test Episode {episode + 1} finished after {step} steps.")
    log_info(f"  Winner: {winner} ({victory_type})")
    log_info(f"  Rewards - Seeker: {total_reward_seeker:.1f}, Hider: {total_reward_hider:.1f}")
    
    # Show visibility stats if seeker was active
    if env.seeker_active:
        seeking_steps = step - env.hiding_phase_steps
        visibility_ratio = env.visibility_count / max(seeking_steps, 1) * 100
        log_info(f"  Visibility: {env.visibility_count}/{seeking_steps} steps ({visibility_ratio:.1f}%)")

# Save frames as a video (RGB)
log_info("Saving video...")
imageio.mimsave(output_video_path, frames, fps=10)

# Save frames as a GIF
log_info("Saving GIF...")
imageio.mimsave(output_gif_path, frames, duration=0.5)

plt.close()
log_info("Testing complete. Video and GIF saved.")