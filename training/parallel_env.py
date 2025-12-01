"""
parallel_env.py

Parallel environment wrapper for running multiple hide and seek environments.
Optimized for multi-core systems.
"""

import numpy as np
from multiprocessing import Process, Pipe
import copy


def worker(remote, parent_remote, env_fn):
    """
    Worker process for running a single environment.
    
    Args:
        remote: Remote connection for this worker
        parent_remote: Parent connection
        env_fn: Function to create environment
    """
    parent_remote.close()
    env = env_fn()
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, done, rewards = env.step(data)
                if done:
                    # Auto-reset on episode end
                    obs = env.reset()
                remote.send((obs, done, rewards))
            
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            
            elif cmd == 'close':
                remote.close()
                break
            
            elif cmd == 'get_env':
                # Send environment state (for observation processing)
                remote.send(env)
            
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
    
    except KeyboardInterrupt:
        print("Worker interrupted")
    finally:
        env.close() if hasattr(env, 'close') else None


class ParallelEnv:
    """
    Parallel environment wrapper that runs multiple environments in separate processes.
    """
    def __init__(self, env_fns):
        """
        Initialize parallel environments.
        
        Args:
            env_fns: List of functions that create environments
        """
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        
        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get initial environment for metadata
        self.remotes[0].send(('get_env', None))
        self.env_sample = self.remotes[0].recv()
    
    def step_async(self, actions):
        """
        Send step commands to all environments asynchronously.
        
        Args:
            actions: List of action dicts for each environment
        """
        if self.waiting:
            raise RuntimeError("Already waiting for step results")
        
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        self.waiting = True
    
    def step_wait(self):
        """
        Wait for step results from all environments.
        
        Returns:
            observations: List of observations
            dones: List of done flags
            rewards: List of reward dicts
        """
        if not self.waiting:
            raise RuntimeError("Not waiting for step results")
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        observations, dones, rewards = zip(*results)
        
        return list(observations), list(dones), list(rewards)
    
    def step(self, actions):
        """
        Perform a step in all environments.
        
        Args:
            actions: List of action dicts for each environment
            
        Returns:
            observations: List of observations
            dones: List of done flags
            rewards: List of reward dicts
        """
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        """
        Reset all environments.
        
        Returns:
            observations: List of initial observations
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        observations = [remote.recv() for remote in self.remotes]
        return observations
    
    def close(self):
        """Close all environments and worker processes."""
        if self.closed:
            return
        
        if self.waiting:
            # Wait for pending operations
            self.step_wait()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()
        
        self.closed = True
    
    def __len__(self):
        """Return number of parallel environments."""
        return self.n_envs
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if not self.closed:
            self.close()


def make_parallel_env(env_fn, n_envs):
    """
    Create parallel environments.
    
    Args:
        env_fn: Function to create a single environment
        n_envs: Number of parallel environments
        
    Returns:
        parallel_env: ParallelEnv instance
    """
    env_fns = [env_fn for _ in range(n_envs)]
    return ParallelEnv(env_fns)


if __name__ == "__main__":
    # Test parallel environments
    from env.hide_and_seek_env import HideAndSeekEnv
    
    def env_fn():
        return HideAndSeekEnv()
    
    print("Creating 4 parallel environments...")
    parallel_env = make_parallel_env(env_fn, n_envs=4)
    
    print("Resetting environments...")
    obs_list = parallel_env.reset()
    print(f"Got {len(obs_list)} observations")
    
    print("Taking random steps...")
    for _ in range(5):
        actions = [
            {
                "hider": np.random.randint(0, 10),
                "seeker": np.random.randint(0, 10)
            }
            for _ in range(4)
        ]
        obs_list, dones, rewards = parallel_env.step(actions)
        print(f"Step completed. Dones: {dones}")
    
    print("Closing environments...")
    parallel_env.close()
    print("Test complete!")