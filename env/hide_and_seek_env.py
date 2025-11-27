"""
env/hide_and_seek_env.py

UPDATED VERSION with:
- Height system (z=0: ground, z=1: ramp/blocks/walls)
- Grab mechanics for moving objects (unified for both blocks and ramp)
- Lock mechanics (hider can lock blocks)
- Automatic falling physics
- Climbing: Only via ramp, but can walk on all z=1 surfaces

Action space (10 actions):
0-3: Move (up, right, down, left)
4-7: Grab/move object while holding (up, right, down, left)
8: Climb (use ramp to go up)
9: Lock/Unlock block (hider only)
"""

import gym
from gym import spaces
import numpy as np
import torch

from env.room import Room
from env.objects import Block, Ramp
from agents.seeker import Seeker
from agents.hider import Hider
from utils.logger import log_debug, log_info

class HideAndSeekEnv(gym.Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()

        # Grid dimensions - 20x20
        self.grid_size = 20

        # Initialize 8x8 room in bottom right corner
        self.room = Room(top_left=(12, 12), width=8, height=8)

        # Initialize movable objects
        self.blocks = []
        self.ramp = None
        self._initialize_objects()

        # Define observation space: (x, y, direction, z)
        self.observation_space = spaces.Dict({
            "seeker": spaces.Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32),
            "hider": spaces.Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32)
        })

        # Action space: 10 actions now
        # 0-3: move, 4-7: grab+move, 8: climb, 9: lock/unlock
        self.action_space = spaces.Dict({
            "seeker": spaces.Discrete(10),
            "hider": spaces.Discrete(10)
        })

        # Movement mappings
        self.moves = {
            0: (0, -1),  # up
            1: (1, 0),   # right
            2: (0, 1),   # down
            3: (-1, 0)   # left
        }

        self.max_steps = 100
        self.step_count = 0

        # Agents
        self.hider = None
        self.seeker = None
        self.seeker_active = False

        # Grabbed objects tracking
        self.hider_grabbed = None  # None or ('block', index) or ('ramp',)
        self.seeker_grabbed = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _initialize_objects(self):
        """Initialize 4 blocks and 1 ramp at random positions."""
        self.blocks = []
        self.ramp = None
        
        # Place 4 blocks
        for i in range(4):
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                pos = (x, y)
                
                if not self.room.is_wall(pos):
                    if not any(block.position == pos for block in self.blocks):
                        if self.ramp is None or self.ramp.position != pos:
                            self.blocks.append(Block(pos))
                            break
        
        # Place 1 ramp
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            pos = (x, y)
            
            if not self.room.is_wall(pos):
                if not any(block.position == pos for block in self.blocks):
                    self.ramp = Ramp(pos)
                    break
    
    def get_height_at(self, x, y):
        """
        Get the height of the surface at position (x, y).
        Returns:
            0: Empty ground
            1: Ramp, Block, or Wall present (all same height)
        """
        # Check if it's a room wall
        if self.room.is_wall((x, y)):
            return 1
        
        # Check if there's a block
        if any(block.position == (x, y) for block in self.blocks):
            return 1
        
        # Check if there's a ramp
        if self.ramp and self.ramp.position == (x, y):
            return 1
        
        return 0  # Ground level
    
    def can_climb(self, agent_pos):
        """
        Check if agent can climb from current position.
        Agent must be on ground (z=0) next to ramp to climb.
        Can only climb RAMP, not blocks or walls directly.
        
        Returns:
            (can_climb: bool, target_position: tuple or None)
        """
        ax, ay, az = agent_pos
        
        # Must be on ground to climb
        if az != 0:
            return False, None
        
        # Look for ramp at current position or adjacent
        ramp_positions = []
        
        # Check current position
        if self.ramp and self.ramp.position == (ax, ay):
            ramp_positions.append((ax, ay))
        
        # Check adjacent positions
        for dx, dy in [(0,-1), (1,0), (0,1), (-1,0)]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.ramp and self.ramp.position == (nx, ny):
                    ramp_positions.append((nx, ny))
        
        if len(ramp_positions) > 0:
            return True, ramp_positions[0]
        
        return False, None
    
    def check_fall(self, x, y, z):
        """
        Check if agent should fall due to lack of support.
        
        Returns:
            new_z: int (height after potential fall)
        """
        if z == 0:
            return 0  # Already on ground
        
        current_height = self.get_height_at(x, y)
        
        # If no support or support is lower than current height, fall
        if current_height < z:
            return current_height
        
        return z  # No fall needed
    
    def get_grabbed_object(self, agent_key):
        """Get the object currently grabbed by an agent."""
        if agent_key == "hider":
            grabbed = self.hider_grabbed
        else:
            grabbed = self.seeker_grabbed
        
        if grabbed is None:
            return None
        
        if grabbed[0] == 'block':
            return self.blocks[grabbed[1]]
        elif grabbed[0] == 'ramp':
            return self.ramp
        
        return None
    
    def try_grab_object(self, agent_key, agent_pos):
        """
        Try to grab an object at or adjacent to agent position.
        
        Returns:
            (success: bool, object_type: str, object_index: int or None)
        """
        ax, ay, az = agent_pos
        
        # Check for objects at current position and adjacent positions
        check_positions = [(ax, ay)]
        for dx, dy in [(0,-1), (1,0), (0,1), (-1,0)]:
            check_positions.append((ax + dx, ay + dy))
        
        for pos in check_positions:
            # Check blocks
            for i, block in enumerate(self.blocks):
                if block.position == pos:
                    # Can't grab if locked and not hider
                    if block.locked and agent_key != "hider":
                        continue
                    # Can't grab if already grabbed by someone else
                    if block.grabbed_by is not None and block.grabbed_by != agent_key:
                        continue
                    return True, 'block', i
            
            # Check ramp
            if self.ramp and self.ramp.position == pos:
                # Can't grab if already grabbed by someone else
                if self.ramp.grabbed_by is not None and self.ramp.grabbed_by != agent_key:
                    continue
                return True, 'ramp', None
        
        return False, None, None

    def compute_visible_cells(self, state, max_distance=15, num_rays=7):
        """
        Compute visible cells using triangular FOV.
        State is now (x, y, direction, z).
        """
        x, y, d, z = state
        visible = {(x, y)}
        
        for i in range(1, max_distance + 1):
            if d == 0:  # facing up
                row_y = y - i
                for dx in range(-i, i + 1):
                    cell = (x + dx, row_y)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self._blocks_vision(cell):
                        break
            elif d == 1:  # facing right
                row_x = x + i
                for dy in range(-i, i + 1):
                    cell = (row_x, y + dy)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self._blocks_vision(cell):
                        break
            elif d == 2:  # facing down
                row_y = y + i
                for dx in range(-i, i + 1):
                    cell = (x + dx, row_y)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self._blocks_vision(cell):
                        break
            elif d == 3:  # facing left
                row_x = x - i
                for dy in range(-i, i + 1):
                    cell = (row_x, y + dy)
                    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
                        continue
                    visible.add(cell)
                    if self._blocks_vision(cell):
                        break
        
        return list(visible)
    
    def _blocks_vision(self, cell):
        """Check if a cell blocks vision."""
        if self.room.blocks_vision(cell):
            return True
        if any(block.position == cell for block in self.blocks):
            return True
        if self.ramp and self.ramp.position == cell:
            return True
        return False

    def reset(self):
        self.step_count = 0
        self.seeker_active = False
        
        # Reinitialize objects
        self._initialize_objects()
        
        # Reset grabbed states
        self.hider_grabbed = None
        self.seeker_grabbed = None

        def get_valid_position():
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                pos = (x, y)
                if not self.room.is_wall(pos):
                    if not any(block.position == pos for block in self.blocks):
                        if not self.ramp or self.ramp.position != pos:
                            return x, y

        # Spawn hider at ground level
        hx, hy = get_valid_position()
        hider_dir = np.random.randint(0, 4)
        self.hider = Hider(hx, hy, hider_dir, z=0)
        self.seeker = None

        return {
            "seeker": {
                "state": (-1, -1, -1, -1),
                "visible": []
            },
            "hider": {
                "state": self.hider.get_state(),
                "visible": self.compute_visible_cells(self.hider.get_state())
            }
        }

    def spawn_seeker(self):
        def get_valid_position():
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                pos = (x, y)
                if not self.room.is_wall(pos):
                    if not any(block.position == pos for block in self.blocks):
                        if not self.ramp or self.ramp.position != pos:
                            if (x, y) != (self.hider.x, self.hider.y):
                                return x, y
        
        sx, sy = get_valid_position()
        seeker_dir = np.random.randint(0, 4)
        self.seeker = Seeker(sx, sy, seeker_dir, z=0)

    def is_valid_move(self, x, y, dx, dy):
        """Check if movement is valid (not out of bounds, not into wall at ground level)."""
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return False
        
        new_pos = (new_x, new_y)
        
        # Can't move into wall from ground level (but can walk on walls at z=1)
        if self.room.is_wall(new_pos):
            return False
        
        # CAN move into blocks and ramps from ground (will stay at z=0)
        # The height change happens in the step logic
            
        return True

    def step(self, actions):
        self.step_count += 1
        rewards = {"seeker": 0.0, "hider": 0.0}

        # Spawn seeker after 10 steps
        if not self.seeker_active and self.step_count >= 10:
            self.spawn_seeker()
            self.seeker_active = True

        # Process actions for both agents
        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
            if agent_key == "seeker" and not self.seeker_active:
                continue
            
            agent = self.seeker if agent_key == "seeker" else self.hider
            x, y, d, z = agent.get_state()
            
            # ACTION 9: Lock/Unlock block (hider only)
            if isinstance(action, int) and action == 9 and agent_key == "hider":
                # Check for blocks at current position or adjacent
                for block in self.blocks:
                    bx, by = block.position
                    if (bx, by) == (x, y) or abs(bx - x) + abs(by - y) == 1:
                        if block.locked:
                            block.unlock()
                            log_info(f"Hider unlocked block at {block.position}")
                        else:
                            block.lock()
                            log_info(f"Hider locked block at {block.position}")
                        break
                continue
            
            # ACTION 8: Climb
            if isinstance(action, int) and action == 8:
                if z == 0:
                    # On ground - try to climb onto ramp
                    can_climb, ramp_pos = self.can_climb((x, y, z))
                    if can_climb and ramp_pos:
                        rx, ry = ramp_pos
                        # Move onto ramp and elevate to z=1
                        agent.update_state(x=rx, y=ry, z=1)
                        log_info(f"{agent_key} climbed onto ramp at ({rx},{ry}), now at z=1")
                
                # If already at z=1, climb does nothing (already elevated)
                # Can't climb higher than z=1
                
                continue
            
            # ACTIONS 0-3: Normal movement
            if isinstance(action, int) and action in self.moves:
                dx, dy = self.moves[action]
                new_x, new_y = x + dx, y + dy
                
                # Check bounds
                if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                    continue
                
                new_pos = (new_x, new_y)
                target_height = self.get_height_at(new_x, new_y)
                
                # Movement logic based on current height
                if z == 0:
                    # Ground level - normal movement
                    # Can't walk into walls, blocks, or ramps from ground
                    if self.room.is_wall(new_pos):
                        continue
                    if any(block.position == new_pos for block in self.blocks):
                        continue
                    if self.ramp and self.ramp.position == new_pos:
                        continue
                    
                    agent.update_state(x=new_x, y=new_y, direction=action, z=0)
                
                elif z == 1:
                    # Elevated (on ramp/block/wall) - can move to any z=1 surface
                    if target_height == 1:
                        # Move to another elevated surface
                        agent.update_state(x=new_x, y=new_y, direction=action, z=1)
                    else:
                        # Moving to ground - fall
                        agent.update_state(x=new_x, y=new_y, direction=action, z=0)
                        log_info(f"{agent_key} fell from z=1 to ground")
            
            # ACTIONS 4-7: Grab and move object
            elif isinstance(action, int) and 4 <= action <= 7:
                # Only works from ground level
                if z != 0:
                    continue
                
                grabbed = self.hider_grabbed if agent_key == "hider" else self.seeker_grabbed
                
                if grabbed is None:
                    # Try to grab an object
                    success, obj_type, obj_idx = self.try_grab_object(agent_key, (x, y, z))
                    if success:
                        if obj_type == 'block':
                            self.blocks[obj_idx].grab(agent_key)
                            if agent_key == "hider":
                                self.hider_grabbed = ('block', obj_idx)
                            else:
                                self.seeker_grabbed = ('block', obj_idx)
                            log_info(f"{agent_key} grabbed block at {self.blocks[obj_idx].position}")
                        elif obj_type == 'ramp':
                            self.ramp.grab(agent_key)
                            if agent_key == "hider":
                                self.hider_grabbed = ('ramp',)
                            else:
                                self.seeker_grabbed = ('ramp',)
                            log_info(f"{agent_key} grabbed ramp at {self.ramp.position}")
                else:
                    # Move with grabbed object
                    obj = self.get_grabbed_object(agent_key)
                    if obj is None:
                        continue
                    
                    # Get movement direction from action (4=up, 5=right, 6=down, 7=left)
                    move_action = action - 4
                    dx, dy = self.moves[move_action]
                    new_x, new_y = x + dx, y + dy
                    
                    # Check if movement is valid
                    if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                        continue
                    
                    new_agent_pos = (new_x, new_y)
                    ox, oy = obj.position
                    
                    # Calculate new object position (can be different from agent position)
                    # Object moves in the same direction as agent
                    new_obj_x, new_obj_y = ox + dx, oy + dy
                    
                    # Validate object movement
                    if not (0 <= new_obj_x < self.grid_size and 0 <= new_obj_y < self.grid_size):
                        # Can't move object out of bounds - release it
                        obj.release()
                        if agent_key == "hider":
                            self.hider_grabbed = None
                        else:
                            self.seeker_grabbed = None
                        log_info(f"{agent_key} released object (out of bounds)")
                        continue
                    
                    # Check if new object position is valid
                    new_obj_pos = (new_obj_x, new_obj_y)
                    
                    # Can't place on walls
                    if self.room.is_wall(new_obj_pos):
                        obj.release()
                        if agent_key == "hider":
                            self.hider_grabbed = None
                        else:
                            self.seeker_grabbed = None
                        log_info(f"{agent_key} released object (wall collision)")
                        continue
                    
                    # Check collision with other objects
                    collision = False
                    for block in self.blocks:
                        if block != obj and block.position == new_obj_pos:
                            collision = True
                            break
                    if not collision and self.ramp and self.ramp != obj and self.ramp.position == new_obj_pos:
                        collision = True
                    
                    if collision:
                        # Release object
                        obj.release()
                        if agent_key == "hider":
                            self.hider_grabbed = None
                        else:
                            self.seeker_grabbed = None
                        log_info(f"{agent_key} released object (object collision)")
                        continue
                    
                    # Check if agent can move to new position
                    if self.room.is_wall(new_agent_pos):
                        continue
                    
                    # Move both agent and object
                    obj.move(new_obj_pos)
                    agent.update_state(x=new_x, y=new_y, direction=move_action, z=0)
                    log_info(f"{agent_key} moved with object to ({new_x},{new_y}), object at ({new_obj_x},{new_obj_y})")
        
        # Apply gravity/falling for both agents
        for agent_key in ["seeker", "hider"]:
            if agent_key == "seeker" and not self.seeker_active:
                continue
            agent = self.seeker if agent_key == "seeker" else self.hider
            x, y, d, z = agent.get_state()
            new_z = self.check_fall(x, y, z)
            if new_z != z:
                agent.update_state(z=new_z)
                log_info(f"{agent_key} fell from z={z} to z={new_z}")
        
        # Vision-based rewards
        if self.seeker_active:
            visible_seeker = set(self.compute_visible_cells(self.seeker.get_state()))
            hider_pos = self.hider.get_state()[:2]
            if hider_pos in visible_seeker:
                rewards["seeker"] += 1.0
                rewards["hider"] += -1.0
        
        # In-room reward
        hider_pos = self.hider.get_state()[:2]
        if self.room.is_inside(hider_pos):
            rewards["hider"] += 1.0
        
        done = (self.step_count >= self.max_steps)
        
        obs = {
            "seeker": {
                "state": self.seeker.get_state() if self.seeker_active else (-1, -1, -1, -1),
                "visible": self.compute_visible_cells(self.seeker.get_state()) if self.seeker_active else []
            },
            "hider": {
                "state": self.hider.get_state(),
                "visible": self.compute_visible_cells(self.hider.get_state())
            }
        }
        
        return obs, done, rewards

    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    env = HideAndSeekEnv()
    observation = env.reset()
    log_info("Environment initialized with height system!")
    log_info(f"Hider state: {env.hider.get_state()}")
    log_info("System: z=0 (ground), z=1 (ramp/blocks/walls - can only climb ramp)")