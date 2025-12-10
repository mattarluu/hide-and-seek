"""
env/hide_and_seek_env.py

UPDATED VERSION with:
- FIX: Infinite loop safety (max_attempts + fallback scan)
- FIX: spawn_seeker logic was hardcoded to (0,0) causing deadlock
- Height system (z=0: ground, z=1: ramp/blocks/walls)
- Grab mechanics for moving objects (unified for both blocks and ramp)
- Lock mechanics (hider can lock blocks)
- Automatic falling physics
- Climbing: Only via ramp, but can walk on all z=1 surfaces
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
        
        # Safe initialization
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

        self.max_steps = 200  # Total episode length
        self.hiding_phase_steps = 40  # Steps for hider to hide before seeker appears
        self.step_count = 0
        
        # Victory conditions
        self.visibility_count = 0  # How many steps seeker has seen hider
        self.capture_distance = 1  # Distance for physical capture

        # Agents
        self.hider = None
        self.seeker = None
        self.seeker_active = False

        # Grabbed objects tracking
        self.hider_grabbed = None  # None or ('block', index) or ('ramp',)
        self.seeker_grabbed = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _find_valid_spawn_pos(self, extra_exclusions=None):
        """
        Robust method to find a free cell. 
        First tries random positions (fast). 
        If that fails, scans the grid linearly (guaranteed).
        """
        if extra_exclusions is None:
            extra_exclusions = []
            
        def is_free(tx, ty):
            pos = (tx, ty)
            # Check walls
            if self.room.is_wall(pos): return False
            # Check blocks
            if any(b.position == pos for b in self.blocks): return False
            # Check ramp
            if self.ramp and self.ramp.position == pos: return False
            # Check extra exclusions (e.g. other agent)
            if pos in extra_exclusions: return False
            return True

        # Method 1: Random attempts (Fast average case)
        for _ in range(100):
            rx = np.random.randint(0, self.grid_size)
            ry = np.random.randint(0, self.grid_size)
            if is_free(rx, ry):
                return rx, ry
        
        # Method 2: Linear Scan (Fallback guarantee)
        # Scan from a random offset to keep some randomness even in fallback
        start_x = np.random.randint(0, self.grid_size)
        start_y = np.random.randint(0, self.grid_size)
        
        for i in range(self.grid_size):
            x = (start_x + i) % self.grid_size
            for j in range(self.grid_size):
                y = (start_y + j) % self.grid_size
                if is_free(x, y):
                    return x, y
        
        # Emergency fallback (should never happen in 20x20 with few objects)
        return 0, 0

    def _initialize_objects(self):
        """Initialize 4 blocks and 1 ramp at random positions."""
        self.blocks = []
        self.ramp = None
        
        # Place 4 blocks
        for _ in range(4):
            bx, by = self._find_valid_spawn_pos()
            self.blocks.append(Block((bx, by)))
        
        # Place 1 ramp
        rx, ry = self._find_valid_spawn_pos()
        self.ramp = Ramp((rx, ry))
    
    def get_height_at(self, x, y):
        """
        Get the height of the surface at position (x, y).
        Returns:
            0: Empty ground
            1: Ramp, Block, or Wall present (all same height)
        """
        if self.room.is_wall((x, y)): return 1
        if any(block.position == (x, y) for block in self.blocks): return 1
        if self.ramp and self.ramp.position == (x, y): return 1
        return 0
    
    def can_climb(self, agent_pos):
        ax, ay, az = agent_pos
        if az != 0: return False, None
        
        ramp_positions = []
        if self.ramp and self.ramp.position == (ax, ay):
            ramp_positions.append((ax, ay))
        
        for dx, dy in [(0,-1), (1,0), (0,1), (-1,0)]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.ramp and self.ramp.position == (nx, ny):
                    ramp_positions.append((nx, ny))
        
        if len(ramp_positions) > 0:
            return True, ramp_positions[0]
        return False, None
    
    def check_fall(self, x, y, z):
        if z == 0: return 0
        current_height = self.get_height_at(x, y)
        if current_height < z:
            return current_height
        return z
    
    def get_grabbed_object(self, agent_key):
        if agent_key == "hider":
            grabbed = self.hider_grabbed
        else:
            grabbed = self.seeker_grabbed
        
        if grabbed is None: return None
        
        if grabbed[0] == 'block':
            return self.blocks[grabbed[1]]
        elif grabbed[0] == 'ramp':
            return self.ramp
        return None
    
    def try_grab_object(self, agent_key, agent_pos):
        ax, ay, az = agent_pos
        check_positions = [(ax, ay)]
        for dx, dy in [(0,-1), (1,0), (0,1), (-1,0)]:
            check_positions.append((ax + dx, ay + dy))
        
        for pos in check_positions:
            for i, block in enumerate(self.blocks):
                if block.position == pos:
                    if block.locked and agent_key != "hider": continue
                    if block.grabbed_by is not None and block.grabbed_by != agent_key: continue
                    return True, 'block', i
            
            if self.ramp and self.ramp.position == pos:
                if self.ramp.grabbed_by is not None and self.ramp.grabbed_by != agent_key: continue
                return True, 'ramp', None
        return False, None, None

    def compute_visible_cells(self, state, max_distance=15):
        x, y, d, z = state
        visible = {(x, y)}
        
        if d == 0:  # up
            for dist in range(1, max_distance + 1):
                target_y = y - dist
                if target_y < 0: break
                for dx in range(-dist, dist + 1):
                    target_x = x + dx
                    if not (0 <= target_x < self.grid_size): continue
                    if self._can_see_cell(x, y, target_x, target_y): visible.add((target_x, target_y))
        
        elif d == 1:  # right
            for dist in range(1, max_distance + 1):
                target_x = x + dist
                if target_x >= self.grid_size: break
                for dy in range(-dist, dist + 1):
                    target_y = y + dy
                    if not (0 <= target_y < self.grid_size): continue
                    if self._can_see_cell(x, y, target_x, target_y): visible.add((target_x, target_y))
        
        elif d == 2:  # down
            for dist in range(1, max_distance + 1):
                target_y = y + dist
                if target_y >= self.grid_size: break
                for dx in range(-dist, dist + 1):
                    target_x = x + dx
                    if not (0 <= target_x < self.grid_size): continue
                    if self._can_see_cell(x, y, target_x, target_y): visible.add((target_x, target_y))
        
        elif d == 3:  # left
            for dist in range(1, max_distance + 1):
                target_x = x - dist
                if target_x < 0: break
                for dy in range(-dist, dist + 1):
                    target_y = y + dy
                    if not (0 <= target_y < self.grid_size): continue
                    if self._can_see_cell(x, y, target_x, target_y): visible.add((target_x, target_y))
        
        return list(visible)
    
    def _can_see_cell(self, x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        err = dx - dy
        cx, cy = x1, y1
        
        # Safety break for loop
        steps = 0
        max_steps = self.grid_size * 3

        while True:
            steps += 1
            if steps > max_steps: return False # Safety break

            if (cx, cy) == (x2, y2): return True
            if (cx, cy) != (x1, y1) and self._blocks_vision((cx, cy)): return False
            
            if cx == x2 and cy == y2: break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
        return True
    
    def _blocks_vision(self, cell):
        if self.room.blocks_vision(cell): return True
        if any(block.position == cell for block in self.blocks): return True
        if self.ramp and self.ramp.position == cell: return True
        return False

    def reset(self):
        self.step_count = 0
        self.seeker_active = False
        self.visibility_count = 0
        
        self._initialize_objects()
        
        self.hider_grabbed = None
        self.seeker_grabbed = None

        # Safe hider spawn
        hx, hy = self._find_valid_spawn_pos()
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
        # FIX: Find a valid position that isn't where the hider is
        hider_pos = (self.hider.x, self.hider.y)
        sx, sy = self._find_valid_spawn_pos(extra_exclusions=[hider_pos])
        
        seeker_dir = np.random.randint(0, 4)
        self.seeker = Seeker(sx, sy, seeker_dir, z=0)

    def is_valid_move(self, x, y, dx, dy):
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size): return False
        new_pos = (new_x, new_y)
        if self.room.is_wall(new_pos): return False
        return True

    def step(self, actions):
        self.step_count += 1
        rewards = {"seeker": 0.0, "hider": 0.0}

        if not self.seeker_active and self.step_count >= self.hiding_phase_steps:
            self.spawn_seeker()
            self.seeker_active = True
            log_info(f"SEEKER SPAWNED at step {self.step_count}!")

        for agent_key in ["seeker", "hider"]:
            action = actions.get(agent_key)
            if action is None: continue
            if agent_key == "seeker" and not self.seeker_active: continue
            
            agent = self.seeker if agent_key == "seeker" else self.hider
            x, y, d, z = agent.get_state()
            
            # Action 9: Lock/Unlock
            if isinstance(action, int) and action == 9 and agent_key == "hider":
                for block in self.blocks:
                    bx, by = block.position
                    if (bx, by) == (x, y) or abs(bx - x) + abs(by - y) == 1:
                        if block.locked: block.unlock()
                        else: block.lock()
                        break
                continue
            
            # Action 8: Climb
            if isinstance(action, int) and action == 8:
                if z == 0:
                    can_climb, ramp_pos = self.can_climb((x, y, z))
                    if can_climb and ramp_pos:
                        agent.update_state(x=ramp_pos[0], y=ramp_pos[1], z=1)
                continue
            
            # Actions 0-3: Move
            if isinstance(action, int) and action in self.moves:
                dx, dy = self.moves[action]
                new_x, new_y = x + dx, y + dy
                
                if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size): continue
                
                new_pos = (new_x, new_y)
                target_height = self.get_height_at(new_x, new_y)
                
                if z == 0:
                    if self.room.is_wall(new_pos): continue
                    if any(block.position == new_pos for block in self.blocks): continue
                    if self.ramp and self.ramp.position == new_pos: continue
                    
                    grabbed = self.hider_grabbed if agent_key == "hider" else self.seeker_grabbed
                    if grabbed is not None:
                        obj = self.get_grabbed_object(agent_key)
                        if obj: obj.release()
                        if agent_key == "hider": self.hider_grabbed = None
                        else: self.seeker_grabbed = None
                    
                    agent.update_state(x=new_x, y=new_y, direction=action, z=0)
                
                elif z == 1:
                    if target_height == 1:
                        agent.update_state(x=new_x, y=new_y, direction=action, z=1)
                    else:
                        agent.update_state(x=new_x, y=new_y, direction=action, z=0)
            
            # Actions 4-7: Grab
            elif isinstance(action, int) and 4 <= action <= 7:
                if z != 0: continue
                
                grabbed = self.hider_grabbed if agent_key == "hider" else self.seeker_grabbed
                
                if grabbed is None:
                    success, obj_type, obj_idx = self.try_grab_object(agent_key, (x, y, z))
                    if success:
                        if obj_type == 'block':
                            self.blocks[obj_idx].grab(agent_key)
                            if agent_key == "hider": self.hider_grabbed = ('block', obj_idx)
                            else: self.seeker_grabbed = ('block', obj_idx)
                        elif obj_type == 'ramp':
                            self.ramp.grab(agent_key)
                            if agent_key == "hider": self.hider_grabbed = ('ramp',)
                            else: self.seeker_grabbed = ('ramp',)
                else:
                    obj = self.get_grabbed_object(agent_key)
                    if obj is None: continue
                    
                    move_action = action - 4
                    dx, dy = self.moves[move_action]
                    new_x, new_y = x + dx, y + dy
                    
                    if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size): continue
                    
                    new_agent_pos = (new_x, new_y)
                    ox, oy = obj.position
                    new_obj_x, new_obj_y = ox + dx, oy + dy
                    
                    if not (0 <= new_obj_x < self.grid_size and 0 <= new_obj_y < self.grid_size):
                        obj.release()
                        if agent_key == "hider": self.hider_grabbed = None
                        else: self.seeker_grabbed = None
                        continue
                    
                    new_obj_pos = (new_obj_x, new_obj_y)
                    
                    if self.room.is_wall(new_obj_pos):
                        obj.release()
                        if agent_key == "hider": self.hider_grabbed = None
                        else: self.seeker_grabbed = None
                        continue
                    
                    collision = False
                    for block in self.blocks:
                        if block != obj and block.position == new_obj_pos:
                            collision = True; break
                    if not collision and self.ramp and self.ramp != obj and self.ramp.position == new_obj_pos:
                        collision = True
                    
                    if collision:
                        obj.release()
                        if agent_key == "hider": self.hider_grabbed = None
                        else: self.seeker_grabbed = None
                        continue
                    
                    if self.room.is_wall(new_agent_pos):
                        obj.release()
                        if agent_key == "hider": self.hider_grabbed = None
                        else: self.seeker_grabbed = None
                        continue
                    
                    agent_collision = False
                    for block in self.blocks:
                        if block != obj and block.position == new_agent_pos:
                            agent_collision = True; break
                    if not agent_collision and self.ramp and self.ramp != obj and self.ramp.position == new_agent_pos:
                        agent_collision = True
                    
                    if agent_collision:
                        obj.release()
                        if agent_key == "hider": self.hider_grabbed = None
                        else: self.seeker_grabbed = None
                        continue
                    
                    obj.move(new_obj_pos)
                    agent.update_state(x=new_x, y=new_y, direction=move_action, z=0)
        
        for agent_key in ["seeker", "hider"]:
            if agent_key == "seeker" and not self.seeker_active: continue
            agent = self.seeker if agent_key == "seeker" else self.hider
            x, y, d, z = agent.get_state()
            new_z = self.check_fall(x, y, z)
            if new_z != z: agent.update_state(z=new_z)
        
        done = False
        if self.seeker_active:
            visible_seeker = set(self.compute_visible_cells(self.seeker.get_state()))
            hider_pos = self.hider.get_state()[:2]
            seeker_pos = self.seeker.get_state()[:2]
            
            if hider_pos in visible_seeker:
                rewards["seeker"] += 2.0
                rewards["hider"] -= 2.0
                self.visibility_count += 1
            
            distance = abs(hider_pos[0] - seeker_pos[0]) + abs(hider_pos[1] - seeker_pos[1])
            if distance <= self.capture_distance:
                done = True
                rewards["seeker"] += 100.0
                rewards["hider"] -= 100.0
        
        if not self.seeker_active:
            hider_pos = self.hider.get_state()[:2]
            if self.room.is_inside(hider_pos): rewards["hider"] += 0.5
            _, _, _, z = self.hider.get_state()
            if z == 1: rewards["hider"] += 0.1
            locked_blocks = sum(1 for b in self.blocks if b.locked)
            rewards["hider"] += locked_blocks * 0.2
        else:
            hider_pos = self.hider.get_state()[:2]
            if self.room.is_inside(hider_pos): rewards["hider"] += 0.3
            rewards["seeker"] += 0.01
        
        if not done and self.step_count >= self.max_steps:
            done = True
            seeking_steps = self.step_count - self.hiding_phase_steps
            visibility_ratio = self.visibility_count / max(seeking_steps, 1)
            
            if visibility_ratio > 0.3:
                rewards["seeker"] += 50.0
                rewards["hider"] -= 50.0
            else:
                rewards["hider"] += 50.0
                rewards["seeker"] -= 50.0
        
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