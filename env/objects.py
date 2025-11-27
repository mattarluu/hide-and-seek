"""
env/objects.py

This module defines movable objects in the environment:
- Block: A 1x1 object that can be grabbed by agents
- Ramp: A 1x1 object that allows agents to climb to height z=1

Height system (2 levels):
- z=0: Ground
- z=1: On ramp, block, or wall (only ramp is climbable)

Objects can be grabbed and moved. Blocks can be locked by hider.
"""

from utils.logger import log_debug

class Block:
    """
    Represents a movable block that agents can grab and move.
    Blocks are 1x1 and have height z=1.
    Can be locked by the hider to prevent movement.
    Blocks cannot be climbed directly - only ramps can be climbed.
    """
    def __init__(self, position):
        """
        Initialize a block.
        
        Args:
            position (tuple): (x, y) coordinate of the block.
        """
        self.position = position
        self.height = 1  # Blocks are at height 1 (same as walls and ramp)
        self.locked = False  # Can be locked by hider
        self.grabbed_by = None  # Which agent is currently grabbing this block
        
    def can_move_to(self, new_position, grid_size, room, other_objects):
        """
        Check if the block can be moved to a new position.
        
        Args:
            new_position (tuple): Target (x, y) position.
            grid_size (int): Size of the grid.
            room: Room object to check collisions.
            other_objects (list): List of other objects to check collisions.
            
        Returns:
            bool: True if the block can move there.
        """
        x, y = new_position
        
        # Check grid bounds
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False
            
        # Check if position is occupied by room wall
        if room.is_wall(new_position):
            return False
            
        # Check if position is occupied by another object
        for obj in other_objects:
            if obj != self and obj.position == new_position:
                return False
                
        return True
    
    def move(self, new_position):
        """
        Move the block to a new position.
        
        Args:
            new_position (tuple): New (x, y) position.
        """
        self.position = new_position
        # log_debug(f"Block moved to {self.position}")
    
    def lock(self):
        """Lock the block (only hider can do this)."""
        self.locked = True
    
    def unlock(self):
        """Unlock the block."""
        self.locked = False
    
    def grab(self, agent_name):
        """Mark block as grabbed by an agent."""
        self.grabbed_by = agent_name
    
    def release(self):
        """Release the block."""
        self.grabbed_by = None


class Ramp:
    """
    Represents a movable ramp that allows climbing to z=1.
    Ramps are 1x1 and are the ONLY climbable object.
    Once on ramp (z=1), agents can walk onto blocks or walls at z=1.
    Ramps CANNOT be locked.
    """
    def __init__(self, position):
        """
        Initialize a ramp.
        
        Args:
            position (tuple): (x, y) coordinate of the ramp.
        """
        self.position = position
        self.height = 1  # Ramps are at height 1 (intermediate level)
        self.grabbed_by = None  # Which agent is currently grabbing this ramp
        
    def can_move_to(self, new_position, grid_size, room, other_objects):
        """
        Check if the ramp can be moved to a new position.
        
        Args:
            new_position (tuple): Target (x, y) position.
            grid_size (int): Size of the grid.
            room: Room object to check collisions.
            other_objects (list): List of other objects to check collisions.
            
        Returns:
            bool: True if the ramp can move there.
        """
        x, y = new_position
        
        # Check grid bounds
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False
            
        # Check if position is occupied by room wall
        if room.is_wall(new_position):
            return False
            
        # Check if position is occupied by another object
        for obj in other_objects:
            if obj != self and obj.position == new_position:
                return False
                
        return True
    
    def move(self, new_position):
        """
        Move the ramp to a new position.
        
        Args:
            new_position (tuple): New (x, y) position.
        """
        self.position = new_position
        # log_debug(f"Ramp moved to {self.position}")
    
    def grab(self, agent_name):
        """Mark ramp as grabbed by an agent."""
        self.grabbed_by = agent_name
    
    def release(self):
        """Release the ramp."""
        self.grabbed_by = None
    
    def allows_climbing_to(self, agent_pos, target_pos, blocks, room):
        """
        Check if this ramp allows an agent to climb from agent_pos to target_pos.
        
        Args:
            agent_pos (tuple): Current agent position.
            target_pos (tuple): Target position (block or wall).
            blocks (list): List of Block objects.
            room: Room object.
            
        Returns:
            bool: True if climbing is allowed.
        """
        # Ramp must be adjacent to agent
        ax, ay = agent_pos
        rx, ry = self.position
        if abs(ax - rx) + abs(ay - ry) != 1:
            return False
            
        # Target must be adjacent to ramp
        tx, ty = target_pos
        if abs(tx - rx) + abs(ty - ry) != 1:
            return False
            
        # Target must be a block or room wall
        is_block = any(block.position == target_pos for block in blocks)
        is_wall = room.is_wall(target_pos)
        
        return is_block or is_wall