"""
agents/hider.py

This module defines the Hider agent for the multi-agent hide and seek project.
The hider's role is to hide from the seeker.
It stores the agent's state (position, direction, and HEIGHT) and provides methods to update the state
and process actions. Debug output is standardized using the custom logger.

Height system:
- z=0: Ground level
- z=1: On ramp, block, or wall (agents can only climb via ramp, but can walk on all z=1 surfaces)
"""

from utils.logger import log_debug, log_info

class Hider:
    def __init__(self, x, y, direction, z=0):
        """
        Initialize the hider with a starting position, direction, and height.
        
        Args:
            x: X position
            y: Y position
            direction: Facing direction (0-3)
            z: Height level (0=ground, 1=on ramp/block/wall)
        """
        self.x = x
        self.y = y
        self.direction = direction
        self.z = z  # Height: 0=ground, 1=elevated (ramp/block/wall)
        # log_debug(f"Hider created at ({self.x}, {self.y}, z={self.z}) facing direction {self.direction}")

    def update_state(self, x=None, y=None, direction=None, z=None):
        """
        Update the hider's state including height.
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if direction is not None:
            self.direction = direction
        if z is not None:
            self.z = z
        # log_debug(f"Hider updated state: ({self.x}, {self.y}, z={self.z}), direction: {self.direction}")

    def process_action(self, action, moves):
        """
        Process a movement action.
        For movement actions (0-3), update the position based on the provided movement mapping.
        """
        if isinstance(action, int) and action in moves:
            dx, dy = moves[action]
            new_x = self.x + dx
            new_y = self.y + dy
            self.update_state(x=new_x, y=new_y, direction=action)
            # log_debug(f"Hider processed movement action {action}: moved to ({new_x}, {new_y})")
            return (new_x, new_y, action)
        else:
            # log_debug(f"Hider received non-movement action: {action}")
            return (self.x, self.y, self.direction)

    def get_state(self):
        """
        Return the current state as a tuple (x, y, direction, z).
        """
        return (self.x, self.y, self.direction, self.z)
    
    def get_position_3d(self):
        """
        Return 3D position (x, y, z).
        """
        return (self.x, self.y, self.z)


if __name__ == "__main__":
    hider = Hider(5, 5, 0, z=0)
    log_info(f"Initial Hider state: {hider.get_state()}")
    hider.process_action(1, {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)})
    log_info(f"Updated Hider state: {hider.get_state()}")