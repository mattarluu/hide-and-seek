"""
env/room.py

This module encapsulates the room layout for the hide and seek environment.
The room has an opening in the top-left corner (3 cells removed) instead of a door.

The room is defined as an 8x8 area located in the bottom right corner of a 20x20 grid.
The opening is in the top-left corner (3 wall cells removed).
"""

from utils.logger import log_debug

class Room:
    """
    Represents a room within the grid with an opening in the top-left corner.
    """
    def __init__(self, top_left, width, height):
        """
        Initialize the room with an opening in the top-left corner.

        Args:
            top_left (tuple): (x, y) coordinate for the top-left corner.
            width (int): Width of the room.
            height (int): Height of the room.
        """
        self.top_left = top_left
        self.width = width
        self.height = height

        # Compute wall cells: all cells on the perimeter
        self.wall_cells = set()
        for x in range(top_left[0], top_left[0] + width):
            self.wall_cells.add((x, top_left[1]))                      # Top wall
            self.wall_cells.add((x, top_left[1] + height - 1))         # Bottom wall
        for y in range(top_left[1], top_left[1] + height):
            self.wall_cells.add((top_left[0], y))                      # Left wall
            self.wall_cells.add((top_left[0] + width - 1, y))          # Right wall

        # Remove the 3 cells in the top-left corner to create the opening
        # Top-left corner cells: (top_left[0], top_left[1]), (top_left[0]+1, top_left[1]), (top_left[0], top_left[1]+1)
        opening_cells = [
            (top_left[0], top_left[1]),
            (top_left[0] + 1, top_left[1]),
            (top_left[0], top_left[1] + 1)
        ]
        for cell in opening_cells:
            if cell in self.wall_cells:
                self.wall_cells.remove(cell)
        
        self.opening_cells = opening_cells

        # log_debug(f"Room created at {self.top_left} with size {width}x{height}.")
        # log_debug(f"Room wall cells: {self.wall_cells}")
        # log_debug(f"Opening cells: {self.opening_cells}")

    def is_inside(self, cell):
        """
        Check if a cell is inside the room boundaries.
        """
        x, y = cell
        return (self.top_left[0] <= x < self.top_left[0] + self.width and
                self.top_left[1] <= y < self.top_left[1] + self.height)

    def is_wall(self, cell):
        """
        Check if a cell is a wall cell of the room.
        """
        return cell in self.wall_cells

    def is_opening(self, cell):
        """
        Check if a cell is part of the opening.
        """
        return cell in self.opening_cells

    def blocks_vision(self, cell):
        """
        Determine if a cell blocks vision.
        A cell blocks vision if it is a wall cell.
        """
        return cell in self.wall_cells

    def get_all_cells(self):
        """
        Get all cells that belong to the room (both interior and walls).
        """
        cells = set()
        for x in range(self.top_left[0], self.top_left[0] + self.width):
            for y in range(self.top_left[1], self.top_left[1] + self.height):
                cells.add((x, y))
        return cells


if __name__ == "__main__":
    # Visualization for the Room module
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    grid_size = 10
    # Create a 4x4 room in the bottom right corner (top_left=(6,6) because 10-4=6)
    room = Room(top_left=(6, 6), width=4, height=4, door_side="left")

    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title("Room Visualization")

    # Draw grid lines for clarity
    for x in range(grid_size + 1):
        ax.plot([x, x], [0, grid_size], color='lightgray', lw=0.5)
    for y in range(grid_size + 1):
        ax.plot([0, grid_size], [y, y], color='lightgray', lw=0.5)

    # Draw room cells
    room_cells = room.get_all_cells()
    for cell in room_cells:
        rect = patches.Rectangle(cell, 1, 1, facecolor='lightyellow', edgecolor='black', lw=0.5)
        ax.add_patch(rect)

    # Highlight wall cells in gray
    for wall in room.wall_cells:
        rect = patches.Rectangle(wall, 1, 1, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)

    # Highlight the door cell with color based on its state
    door_color = 'green' if room.door.is_open and not room.door.is_locked else 'red'
    door_rect = patches.Rectangle(room.door.position, 1, 1, facecolor=door_color, edgecolor='black')
    ax.add_patch(door_rect)
    ax.text(room.door.position[0] + 0.2, room.door.position[1] + 0.5, "Door", color='white', fontsize=8)

    plt.show()