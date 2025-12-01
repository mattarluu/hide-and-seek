"""
tests/pygame_interactive_test_v2.py

Interactive test with ALL new mechanics:
- Height system (z=0: ground, z=1: ramp/blocks/walls)
- Grab mechanics (unified for all objects)
- Lock mechanics (hider only)
- Visual indicators for height and locked blocks
- Climbing: Only via ramp, then can walk on all z=1 surfaces
- FIELD OF VIEW: Blue shade for hider, Red shade for seeker

INSTALLATION:
pip install pygame

CONTROLS:
=========
HIDER (Blue):
  W/A/S/D     - Move
  T/F/G/H     - Grab/Move with object
  Y           - Climb (only works on ramp)
  B           - Lock/Unlock block

SEEKER (Red):
  Arrows      - Move
  I/J/K/L     - Grab/Move with object
  U           - Climb (only works on ramp)

GENERAL:
  R           - Reset
  V           - Toggle FOV (Field of View)
  ESC         - Quit
  SPACE       - Pause

VISUAL:
  Blue transparent area - Hider's field of view
  Red transparent area  - Seeker's field of view
  Purple area          - Overlapping vision
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pygame
except ImportError:
    print("ERROR: pygame not installed")
    print("Install with: pip install pygame")
    sys.exit(1)

from env.hide_and_seek_env import HideAndSeekEnv
from utils.logger import log_info

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
BLUE = (0, 100, 255)
LIGHT_BLUE = (100, 150, 255)
RED = (255, 50, 50)
LIGHT_RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 150)
BROWN = (139, 69, 19)
DARK_BROWN = (100, 50, 10)
ORANGE = (255, 165, 0)
GOLD = (255, 215, 0)

class PyGameInteractiveTestV2:
    def __init__(self):
        pygame.init()
        
        # Window setup
        self.cell_size = 35
        self.info_width = 350
        self.grid_width = 20 * self.cell_size
        self.window_width = self.grid_width + self.info_width
        self.window_height = 20 * self.cell_size + 50
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Hide and Seek V2 - Interactive Test")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)
        
        # Environment
        self.env = HideAndSeekEnv()
        self.env.reset()
        self.env.spawn_seeker()
        self.env.seeker_active = True
        
        self.paused = False
        self.last_action_hider = "None"
        self.last_action_seeker = "None"
        self.show_fov = True  # Toggle for field of view visibility
        
        log_info("PyGame Interactive Test V2 Started!")
        
    def handle_input(self):
        """Handle keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                actions = {"seeker": None, "hider": None}  # Use None instead of 0
                execute = False
                
                # ESC to quit
                if event.key == pygame.K_ESCAPE:
                    return False
                
                # R to reset
                if event.key == pygame.K_r:
                    self.env.reset()
                    self.env.spawn_seeker()
                    self.env.seeker_active = True
                    self.last_action_hider = "RESET"
                    self.last_action_seeker = "RESET"
                    log_info("Environment reset!")
                    continue
                
                # SPACE to pause
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    log_info(f"{'PAUSED' if self.paused else 'RESUMED'}")
                    continue
                
                # V to toggle field of view
                if event.key == pygame.K_v:
                    self.show_fov = not self.show_fov
                    log_info(f"Field of View: {'ON' if self.show_fov else 'OFF'}")
                    continue
                
                if self.paused:
                    continue
                
                # HIDER controls (W/A/S/D) - Movement
                if event.key == pygame.K_w:
                    actions["hider"] = 0
                    self.last_action_hider = "Move Up"
                    execute = True
                elif event.key == pygame.K_a:
                    actions["hider"] = 3
                    self.last_action_hider = "Move Left"
                    execute = True
                elif event.key == pygame.K_s:
                    actions["hider"] = 2
                    self.last_action_hider = "Move Down"
                    execute = True
                elif event.key == pygame.K_d:
                    actions["hider"] = 1
                    self.last_action_hider = "Move Right"
                    execute = True
                
                # HIDER grab+move (T/F/G/H)
                elif event.key == pygame.K_t:
                    actions["hider"] = 4
                    self.last_action_hider = "Grab+Move Up"
                    execute = True
                elif event.key == pygame.K_f:
                    actions["hider"] = 7
                    self.last_action_hider = "Grab+Move Left"
                    execute = True
                elif event.key == pygame.K_g:
                    actions["hider"] = 6
                    self.last_action_hider = "Grab+Move Down"
                    execute = True
                elif event.key == pygame.K_h:
                    actions["hider"] = 5
                    self.last_action_hider = "Grab+Move Right"
                    execute = True
                
                # HIDER climb (Y)
                elif event.key == pygame.K_y:
                    actions["hider"] = 8
                    self.last_action_hider = "Climb"
                    execute = True
                
                # HIDER lock/unlock (B)
                elif event.key == pygame.K_b:
                    actions["hider"] = 9
                    self.last_action_hider = "Lock/Unlock"
                    execute = True
                
                # SEEKER controls (Arrow keys) - Movement
                elif event.key == pygame.K_UP:
                    actions["seeker"] = 0
                    self.last_action_seeker = "Move Up"
                    execute = True
                elif event.key == pygame.K_LEFT:
                    actions["seeker"] = 3
                    self.last_action_seeker = "Move Left"
                    execute = True
                elif event.key == pygame.K_DOWN:
                    actions["seeker"] = 2
                    self.last_action_seeker = "Move Down"
                    execute = True
                elif event.key == pygame.K_RIGHT:
                    actions["seeker"] = 1
                    self.last_action_seeker = "Move Right"
                    execute = True
                
                # SEEKER grab+move (I/J/K/L)
                elif event.key == pygame.K_i:
                    actions["seeker"] = 4
                    self.last_action_seeker = "Grab+Move Up"
                    execute = True
                elif event.key == pygame.K_j:
                    actions["seeker"] = 7
                    self.last_action_seeker = "Grab+Move Left"
                    execute = True
                elif event.key == pygame.K_k:
                    actions["seeker"] = 6
                    self.last_action_seeker = "Grab+Move Down"
                    execute = True
                elif event.key == pygame.K_l:
                    actions["seeker"] = 5
                    self.last_action_seeker = "Grab+Move Right"
                    execute = True
                
                # SEEKER climb (U)
                elif event.key == pygame.K_u:
                    actions["seeker"] = 8
                    self.last_action_seeker = "Climb"
                    execute = True
                
                # Execute action
                if execute:
                    obs, done, rewards = self.env.step(actions)
                    
                    if rewards['seeker'] != 0 or rewards['hider'] != 0:
                        log_info(f"Rewards: Seeker={rewards['seeker']:.1f}, Hider={rewards['hider']:.1f}")
                    
                    if done:
                        log_info("Episode finished! Press R to reset.")
        
        return True
    
    def draw_grid(self):
        """Draw the grid lines."""
        for i in range(21):
            pygame.draw.line(self.screen, LIGHT_GRAY, 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, 20 * self.cell_size), 1)
            pygame.draw.line(self.screen, LIGHT_GRAY,
                           (0, i * self.cell_size),
                           (20 * self.cell_size, i * self.cell_size), 1)
    
    def draw_room(self):
        """Draw the room."""
        # Room interior
        room_rect = pygame.Rect(
            self.env.room.top_left[0] * self.cell_size,
            self.env.room.top_left[1] * self.cell_size,
            self.env.room.width * self.cell_size,
            self.env.room.height * self.cell_size
        )
        pygame.draw.rect(self.screen, YELLOW, room_rect)
        
        # Walls
        for wx, wy in self.env.room.wall_cells:
            wall_rect = pygame.Rect(
                wx * self.cell_size,
                wy * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, GRAY, wall_rect)
            pygame.draw.rect(self.screen, BLACK, wall_rect, 2)
        
        # Opening
        for ox, oy in self.env.room.opening_cells:
            open_rect = pygame.Rect(
                ox * self.cell_size,
                oy * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, GREEN, open_rect)
            pygame.draw.rect(self.screen, DARK_GRAY, open_rect, 2)
            
            text = self.font.render("O", True, DARK_GRAY)
            text_rect = text.get_rect(center=open_rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_objects(self):
        """Draw blocks and ramp with lock indicators."""
        # Blocks
        for block in self.env.blocks:
            bx, by = block.position
            block_rect = pygame.Rect(
                bx * self.cell_size,
                by * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            
            # Color based on lock status
            if block.locked:
                pygame.draw.rect(self.screen, DARK_BROWN, block_rect)
            else:
                pygame.draw.rect(self.screen, BROWN, block_rect)
            
            pygame.draw.rect(self.screen, BLACK, block_rect, 3)
            
            # Draw "B" or lock icon
            if block.locked:
                text = self.font.render("🔒", True, GOLD)
            else:
                text = self.font.render("B", True, WHITE)
            text_rect = text.get_rect(center=block_rect.center)
            self.screen.blit(text, text_rect)
            
            # Show if grabbed
            if block.grabbed_by:
                grab_text = self.tiny_font.render(f"[{block.grabbed_by[0].upper()}]", True, WHITE)
                self.screen.blit(grab_text, (bx * self.cell_size + 2, by * self.cell_size + 2))
        
        # Ramp
        if self.env.ramp:
            rx, ry = self.env.ramp.position
            points = [
                (rx * self.cell_size, (ry + 1) * self.cell_size),
                ((rx + 1) * self.cell_size, (ry + 1) * self.cell_size),
                ((rx + 1) * self.cell_size, ry * self.cell_size)
            ]
            pygame.draw.polygon(self.screen, ORANGE, points)
            pygame.draw.polygon(self.screen, BLACK, points, 3)
            
            ramp_rect = pygame.Rect(
                rx * self.cell_size,
                ry * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            text = self.small_font.render("R", True, WHITE)
            text_rect = text.get_rect(center=ramp_rect.center)
            self.screen.blit(text, text_rect)
            
            # Show if grabbed
            if self.env.ramp.grabbed_by:
                grab_text = self.tiny_font.render(f"[{self.env.ramp.grabbed_by[0].upper()}]", True, WHITE)
                self.screen.blit(grab_text, (rx * self.cell_size + 2, ry * self.cell_size + 2))
    
    def draw_field_of_view(self):
        """Draw the field of view for both agents with transparent overlays."""
        # Create transparent surfaces
        fov_surface = pygame.Surface((self.grid_width, 20 * self.cell_size))
        fov_surface.set_alpha(40)  # Transparency level (0-255, lower = more transparent)
        
        # Draw hider's field of view in blue
        if self.env.hider:
            visible_cells = self.env.compute_visible_cells(self.env.hider.get_state())
            for cell_x, cell_y in visible_cells:
                rect = pygame.Rect(
                    cell_x * self.cell_size,
                    cell_y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(fov_surface, BLUE, rect)
        
        # Blit hider's FOV
        self.screen.blit(fov_surface, (0, 0))
        
        # Draw seeker's field of view in red
        if self.env.seeker_active and self.env.seeker:
            fov_surface_seeker = pygame.Surface((self.grid_width, 20 * self.cell_size))
            fov_surface_seeker.set_alpha(40)
            
            visible_cells = self.env.compute_visible_cells(self.env.seeker.get_state())
            for cell_x, cell_y in visible_cells:
                rect = pygame.Rect(
                    cell_x * self.cell_size,
                    cell_y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(fov_surface_seeker, RED, rect)
            
            # Blit seeker's FOV
            self.screen.blit(fov_surface_seeker, (0, 0))
    
    def draw_agent(self, agent, color, light_color, label):
        """Draw an agent with height indicator."""
        x, y, d, z = agent.get_state()
        
        # Adjust color based on height (z=0 or z=1)
        if z == 0:
            agent_color = color
        else:  # z == 1
            agent_color = light_color
        
        # Draw circle
        center = (int((x + 0.5) * self.cell_size), int((y + 0.5) * self.cell_size))
        radius = self.cell_size // 3
        
        # Draw shadow if elevated
        if z > 0:
            shadow_offset = z * 4
            pygame.draw.circle(self.screen, DARK_GRAY, 
                             (center[0] + shadow_offset, center[1] + shadow_offset), 
                             radius, 0)
        
        pygame.draw.circle(self.screen, agent_color, center, radius)
        pygame.draw.circle(self.screen, BLACK, center, radius, 2)
        
        # Draw direction arrow
        dx, dy = self.env.moves[d]
        arrow_end = (
            center[0] + dx * self.cell_size // 3,
            center[1] + dy * self.cell_size // 3
        )
        pygame.draw.line(self.screen, BLACK, center, arrow_end, 3)
        
        # Draw label with height
        label_text = f"{label} z={z}"
        text = self.small_font.render(label_text, True, BLACK)
        text_rect = text.get_rect(center=(center[0], center[1] - self.cell_size // 2 - 8))
        self.screen.blit(text, text_rect)
    
    def draw_info_panel(self):
        """Draw information panel."""
        x_offset = self.grid_width + 10
        y = 10
        
        # Title
        if self.paused:
            title = self.font.render("PAUSED", True, RED)
        else:
            title = self.font.render("CONTROLS V2", True, BLACK)
        self.screen.blit(title, (x_offset, y))
        y += 35
        
        # Controls info
        info_lines = [
            ("HIDER (Blue)", BLUE),
            ("W/A/S/D - Move", BLACK),
            ("T/F/G/H - Grab+Move", BLACK),
            ("Y - Climb", BLACK),
            ("B - Lock/Unlock", BLACK),
            (f"Last: {self.last_action_hider}", DARK_GRAY),
            ("", BLACK),
            ("SEEKER (Red)", RED),
            ("Arrows - Move", BLACK),
            ("I/J/K/L - Grab+Move", BLACK),
            ("U - Climb", BLACK),
            (f"Last: {self.last_action_seeker}", DARK_GRAY),
            ("", BLACK),
            ("GENERAL", BLACK),
            ("R - Reset", BLACK),
            ("V - Toggle FOV", BLACK),
            ("SPACE - Pause", BLACK),
            ("ESC - Quit", BLACK),
            ("", BLACK),
            (f"FOV: {'ON' if self.show_fov else 'OFF'}", GREEN if self.show_fov else GRAY),
            ("Blue - Hider FOV", BLUE if self.show_fov else GRAY),
            ("Red - Seeker FOV", RED if self.show_fov else GRAY),
        ]
        
        for text, color in info_lines:
            if text:
                rendered = self.small_font.render(text, True, color)
                self.screen.blit(rendered, (x_offset, y))
            y += 20
        
        # Status
        y += 10
        pygame.draw.line(self.screen, BLACK, (x_offset, y), (x_offset + 300, y), 2)
        y += 15
        
        hx, hy, hd, hz = self.env.hider.get_state()
        sx, sy, sd, sz = self.env.seeker.get_state() if self.env.seeker_active else (-1, -1, -1, -1)
        
        status_lines = [
            f"Step: {self.env.step_count}/{self.env.max_steps}",
            "",
        ]
        
        # Phase indicator
        if not self.env.seeker_active:
            remaining_hide = self.env.hiding_phase_steps - self.env.step_count
            status_lines.append(f"🔵 HIDING PHASE")
            status_lines.append(f"Seeker in: {remaining_hide} steps")
        else:
            status_lines.append(f"🔴 SEEKING PHASE")
        
        status_lines.extend([
            "",
            f"Hider: ({hx},{hy}) z={hz}",
            f"Seeker: ({sx},{sy}) z={sz}",
            "",
            "Heights:",
            "  z=0: Ground",
            "  z=1: Elevated",
            "  (ramp/blocks/walls)",
            "",
            "Climb: Only ramp!",
            "Walk: All z=1",
            "",
        ])
        
        # Grabbed status
        if self.env.hider_grabbed:
            obj_type = self.env.hider_grabbed[0]
            status_lines.append(f"Hider holding: {obj_type}")
        
        if self.env.seeker_grabbed:
            obj_type = self.env.seeker_grabbed[0]
            status_lines.append(f"Seeker holding: {obj_type}")
        
        status_lines.append("")
        
        # Locked blocks
        locked_count = sum(1 for b in self.env.blocks if b.locked)
        if locked_count > 0:
            status_lines.append(f"Locked blocks: {locked_count}")
        
        status_lines.append("")
        
        # Victory conditions
        if self.env.seeker_active:
            seeking_steps = self.env.step_count - self.env.hiding_phase_steps
            visibility_ratio = self.env.visibility_count / max(seeking_steps, 1) * 100
            status_lines.append(f"Visible: {self.env.visibility_count}/{seeking_steps} ({visibility_ratio:.0f}%)")
            
            # Distance for capture
            distance = abs(hx - sx) + abs(hy - sy)
            status_lines.append(f"Distance: {distance}")
            if distance <= self.env.capture_distance:
                status_lines.append("⚠️ CAPTURE RANGE!")
        
        status_lines.append("")
        
        # Check status
        in_room = self.env.room.is_inside((hx, hy))
        if in_room:
            status_lines.append("Hider IN ROOM ✓")
        
        if self.env.seeker_active:
            visible = set(self.env.compute_visible_cells(self.env.seeker.get_state()))
            if (hx, hy) in visible:
                status_lines.append("Seeker SEES Hider!")
        
        for line in status_lines:
            rendered = self.tiny_font.render(line, True, BLACK)
            self.screen.blit(rendered, (x_offset, y))
            y += 18
    
    def render(self):
        """Render everything."""
        self.screen.fill(WHITE)
        
        # Draw environment
        self.draw_grid()
        self.draw_room()
        
        # Draw field of view BEFORE objects and agents (as background overlay)
        if self.show_fov:
            self.draw_field_of_view()
        
        self.draw_objects()
        
        # Draw agents
        self.draw_agent(self.env.hider, BLUE, LIGHT_BLUE, "H")
        if self.env.seeker_active:
            self.draw_agent(self.env.seeker, RED, LIGHT_RED, "S")
        
        # Draw info panel
        self.draw_info_panel()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_input()
            self.render()
            self.clock.tick(50)  # 50 FPS
        
        pygame.quit()
        log_info("PyGame test V2 closed.")


if __name__ == "__main__":
    test = PyGameInteractiveTestV2()
    test.run()