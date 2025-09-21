import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
from collections import deque
import colorsys
import heapq
import time
import signal
import sys

np.random.seed(42)

# Configuration constants - adjusted for better visibility
FIG_WIDTH = 12  # Increased for better maze visibility
FIG_HEIGHT = 12  # Square aspect ratio
DPI = 80  # Slightly reduced for performance but maintaining quality

MAZE_WIDTH = 25
MAZE_HEIGHT = 25

# Color definitions
BG_COLOR = '#0A0A15'
WALL_COLOR = '#FFFFFF'
CURRENT_COLOR = '#FF1493'
GENERATION_PARTICLE = '#00FF7F'
SOLVING_PARTICLE = '#1E90FF'
VISITED_COLOR = '#3A1C71'
PATH_COLOR = '#FF3333'
START_COLOR = '#00FF7F'
END_COLOR = '#FF4500'

PRIM_GENERATION_PARTICLE = '#FF9500'
DIJKSTRA_SOLVING_PARTICLE = '#00EEFF'

# Reduced particle effects for better performance
PARTICLE_LIFETIME = 3  # Reduced for performance
PARTICLE_SIZE = 0.8    # Reduced for performance

GEN_DURATION = 3       # Reduced for faster generation
SOLVE_DURATION = 4     # Reduced for faster solving
TOTAL_DURATION = GEN_DURATION + SOLVE_DURATION
TARGET_FPS = 30        # Reduced from 60 to 30 for better performance
GEN_FRAMES = GEN_DURATION * TARGET_FPS
SOLVE_FRAMES = SOLVE_DURATION * TARGET_FPS
TOTAL_FRAMES = TOTAL_DURATION * TARGET_FPS

GLOW_INTENSITY = 1.2
PULSE_SPEED = 0.15


class Particle:
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME, velocity=(0, 0)):
        self.x = x
        self.y = y
        self.original_color = color
        self.color = color
        self.size = size * random.uniform(0.8, 1.2)
        self.max_lifetime = lifetime
        self.lifetime = lifetime
        self.decay = 0.85
        self.vx = velocity[0] * random.uniform(0.5, 1.5)
        self.vy = velocity[1] * random.uniform(0.5, 1.5)

    def update(self):
        self.lifetime -= 1
        self.size *= self.decay
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.9
        self.vy *= 0.9

        alpha = self.lifetime / self.max_lifetime
        r, g, b = self.original_color
        self.color = (r, g, b, alpha)
        return self.lifetime > 0 and self.size > 0.1


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'N': True, 'E': True, 'S': True, 'W': True}
        self.visited = False
        self.in_path = False
        self.is_start = False
        self.is_end = False
        self.is_frontier = False
        self.generation_order = -1
        self.distance = float('inf')
        self.in_current_path = False


def create_maze_prim(width, height):
    grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
    generation_states = []
    cells_added = 0

    # Start with a random cell
    start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
    grid[start_x][start_y].visited = True
    grid[start_x][start_y].generation_order = cells_added
    cells_added += 1

    # Add the initial cell to the maze
    generation_states.append(capture_generation_state(grid, grid[start_x][start_y], [], [], [(start_x, start_y)]))

    # Create a list of frontier cells (neighbors of visited cells)
    frontier = []
    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]
    opposite = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

    # Add the initial frontier cells
    for direction, dx, dy in directions:
        nx, ny = start_x + dx, start_y + dy
        if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].visited:
            frontier.append((nx, ny, start_x, start_y, direction))
            grid[nx][ny].is_frontier = True

    # Capture state with frontier
    generation_states.append(capture_generation_state(grid, grid[start_x][start_y], [], [], frontier_cells=[(nx, ny) for nx, ny, _, _, _ in frontier]))

    while frontier:
        # Randomly select a frontier cell
        idx = random.randint(0, len(frontier) - 1)
        fx, fy, px, py, dir_from_parent = frontier.pop(idx)
        
        # Mark the frontier cell as visited and remove it from frontier
        grid[fx][fy].visited = True
        grid[fx][fy].generation_order = cells_added
        cells_added += 1
        grid[fx][fy].is_frontier = False
        
        # Remove the wall between the frontier cell and its parent
        grid[px][py].walls[dir_from_parent] = False
        grid[fx][fy].walls[opposite[dir_from_parent]] = False
        
        # Capture state after connecting the cell
        generation_states.append(capture_generation_state(grid, grid[fx][fy], [], [], [(fx, fy)], frontier_cells=[(nx, ny) for nx, ny, _, _, _ in frontier]))
        
        # Add new frontier cells
        for direction, dx, dy in directions:
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].visited and not grid[nx][ny].is_frontier:
                frontier.append((nx, ny, fx, fy, direction))
                grid[nx][ny].is_frontier = True
                
        # Capture state with updated frontier
        generation_states.append(capture_generation_state(grid, grid[fx][fy], [], [], [(fx, fy)], frontier_cells=[(nx, ny) for nx, ny, _, _, _ in frontier]))

    # Create entrance and exit
    entrance_x = random.randint(1, width - 2)
    exit_x = random.randint(1, width - 2)
    grid[entrance_x][0].walls['N'] = False
    grid[exit_x][height - 1].walls['S'] = False
    grid[entrance_x][0].visited = True
    grid[exit_x][height - 1].visited = True
    grid[entrance_x][0].is_start = True
    grid[exit_x][height - 1].is_end = True
    entrance_pos = (entrance_x, 0)
    exit_pos = (exit_x, height - 1)

    # Ensure outer walls are intact except for entrance and exit
    for x in range(width):
        for y in range(height):
            if y == 0 and x != entrance_x:
                grid[x][y].walls['N'] = True
            if y == height - 1 and x != exit_x:
                grid[x][y].walls['S'] = True
            if x == 0:
                grid[x][y].walls['W'] = True
            if x == width - 1:
                grid[x][y].walls['E'] = True

    generation_states.append(capture_generation_state(grid, None, [], []))
    return grid, generation_states, cells_added, entrance_pos, exit_pos


def capture_generation_state(grid, current_cell, path, walls, active_cells=None, frontier_cells=None, is_wilson_walk=False):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'is_frontier': [[cell.is_frontier for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'in_current_path': [[cell.in_current_path for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'walls_to_check': walls,
        'phase': 'generation',
        'active_cells': active_cells or [],
        'frontier_cells': frontier_cells or [],
        'is_wilson_walk': is_wilson_walk
    }
    return state


def create_gradient_color(order, total):
    if order < 0:
        return (0.1, 0.1, 0.2)
    norm_pos = min(1.0, order / total)
    h1, s1, v1 = 0.11, 0.9, 0.8
    h2, s2, v2 = 0.65, 0.85, 0.9
    h = h1 + (h2 - h1) * norm_pos
    s = s1 + (s2 - s1) * norm_pos
    v = v1 + (v2 - v1) * norm_pos
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def solve_maze_dijkstra(grid, start_pos, end_pos):
    width = len(grid)
    height = len(grid[0])
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    for x in range(width):
        for y in range(height):
            grid[x][y].visited = False
            grid[x][y].in_path = False
            grid[x][y].distance = float('inf')

    grid[start_x][start_y].is_start = True
    grid[end_x][end_y].is_end = True
    grid[start_x][start_y].distance = 0

    priority_queue = [(0, start_pos)]
    visited = set()
    came_from = {}
    solving_states = []
    exploration_path = [grid[start_x][start_y]]

    active_cells = [(start_x, start_y)]
    solving_states.append(capture_solving_state(grid, grid[start_x][start_y],
                                             exploration_path, [], False, active_cells))

    solution_found = False
    final_path = []

    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]

    distance_grid = [[float('inf') for _ in range(height)] for _ in range(width)]
    distance_grid[start_x][start_y] = 0

    while priority_queue and not solution_found:
        _, current_pos = heapq.heappop(priority_queue)

        if current_pos in visited:
            continue

        x, y = current_pos
        visited.add(current_pos)
        grid[x][y].visited = True

        current_distance = distance_grid[x][y]

        exploration_path.append(grid[x][y])
        if len(exploration_path) > 6:  # Reduced for performance
            exploration_path = exploration_path[-6:]

        if current_pos == end_pos:
            solution_found = True
            temp_pos = current_pos
            while temp_pos in came_from:
                final_path.append(temp_pos)
                temp_pos = came_from[temp_pos]
            final_path.append(start_pos)
            final_path.reverse()

            active_cells = [(end_x, end_y)]
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                                      exploration_path, [], False, active_cells))
            continue

        frontier_positions = []
        active_cells = [(x, y)]

        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if (direction == 'N' and not grid[x][y].walls['N']) or \
                   (direction == 'S' and not grid[x][y].walls['S']) or \
                   (direction == 'E' and not grid[x][y].walls['E']) or \
                   (direction == 'W' and not grid[x][y].walls['W']):

                    new_distance = current_distance + 1

                    if new_distance < distance_grid[nx][ny]:
                        distance_grid[nx][ny] = new_distance
                        came_from[(nx, ny)] = (x, y)

                        heapq.heappush(priority_queue, (new_distance, (nx, ny)))
                        frontier_positions.append((nx, ny))
                    active_cells.append((nx, ny))

        if not solution_found:
            # Capture fewer states for better performance
            if len(solving_states) % 8 == 0:
                solving_states.append(capture_solving_state(grid, grid[x][y],
                                    exploration_path, frontier_positions, False, active_cells,
                                    distance_grid=distance_grid))

    if solution_found:
        # Capture path progression with fewer states
        step = max(1, len(final_path) // 8)
        for i in range(0, len(final_path) + 1, step):
            partial_path = []
            for j in range(min(i, len(final_path))):
                pos = final_path[j]
                partial_path.append(grid[pos[0]][pos[1]])
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                partial_path, [], True))

        for pos in final_path:
            px, py = pos
            grid[px][py].in_path = True

        final_path_cells = [grid[p[0]][p[1]] for p in final_path]
        for _ in range(2):  # Reduced for performance
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                final_path_cells, [], True))

    return solving_states


def capture_solving_state(grid, current_cell, path, frontier_positions, is_solution_phase=False, active_cells=None, distance_grid=None):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path or cell in path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'frontier_positions': frontier_positions,
        'phase': 'solving',
        'is_solution_phase': is_solution_phase,
        'active_cells': active_cells or []
    }

    if distance_grid:
        state['distance_grid'] = distance_grid

    return state


def create_animation_frames(generation_states, solving_states):
    frames = []
    
    # Sample generation states for better performance
    gen_total_states = len(generation_states)
    step = max(1, gen_total_states // GEN_FRAMES)
    for frame_idx in range(0, gen_total_states, step):
        if frame_idx < gen_total_states:
            frames.append(generation_states[frame_idx])
    
    # Fill any remaining frames with the last state
    while len(frames) < GEN_FRAMES:
        frames.append(generation_states[-1])
    
    # Sample solving states for better performance
    solve_total_states = len(solving_states)
    step = max(1, solve_total_states // SOLVE_FRAMES)
    for frame_idx in range(0, solve_total_states, step):
        if frame_idx < solve_total_states:
            frames.append(solving_states[frame_idx])
    
    # Fill any remaining frames with the last state
    while len(frames) < GEN_FRAMES + SOLVE_FRAMES:
        frames.append(solving_states[-1])
    
    return frames


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))


def create_animation(frames, width, height, total_cells, algorithm_name, solving_method, gen_particle_color, solving_particle_color):
    # Calculate cell size to fit the maze in the figure
    cell_size = min(FIG_HEIGHT / height, FIG_WIDTH / width) * 0.85
    fig, axes = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    
    # Center the maze in the figure
    x_offset = (FIG_WIDTH - width * cell_size) / 2
    y_offset = (FIG_HEIGHT - height * cell_size) / 2
    
    particles = []

    def get_random_direction():
        angle = random.uniform(0, 2 * np.pi)
        return np.cos(angle) * 0.02, np.sin(angle) * 0.02

    # Pre-calculate cell positions for performance
    cell_positions = {}
    for x in range(width):
        for y in range(height):
            cell_x = x_offset + x * cell_size
            cell_y = y_offset + (height - 1 - y) * cell_size
            cell_positions[(x, y)] = (cell_x, cell_y)

    def update(i):
        nonlocal particles
        if i >= len(frames):
            return

        axes.clear()
        axes.set_xlim(0, FIG_WIDTH)
        axes.set_ylim(0, FIG_HEIGHT)
        axes.set_facecolor(BG_COLOR)
        axes.axis('off')

        frame = frames[i]
        walls_data = frame['walls']
        visited_data = frame['visited']
        in_path_data = frame['in_path']
        is_start_data = frame['is_start']
        is_end_data = frame['is_end']
        generation_order_data = frame['generation_order']
        current_pos = frame['current']
        path_positions = frame['path']
        phase = frame['phase']

        walls_to_check = frame.get('walls_to_check', [])
        in_current_path_data = frame.get('in_current_path') if phase == 'generation' else None
        is_wilson_walk = frame.get('is_wilson_walk', False) if phase == 'generation' else False
        frontier_positions = frame.get('frontier_positions', []) if phase == 'solving' else []
        distance_grid = frame.get('distance_grid') if phase == 'solving' else None
        generation_phase = phase == 'generation'
        frontier_cells = frame.get('frontier_cells', []) if generation_phase else []

        if generation_phase:
            progress_percent = min(100, int((i / max(1, GEN_FRAMES - 1)) * 100))
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 1.0,
                      "Maze Generation",
                      color='white', fontsize=16, ha='center', weight='bold')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.6,
                      f"Algorithm: {algorithm_name}",
                      color='white', fontsize=10, ha='center')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.3,
                      f"Progress: {progress_percent}%",
                      color='white', fontsize=10, ha='center')
        else:
            is_solution_phase = frame.get('is_solution_phase', False)
            title_text = "Solution Path" if is_solution_phase else "Exploring Maze"
            progress_percent = min(100, int(((i - GEN_FRAMES) / max(1, SOLVE_FRAMES - 1)) * 100))
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 1.0,
                      title_text,
                      color='white', fontsize=16, ha='center', weight='bold')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.6,
                      f"Algorithm: {solving_method}",
                      color='white', fontsize=10, ha='center')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.3,
                      f"Progress: {progress_percent}%",
                      color='white', fontsize=10, ha='center')

        pulse_factor = 1.0 + 0.1 * np.sin(i * PULSE_SPEED)
        active_cells = frame.get('active_cells', [])

        # Drastically reduced particle effects for better performance
        if generation_phase and is_wilson_walk and in_current_path_data:
            path_cells = [(x, y) for x in range(width) for y in range(height) if in_current_path_data[x][y]]
            if path_cells and random.random() < 0.2:  # Reduced particle frequency
                cell_idx = len(path_cells) - 1
                ax, ay = path_cells[cell_idx]
                cell_center_x, cell_center_y = cell_positions[(ax, ay)]
                cell_center_x += cell_size / 2
                cell_center_y += cell_size / 2

                progress = cell_idx / max(1, len(path_cells) - 1)
                h = 0.11 + progress * 0.15
                s = 0.9 - progress * 0.1
                v = 0.8 + progress * 0.2
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                cell_color = (r, g, b)

                if random.random() < 0.05:  # Very reduced particle frequency
                    particles.append(Particle(
                        cell_center_x + random.uniform(-0.2, 0.2) * cell_size,
                        cell_center_y + random.uniform(-0.2, 0.2) * cell_size,
                        cell_color,
                        size=random.uniform(0.8, 1.2) * PARTICLE_SIZE * 0.6,
                        lifetime=random.randint(2, 4)
                    ))

        if generation_phase and not is_wilson_walk:
            for ax, ay in active_cells:
                if random.random() < 0.15:  # Reduced particle frequency
                    cell_center_x, cell_center_y = cell_positions[(ax, ay)]
                    cell_center_x += cell_size / 2
                    cell_center_y += cell_size / 2
                    particle_color = hex_to_rgb(gen_particle_color)

                    for _ in range(random.randint(1, 2)):  # Reduced particles
                        vel_x, vel_y = get_random_direction()
                        particles.append(Particle(
                            cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                            cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                            particle_color,
                            size=random.uniform(1.0, 1.5) * PARTICLE_SIZE,
                            lifetime=random.randint(3, 5),
                            velocity=(vel_x, vel_y)
                        ))

        if not generation_phase:
            for ax, ay in active_cells:
                if random.random() < 0.15:  # Reduced particle frequency
                    cell_center_x, cell_center_y = cell_positions[(ax, ay)]
                    cell_center_x += cell_size / 2
                    cell_center_y += cell_size / 2

                    particle_color = hex_to_rgb(solving_particle_color)
                    if distance_grid and distance_grid[ax][ay] != float('inf'):
                        max_dist_list = [d for row in distance_grid for d in row if d != float('inf')]
                        max_distance = max(max_dist_list) if max_dist_list else 1
                        norm_distance = distance_grid[ax][ay] / max(1, max_distance)

                        h1, s1, v1 = 0.58, 0.9, 0.9
                        h2, s2, v2 = 0.85, 0.9, 0.9
                        h = h1 + (h2 - h1) * norm_distance
                        s = s1
                        v = v1
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        particle_color = (r, g, b)

                    for _ in range(random.randint(1, 2)):  # Reduced particles
                        vel_x, vel_y = get_random_direction()
                        particles.append(Particle(
                            cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                            cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                            particle_color,
                            size=random.uniform(1.0, 1.5) * PARTICLE_SIZE,
                            lifetime=random.randint(3, 5),
                            velocity=(vel_x, vel_y)
                        ))

        # Draw cells
        for x in range(width):
            for y in range(height):
                cell_x, cell_y = cell_positions[(x, y)]
                is_current = current_pos and (x, y) == current_pos
                in_path = in_path_data[x][y]
                visited = visited_data[x][y]
                is_start = is_start_data[x][y]
                is_end = is_end_data[x][y]
                generation_order = generation_order_data[x][y]
                in_current_path = in_current_path_data and in_current_path_data[x][y]
                is_frontier_wall = any((x, y) == (wx, wy) for wx, wy, _ in walls_to_check) if generation_phase else False
                in_frontier = not generation_phase and (x, y) in frontier_positions
                is_frontier_cell = (x, y) in frontier_cells

                glow = False
                glow_size_factor = 1.0
                cell_color = 'white'
                alpha = 0.05
                zorder = 1

                if is_start:
                    cell_color = START_COLOR
                    alpha = 1.0
                    zorder = 25
                    glow = True
                    glow_color = START_COLOR
                    glow_size_factor = 1.2 * pulse_factor
                elif is_end:
                    cell_color = END_COLOR
                    alpha = 1.0
                    zorder = 25
                    glow = True
                    glow_color = END_COLOR
                    glow_size_factor = 1.2 * pulse_factor
                elif is_current:
                    cell_color = gen_particle_color if generation_phase else solving_particle_color
                    alpha = 0.9
                    zorder = 20
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.15 * pulse_factor
                elif in_current_path:
                    h, s, v = 0.15, 0.8, 0.9
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                    alpha = 0.7
                    zorder = 15
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                elif is_frontier_wall and generation_phase:
                    cell_color = '#AAAAFF'
                    alpha = 0.6
                    zorder = 15
                elif in_frontier:
                    cell_color = solving_particle_color
                    alpha = 0.7
                    zorder = 18
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                    if distance_grid and distance_grid[x][y] != float('inf'):
                         max_dist_list = [d for row in distance_grid for d in row if d != float('inf')]
                         max_distance = max(max_dist_list) if max_dist_list else 1
                         norm_distance = distance_grid[x][y] / max(1, max_distance)
                         h1, s1, v1 = 0.58, 0.9, 0.9
                         h2, s2, v2 = 0.85, 0.9, 0.9
                         h = h1 + (h2 - h1) * norm_distance
                         s = s1
                         v = v1
                         r, g, b = colorsys.hsv_to_rgb(h, s, v)
                         cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                         glow_color = cell_color # Update glow color too
                elif is_frontier_cell:
                    cell_color = '#FF9500'
                    alpha = 0.6
                    zorder = 12
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                elif in_path:
                    cell_color = PATH_COLOR
                    is_final_solution_frame = not generation_phase and i >= GEN_FRAMES + SOLVE_FRAMES - 30
                    alpha = 1.0 if is_final_solution_frame else 0.8
                    glow = True
                    glow_color = PATH_COLOR
                    glow_size_factor = (1.1 if is_final_solution_frame else 1.05) * pulse_factor
                    zorder = 15
                elif visited:
                    alpha = 0.7
                    zorder = 8 if not generation_phase else 5
                    if generation_phase:
                         cell_color = create_gradient_color(generation_order, total_cells) if generation_order >= 0 else (0.2, 0.3, 0.5)
                    else: # Solving phase visited
                        cell_color = VISITED_COLOR
                        if distance_grid and distance_grid[x][y] != float('inf'):
                            max_dist_list = [d for row in distance_grid for d in row if d != float('inf')]
                            max_distance = max(max_dist_list) if max_dist_list else 1
                            norm_distance = distance_grid[x][y] / max(1, max_distance)
                            h1, s1, v1 = 0.6, 0.7, 0.4
                            h2, s2, v2 = 0.7, 0.6, 0.6
                            h = h1 + (h2 - h1) * norm_distance
                            s = s1 + (s2 - s1) * norm_distance
                            v = v1 + (v2 - v1) * norm_distance
                            r, g, b = colorsys.hsv_to_rgb(h, s, v)
                            cell_color = (r, g, b)

                cell_rect = patches.Rectangle(
                    (cell_x, cell_y), cell_size, cell_size,
                    fill=True, color=cell_color, alpha=alpha,
                    linewidth=0, zorder=zorder
                )
                axes.add_patch(cell_rect)

                if glow:
                    glow_size = cell_size * glow_size_factor
                    glow_offset = (cell_size - glow_size) / 2
                    glow_rect = patches.Rectangle(
                        (cell_x + glow_offset, cell_y + glow_offset),
                        glow_size, glow_size,
                        fill=False, edgecolor=glow_color, alpha=0.4,
                        linewidth=1.5,
                        zorder=zorder - 1
                    )
                    axes.add_patch(glow_rect)

        # Update and draw particles
        new_particles = []
        for particle in particles:
            if particle.update():
                new_particles.append(particle)
                circle = plt.Circle(
                    (particle.x, particle.y),
                    particle.size * cell_size * 0.07,
                    color=particle.color,
                    alpha=particle.lifetime / particle.max_lifetime,
                    zorder=100
                )
                axes.add_patch(circle)
        particles = new_particles

        # Draw walls
        for x in range(width):
            for y in range(height):
                walls = walls_data[x][y]
                cell_x, cell_y = cell_positions[(x, y)]
                wall_color = WALL_COLOR
                line_width = 1.0
                alpha_wall = 0.7
                zorder_wall = 30
                if walls['N']:
                    axes.plot([cell_x, cell_x + cell_size],
                              [cell_y + cell_size, cell_y + cell_size],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)
                if walls['E']:
                    axes.plot([cell_x + cell_size, cell_x + cell_size],
                              [cell_y, cell_y + cell_size],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)
                if walls['S']:
                    axes.plot([cell_x, cell_x + cell_size],
                              [cell_y, cell_y],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)
                if walls['W']:
                    axes.plot([cell_x, cell_x],
                              [cell_y, cell_y + cell_size],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)

    # Reduce the number of frames for better performance
    display_frames = min(TOTAL_FRAMES, len(frames))
    
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=display_frames,
        blit=False,
        interval=1000 / TARGET_FPS,
        repeat=False
    )

    return ani, fig


# Signal handler to properly close the animation
def signal_handler(sig, frame):
    plt.close('all')
    sys.exit(0)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
from collections import deque
import colorsys
import heapq
import time
import signal
import sys

np.random.seed(42)

# Configuration constants - adjusted for better visibility
FIG_WIDTH = 12  # Increased for better maze visibility
FIG_HEIGHT = 12  # Square aspect ratio
DPI = 80  # Slightly reduced for performance but maintaining quality

MAZE_WIDTH = 25
MAZE_HEIGHT = 25

# Color definitions
BG_COLOR = '#0A0A15'
WALL_COLOR = '#FFFFFF'
CURRENT_COLOR = '#FF1493'
GENERATION_PARTICLE = '#00FF7F'
SOLVING_PARTICLE = '#1E90FF'
VISITED_COLOR = '#3A1C71'
PATH_COLOR = '#FF3333'
START_COLOR = '#00FF7F'
END_COLOR = '#FF4500'

PRIM_GENERATION_PARTICLE = '#FF9500'
DIJKSTRA_SOLVING_PARTICLE = '#00EEFF'

# Reduced particle effects for better performance
PARTICLE_LIFETIME = 3  # Reduced for performance
PARTICLE_SIZE = 0.8    # Reduced for performance

GEN_DURATION = 3       # Reduced for faster generation
SOLVE_DURATION = 4     # Reduced for faster solving
TOTAL_DURATION = GEN_DURATION + SOLVE_DURATION
TARGET_FPS = 30        # Reduced from 60 to 30 for better performance
GEN_FRAMES = GEN_DURATION * TARGET_FPS
SOLVE_FRAMES = SOLVE_DURATION * TARGET_FPS
TOTAL_FRAMES = TOTAL_DURATION * TARGET_FPS

GLOW_INTENSITY = 1.2
PULSE_SPEED = 0.15


class Particle:
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME, velocity=(0, 0)):
        self.x = x
        self.y = y
        self.original_color = color
        self.color = color
        self.size = size * random.uniform(0.8, 1.2)
        self.max_lifetime = lifetime
        self.lifetime = lifetime
        self.decay = 0.85
        self.vx = velocity[0] * random.uniform(0.5, 1.5)
        self.vy = velocity[1] * random.uniform(0.5, 1.5)

    def update(self):
        self.lifetime -= 1
        self.size *= self.decay
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.9
        self.vy *= 0.9

        alpha = self.lifetime / self.max_lifetime
        r, g, b = self.original_color
        self.color = (r, g, b, alpha)
        return self.lifetime > 0 and self.size > 0.1


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'N': True, 'E': True, 'S': True, 'W': True}
        self.visited = False
        self.in_path = False
        self.is_start = False
        self.is_end = False
        self.is_frontier = False
        self.generation_order = -1
        self.distance = float('inf')
        self.in_current_path = False


def create_maze_prim(width, height):
    grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
    generation_states = []
    cells_added = 0

    # Start with a random cell
    start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
    grid[start_x][start_y].visited = True
    grid[start_x][start_y].generation_order = cells_added
    cells_added += 1

    # Add the initial cell to the maze
    generation_states.append(capture_generation_state(grid, grid[start_x][start_y], [], [], [(start_x, start_y)]))

    # Create a list of frontier cells (neighbors of visited cells)
    frontier = []
    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]
    opposite = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

    # Add the initial frontier cells
    for direction, dx, dy in directions:
        nx, ny = start_x + dx, start_y + dy
        if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].visited:
            frontier.append((nx, ny, start_x, start_y, direction))
            grid[nx][ny].is_frontier = True

    # Capture state with frontier
    generation_states.append(capture_generation_state(grid, grid[start_x][start_y], [], [], frontier_cells=[(nx, ny) for nx, ny, _, _, _ in frontier]))

    while frontier:
        # Randomly select a frontier cell
        idx = random.randint(0, len(frontier) - 1)
        fx, fy, px, py, dir_from_parent = frontier.pop(idx)
        
        # Mark the frontier cell as visited and remove it from frontier
        grid[fx][fy].visited = True
        grid[fx][fy].generation_order = cells_added
        cells_added += 1
        grid[fx][fy].is_frontier = False
        
        # Remove the wall between the frontier cell and its parent
        grid[px][py].walls[dir_from_parent] = False
        grid[fx][fy].walls[opposite[dir_from_parent]] = False
        
        # Capture state after connecting the cell
        generation_states.append(capture_generation_state(grid, grid[fx][fy], [], [], [(fx, fy)], frontier_cells=[(nx, ny) for nx, ny, _, _, _ in frontier]))
        
        # Add new frontier cells
        for direction, dx, dy in directions:
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].visited and not grid[nx][ny].is_frontier:
                frontier.append((nx, ny, fx, fy, direction))
                grid[nx][ny].is_frontier = True
                
        # Capture state with updated frontier
        generation_states.append(capture_generation_state(grid, grid[fx][fy], [], [], [(fx, fy)], frontier_cells=[(nx, ny) for nx, ny, _, _, _ in frontier]))

    # Create entrance and exit
    entrance_x = random.randint(1, width - 2)
    exit_x = random.randint(1, width - 2)
    grid[entrance_x][0].walls['N'] = False
    grid[exit_x][height - 1].walls['S'] = False
    grid[entrance_x][0].visited = True
    grid[exit_x][height - 1].visited = True
    grid[entrance_x][0].is_start = True
    grid[exit_x][height - 1].is_end = True
    entrance_pos = (entrance_x, 0)
    exit_pos = (exit_x, height - 1)

    # Ensure outer walls are intact except for entrance and exit
    for x in range(width):
        for y in range(height):
            if y == 0 and x != entrance_x:
                grid[x][y].walls['N'] = True
            if y == height - 1 and x != exit_x:
                grid[x][y].walls['S'] = True
            if x == 0:
                grid[x][y].walls['W'] = True
            if x == width - 1:
                grid[x][y].walls['E'] = True

    generation_states.append(capture_generation_state(grid, None, [], []))
    return grid, generation_states, cells_added, entrance_pos, exit_pos


def capture_generation_state(grid, current_cell, path, walls, active_cells=None, frontier_cells=None, is_wilson_walk=False):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'is_frontier': [[cell.is_frontier for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'in_current_path': [[cell.in_current_path for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'walls_to_check': walls,
        'phase': 'generation',
        'active_cells': active_cells or [],
        'frontier_cells': frontier_cells or [],
        'is_wilson_walk': is_wilson_walk
    }
    return state


def create_gradient_color(order, total):
    if order < 0:
        return (0.1, 0.1, 0.2)
    norm_pos = min(1.0, order / total)
    h1, s1, v1 = 0.11, 0.9, 0.8
    h2, s2, v2 = 0.65, 0.85, 0.9
    h = h1 + (h2 - h1) * norm_pos
    s = s1 + (s2 - s1) * norm_pos
    v = v1 + (v2 - v1) * norm_pos
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def solve_maze_dijkstra(grid, start_pos, end_pos):
    width = len(grid)
    height = len(grid[0])
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    for x in range(width):
        for y in range(height):
            grid[x][y].visited = False
            grid[x][y].in_path = False
            grid[x][y].distance = float('inf')

    grid[start_x][start_y].is_start = True
    grid[end_x][end_y].is_end = True
    grid[start_x][start_y].distance = 0

    priority_queue = [(0, start_pos)]
    visited = set()
    came_from = {}
    solving_states = []
    exploration_path = [grid[start_x][start_y]]

    active_cells = [(start_x, start_y)]
    solving_states.append(capture_solving_state(grid, grid[start_x][start_y],
                                             exploration_path, [], False, active_cells))

    solution_found = False
    final_path = []

    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]

    distance_grid = [[float('inf') for _ in range(height)] for _ in range(width)]
    distance_grid[start_x][start_y] = 0

    while priority_queue and not solution_found:
        _, current_pos = heapq.heappop(priority_queue)

        if current_pos in visited:
            continue

        x, y = current_pos
        visited.add(current_pos)
        grid[x][y].visited = True

        current_distance = distance_grid[x][y]

        exploration_path.append(grid[x][y])
        if len(exploration_path) > 6:  # Reduced for performance
            exploration_path = exploration_path[-6:]

        if current_pos == end_pos:
            solution_found = True
            temp_pos = current_pos
            while temp_pos in came_from:
                final_path.append(temp_pos)
                temp_pos = came_from[temp_pos]
            final_path.append(start_pos)
            final_path.reverse()

            active_cells = [(end_x, end_y)]
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                                      exploration_path, [], False, active_cells))
            continue

        frontier_positions = []
        active_cells = [(x, y)]

        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if (direction == 'N' and not grid[x][y].walls['N']) or \
                   (direction == 'S' and not grid[x][y].walls['S']) or \
                   (direction == 'E' and not grid[x][y].walls['E']) or \
                   (direction == 'W' and not grid[x][y].walls['W']):

                    new_distance = current_distance + 1

                    if new_distance < distance_grid[nx][ny]:
                        distance_grid[nx][ny] = new_distance
                        came_from[(nx, ny)] = (x, y)

                        heapq.heappush(priority_queue, (new_distance, (nx, ny)))
                        frontier_positions.append((nx, ny))
                    active_cells.append((nx, ny))

        if not solution_found:
            # Capture fewer states for better performance
            if len(solving_states) % 8 == 0:
                solving_states.append(capture_solving_state(grid, grid[x][y],
                                    exploration_path, frontier_positions, False, active_cells,
                                    distance_grid=distance_grid))

    if solution_found:
        # Capture path progression with fewer states
        step = max(1, len(final_path) // 8)
        for i in range(0, len(final_path) + 1, step):
            partial_path = []
            for j in range(min(i, len(final_path))):
                pos = final_path[j]
                partial_path.append(grid[pos[0]][pos[1]])
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                partial_path, [], True))

        for pos in final_path:
            px, py = pos
            grid[px][py].in_path = True

        final_path_cells = [grid[p[0]][p[1]] for p in final_path]
        for _ in range(2):  # Reduced for performance
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                final_path_cells, [], True))

    return solving_states


def capture_solving_state(grid, current_cell, path, frontier_positions, is_solution_phase=False, active_cells=None, distance_grid=None):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path or cell in path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'frontier_positions': frontier_positions,
        'phase': 'solving',
        'is_solution_phase': is_solution_phase,
        'active_cells': active_cells or []
    }

    if distance_grid:
        state['distance_grid'] = distance_grid

    return state


def create_animation_frames(generation_states, solving_states):
    frames = []
    
    # Sample generation states for better performance
    gen_total_states = len(generation_states)
    step = max(1, gen_total_states // GEN_FRAMES)
    for frame_idx in range(0, gen_total_states, step):
        if frame_idx < gen_total_states:
            frames.append(generation_states[frame_idx])
    
    # Fill any remaining frames with the last state
    while len(frames) < GEN_FRAMES:
        frames.append(generation_states[-1])
    
    # Sample solving states for better performance
    solve_total_states = len(solving_states)
    step = max(1, solve_total_states // SOLVE_FRAMES)
    for frame_idx in range(0, solve_total_states, step):
        if frame_idx < solve_total_states:
            frames.append(solving_states[frame_idx])
    
    # Fill any remaining frames with the last state
    while len(frames) < GEN_FRAMES + SOLVE_FRAMES:
        frames.append(solving_states[-1])
    
    return frames


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))


def create_animation(frames, width, height, total_cells, algorithm_name, solving_method, gen_particle_color, solving_particle_color):
    # Calculate cell size to fit the maze in the figure
    cell_size = min(FIG_HEIGHT / height, FIG_WIDTH / width) * 0.85
    fig, axes = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    
    # Center the maze in the figure
    x_offset = (FIG_WIDTH - width * cell_size) / 2
    y_offset = (FIG_HEIGHT - height * cell_size) / 2
    
    particles = []

    def get_random_direction():
        angle = random.uniform(0, 2 * np.pi)
        return np.cos(angle) * 0.02, np.sin(angle) * 0.02

    # Pre-calculate cell positions for performance
    cell_positions = {}
    for x in range(width):
        for y in range(height):
            cell_x = x_offset + x * cell_size
            cell_y = y_offset + (height - 1 - y) * cell_size
            cell_positions[(x, y)] = (cell_x, cell_y)

    def update(i):
        nonlocal particles
        if i >= len(frames):
            return

        axes.clear()
        axes.set_xlim(0, FIG_WIDTH)
        axes.set_ylim(0, FIG_HEIGHT)
        axes.set_facecolor(BG_COLOR)
        axes.axis('off')

        frame = frames[i]
        walls_data = frame['walls']
        visited_data = frame['visited']
        in_path_data = frame['in_path']
        is_start_data = frame['is_start']
        is_end_data = frame['is_end']
        generation_order_data = frame['generation_order']
        current_pos = frame['current']
        path_positions = frame['path']
        phase = frame['phase']

        walls_to_check = frame.get('walls_to_check', [])
        in_current_path_data = frame.get('in_current_path') if phase == 'generation' else None
        is_wilson_walk = frame.get('is_wilson_walk', False) if phase == 'generation' else False
        frontier_positions = frame.get('frontier_positions', []) if phase == 'solving' else []
        distance_grid = frame.get('distance_grid') if phase == 'solving' else None
        generation_phase = phase == 'generation'
        frontier_cells = frame.get('frontier_cells', []) if generation_phase else []

        if generation_phase:
            progress_percent = min(100, int((i / max(1, GEN_FRAMES - 1)) * 100))
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 1.0,
                      "Maze Generation",
                      color='white', fontsize=16, ha='center', weight='bold')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.6,
                      f"Algorithm: {algorithm_name}",
                      color='white', fontsize=10, ha='center')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.3,
                      f"Progress: {progress_percent}%",
                      color='white', fontsize=10, ha='center')
        else:
            is_solution_phase = frame.get('is_solution_phase', False)
            title_text = "Solution Path" if is_solution_phase else "Exploring Maze"
            progress_percent = min(100, int(((i - GEN_FRAMES) / max(1, SOLVE_FRAMES - 1)) * 100))
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 1.0,
                      title_text,
                      color='white', fontsize=16, ha='center', weight='bold')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.6,
                      f"Algorithm: {solving_method}",
                      color='white', fontsize=10, ha='center')
            axes.text(FIG_WIDTH / 2, y_offset + height * cell_size + 0.3,
                      f"Progress: {progress_percent}%",
                      color='white', fontsize=10, ha='center')

        pulse_factor = 1.0 + 0.1 * np.sin(i * PULSE_SPEED)
        active_cells = frame.get('active_cells', [])

        # Drastically reduced particle effects for better performance
        if generation_phase and is_wilson_walk and in_current_path_data:
            path_cells = [(x, y) for x in range(width) for y in range(height) if in_current_path_data[x][y]]
            if path_cells and random.random() < 0.2:  # Reduced particle frequency
                cell_idx = len(path_cells) - 1
                ax, ay = path_cells[cell_idx]
                cell_center_x, cell_center_y = cell_positions[(ax, ay)]
                cell_center_x += cell_size / 2
                cell_center_y += cell_size / 2

                progress = cell_idx / max(1, len(path_cells) - 1)
                h = 0.11 + progress * 0.15
                s = 0.9 - progress * 0.1
                v = 0.8 + progress * 0.2
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                cell_color = (r, g, b)

                if random.random() < 0.05:  # Very reduced particle frequency
                    particles.append(Particle(
                        cell_center_x + random.uniform(-0.2, 0.2) * cell_size,
                        cell_center_y + random.uniform(-0.2, 0.2) * cell_size,
                        cell_color,
                        size=random.uniform(0.8, 1.2) * PARTICLE_SIZE * 0.6,
                        lifetime=random.randint(2, 4)
                    ))

        if generation_phase and not is_wilson_walk:
            for ax, ay in active_cells:
                if random.random() < 0.15:  # Reduced particle frequency
                    cell_center_x, cell_center_y = cell_positions[(ax, ay)]
                    cell_center_x += cell_size / 2
                    cell_center_y += cell_size / 2
                    particle_color = hex_to_rgb(gen_particle_color)

                    for _ in range(random.randint(1, 2)):  # Reduced particles
                        vel_x, vel_y = get_random_direction()
                        particles.append(Particle(
                            cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                            cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                            particle_color,
                            size=random.uniform(1.0, 1.5) * PARTICLE_SIZE,
                            lifetime=random.randint(3, 5),
                            velocity=(vel_x, vel_y)
                        ))

        if not generation_phase:
            for ax, ay in active_cells:
                if random.random() < 0.15:  # Reduced particle frequency
                    cell_center_x, cell_center_y = cell_positions[(ax, ay)]
                    cell_center_x += cell_size / 2
                    cell_center_y += cell_size / 2

                    particle_color = hex_to_rgb(solving_particle_color)
                    if distance_grid and distance_grid[ax][ay] != float('inf'):
                        max_dist_list = [d for row in distance_grid for d in row if d != float('inf')]
                        max_distance = max(max_dist_list) if max_dist_list else 1
                        norm_distance = distance_grid[ax][ay] / max(1, max_distance)

                        h1, s1, v1 = 0.58, 0.9, 0.9
                        h2, s2, v2 = 0.85, 0.9, 0.9
                        h = h1 + (h2 - h1) * norm_distance
                        s = s1
                        v = v1
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        particle_color = (r, g, b)

                    for _ in range(random.randint(1, 2)):  # Reduced particles
                        vel_x, vel_y = get_random_direction()
                        particles.append(Particle(
                            cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                            cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                            particle_color,
                            size=random.uniform(1.0, 1.5) * PARTICLE_SIZE,
                            lifetime=random.randint(3, 5),
                            velocity=(vel_x, vel_y)
                        ))

        # Draw cells
        for x in range(width):
            for y in range(height):
                cell_x, cell_y = cell_positions[(x, y)]
                is_current = current_pos and (x, y) == current_pos
                in_path = in_path_data[x][y]
                visited = visited_data[x][y]
                is_start = is_start_data[x][y]
                is_end = is_end_data[x][y]
                generation_order = generation_order_data[x][y]
                in_current_path = in_current_path_data and in_current_path_data[x][y]
                is_frontier_wall = any((x, y) == (wx, wy) for wx, wy, _ in walls_to_check) if generation_phase else False
                in_frontier = not generation_phase and (x, y) in frontier_positions
                is_frontier_cell = (x, y) in frontier_cells

                glow = False
                glow_size_factor = 1.0
                cell_color = 'white'
                alpha = 0.05
                zorder = 1

                if is_start:
                    cell_color = START_COLOR
                    alpha = 1.0
                    zorder = 25
                    glow = True
                    glow_color = START_COLOR
                    glow_size_factor = 1.2 * pulse_factor
                elif is_end:
                    cell_color = END_COLOR
                    alpha = 1.0
                    zorder = 25
                    glow = True
                    glow_color = END_COLOR
                    glow_size_factor = 1.2 * pulse_factor
                elif is_current:
                    cell_color = gen_particle_color if generation_phase else solving_particle_color
                    alpha = 0.9
                    zorder = 20
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.15 * pulse_factor
                elif in_current_path:
                    h, s, v = 0.15, 0.8, 0.9
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                    alpha = 0.7
                    zorder = 15
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                elif is_frontier_wall and generation_phase:
                    cell_color = '#AAAAFF'
                    alpha = 0.6
                    zorder = 15
                elif in_frontier:
                    cell_color = solving_particle_color
                    alpha = 0.7
                    zorder = 18
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                    if distance_grid and distance_grid[x][y] != float('inf'):
                         max_dist_list = [d for row in distance_grid for d in row if d != float('inf')]
                         max_distance = max(max_dist_list) if max_dist_list else 1
                         norm_distance = distance_grid[x][y] / max(1, max_distance)
                         h1, s1, v1 = 0.58, 0.9, 0.9
                         h2, s2, v2 = 0.85, 0.9, 0.9
                         h = h1 + (h2 - h1) * norm_distance
                         s = s1
                         v = v1
                         r, g, b = colorsys.hsv_to_rgb(h, s, v)
                         cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                         glow_color = cell_color # Update glow color too
                elif is_frontier_cell:
                    cell_color = '#FF9500'
                    alpha = 0.6
                    zorder = 12
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                elif in_path:
                    cell_color = PATH_COLOR
                    is_final_solution_frame = not generation_phase and i >= GEN_FRAMES + SOLVE_FRAMES - 30
                    alpha = 1.0 if is_final_solution_frame else 0.8
                    glow = True
                    glow_color = PATH_COLOR
                    glow_size_factor = (1.1 if is_final_solution_frame else 1.05) * pulse_factor
                    zorder = 15
                elif visited:
                    alpha = 0.7
                    zorder = 8 if not generation_phase else 5
                    if generation_phase:
                         cell_color = create_gradient_color(generation_order, total_cells) if generation_order >= 0 else (0.2, 0.3, 0.5)
                    else: # Solving phase visited
                        cell_color = VISITED_COLOR
                        if distance_grid and distance_grid[x][y] != float('inf'):
                            max_dist_list = [d for row in distance_grid for d in row if d != float('inf')]
                            max_distance = max(max_dist_list) if max_dist_list else 1
                            norm_distance = distance_grid[x][y] / max(1, max_distance)
                            h1, s1, v1 = 0.6, 0.7, 0.4
                            h2, s2, v2 = 0.7, 0.6, 0.6
                            h = h1 + (h2 - h1) * norm_distance
                            s = s1 + (s2 - s1) * norm_distance
                            v = v1 + (v2 - v1) * norm_distance
                            r, g, b = colorsys.hsv_to_rgb(h, s, v)
                            cell_color = (r, g, b)

                cell_rect = patches.Rectangle(
                    (cell_x, cell_y), cell_size, cell_size,
                    fill=True, color=cell_color, alpha=alpha,
                    linewidth=0, zorder=zorder
                )
                axes.add_patch(cell_rect)

                if glow:
                    glow_size = cell_size * glow_size_factor
                    glow_offset = (cell_size - glow_size) / 2
                    glow_rect = patches.Rectangle(
                        (cell_x + glow_offset, cell_y + glow_offset),
                        glow_size, glow_size,
                        fill=False, edgecolor=glow_color, alpha=0.4,
                        linewidth=1.5,
                        zorder=zorder - 1
                    )
                    axes.add_patch(glow_rect)

        # Update and draw particles
        new_particles = []
        for particle in particles:
            if particle.update():
                new_particles.append(particle)
                circle = plt.Circle(
                    (particle.x, particle.y),
                    particle.size * cell_size * 0.07,
                    color=particle.color,
                    alpha=particle.lifetime / particle.max_lifetime,
                    zorder=100
                )
                axes.add_patch(circle)
        particles = new_particles

        # Draw walls
        for x in range(width):
            for y in range(height):
                walls = walls_data[x][y]
                cell_x, cell_y = cell_positions[(x, y)]
                wall_color = WALL_COLOR
                line_width = 1.0
                alpha_wall = 0.7
                zorder_wall = 30
                if walls['N']:
                    axes.plot([cell_x, cell_x + cell_size],
                              [cell_y + cell_size, cell_y + cell_size],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)
                if walls['E']:
                    axes.plot([cell_x + cell_size, cell_x + cell_size],
                              [cell_y, cell_y + cell_size],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)
                if walls['S']:
                    axes.plot([cell_x, cell_x + cell_size],
                              [cell_y, cell_y],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)
                if walls['W']:
                    axes.plot([cell_x, cell_x],
                              [cell_y, cell_y + cell_size],
                              wall_color, linewidth=line_width, alpha=alpha_wall, zorder=zorder_wall)

    # Reduce the number of frames for better performance
    display_frames = min(TOTAL_FRAMES, len(frames))
    
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=display_frames,
        blit=False,
        interval=1000 / TARGET_FPS,
        repeat=False
    )

    return ani, fig


# Signal handler to properly close the animation
def signal_handler(sig, frame):
    plt.close('all')
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    combo = {
        "generation_algo": "create_maze_prim",
        "solving_algo": "solve_maze_dijkstra",
        "gen_name": "Prim's Algorithm",
        "solving_name": "Dijkstra's Algorithm",
        "gen_particle_color": PRIM_GENERATION_PARTICLE,
        "solving_particle_color": DIJKSTRA_SOLVING_PARTICLE,
        "output_file": "prim_dijkstra_maze.mp4"
    }

    print("≈" * 70)
    print(f"📱 GENERATING & SOLVING MAZE ANIMATION - {MAZE_WIDTH}x{MAZE_HEIGHT} MAZE 📱")
    print(f"Generation: {combo['gen_name']} | Solving: {combo['solving_name']}")
    print("≈" * 70)

    print(f"🧩 Generating maze using {combo['gen_name']} ({MAZE_WIDTH}x{MAZE_HEIGHT})...")
    start_time = time.time()
    grid, generation_states, total_cells, entrance_pos, exit_pos = globals()[combo['generation_algo']](MAZE_WIDTH, MAZE_HEIGHT)
    gen_time = time.time() - start_time
    print(f"✅ Maze generated in {gen_time:.2f} seconds")

    start_pos = entrance_pos
    end_pos = exit_pos

    print(f"🔍 Solving maze using {combo['solving_name']}...")
    start_time = time.time()
    solving_states = globals()[combo['solving_algo']](grid, start_pos, end_pos)
    solve_time = time.time() - start_time
    print(f"✅ Maze solved in {solve_time:.2f} seconds")

    print(f"🎬 Creating {TOTAL_FRAMES} animation frames...")
    frames = create_animation_frames(generation_states, solving_states)

    print(f"🎨 Building animation...")
    ani, fig = create_animation(
        frames, MAZE_WIDTH, MAZE_HEIGHT, total_cells,
        combo['gen_name'], combo['solving_name'],
        combo['gen_particle_color'], combo['solving_particle_color']
    )

    # Show the animation instead of saving it
    print("🖥️  Displaying animation...")
    plt.tight_layout()
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Animation interrupted by user")
        plt.close('all')

    print("🚀 Animation complete!")


if __name__ == "__main__":
    main()
