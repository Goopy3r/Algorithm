# Maze Generator & Solver (Wilson’s Algorithm + Dijkstra)

This project generates and solves a maze using **Wilson’s algorithm** for generation and **Dijkstra’s algorithm** for solving.  
It includes smooth **matplotlib animations** with particle effects, glowing highlights, and progress indicators.

---

## Features
- **Maze Generation**: Uses **Wilson’s Algorithm** with random walks.
- **Maze Solving**: Uses **Dijkstra’s Algorithm** for shortest path.
- **Particle Effects**: Lightweight, optimized for **60 FPS**.
- **Glow & Pulse Effects**: Start, End, Current, Path, and Frontier cells are highlighted.
- **Progress Tracking**: Displays percentage completion during generation & solving.
- **Customizable Parameters**: Maze size, colors, particle behavior, and animation speed.

---

## Configuration
You can tune the maze and animation with these constants:

| Constant | Description | Default |
|----------|-------------|---------|
| `MAZE_WIDTH` | Maze width (cells) | `25` |
| `MAZE_HEIGHT` | Maze height (cells) | `25` |
| `GEN_DURATION` | Maze generation animation duration (seconds) | `3` |
| `SOLVE_DURATION` | Maze solving animation duration (seconds) | `4` |
| `TARGET_FPS` | Frames per second | `60` |
| `PARTICLE_LIFETIME` | Particle lifetime (frames) | `3` |
| `PARTICLE_SIZE` | Particle size multiplier | `0.8` |
| `BG_COLOR` | Background color | `'#0A0A15'` |
| `START_COLOR` | Start cell glow color | `'#00FF7F'` |
| `END_COLOR` | End cell glow color | `'#FF4500'` |
| `PATH_COLOR` | Final path color | `'#FF3333'` |

---

## Algorithms
### 🔹 Wilson’s Maze Generation
- Picks a random unvisited cell.
- Performs a **random walk** until reaching the visited tree.
- Loops are erased.
- Adds the walk to the maze, ensuring a **uniform spanning tree**.

### 🔹 Dijkstra’s Maze Solving
- Computes shortest path distances from start.
- Explores cells in order of increasing distance.
- Stops once the end cell is reached.
- Final path is reconstructed and highlighted.

---

## Animation Workflow
1. **Generation Phase** (Wilson’s Algorithm)  
   - Shows active random walks.  
   - Active cells glow pink/orange.  
   - Particle effects mark exploration.  
   - Progress (%) is displayed.

2. **Solving Phase** (Dijkstra’s Algorithm)  
   - Explored cells shown in blue/purple.  
   - Frontier cells glow with distance-based gradients.  
   - Path gradually appears in red.  
   - Final solution pulses for emphasis.

---

## Example Output
- Maze size: **25×25**
- Generation: **3s**
- Solving: **4s**
- Smooth 60 FPS animation with glowing path.

---

## Usage
Run the script with:

```bash
python some-algo.py
