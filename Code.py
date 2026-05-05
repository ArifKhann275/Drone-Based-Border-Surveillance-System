import numpy as np
import matplotlib.pyplot as plt
import heapq


grid = np.zeros((10,10))
obstacles = [(2,3), (2,4), (5,4), (1,7), (2,2), (2,1), (2,0), (3,5), (3,7), (3,8), (7,9), (5,1), (5,2)]
for obs in obstacles:
    grid[obs] = 1

start = (0,0)
goal = (6,1)

def turn_penalized_astar(grid, start, goal, turn_penalty=5):
    rows, cols = grid.shape
    
    
    open_list = []
    
    heapq.heappush(open_list, (0, 0, start, (0,0))) 
    
    came_from = {}
    
    
    g_cost_dict = {(start, (0,0)): 0} 
    
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while open_list:
        f_cost, current_g, current_pos, current_dir = heapq.heappop(open_list)
        
        
        if current_pos == goal:
            path = []
            state = (current_pos, current_dir)
            while state in came_from:
                path.append(state[0])
                state = came_from[state]
            path.append(start)
            return path[::-1]
            
        for dx, dy in directions:
            neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            if 0 <= neighbor_pos[0] < rows and 0 <= neighbor_pos[1] < cols:
                if grid[neighbor_pos] == 1:
                    continue
                
                
                move_cost = 1
                turn_cost = 0
                
                
                if current_dir != (0,0) and current_dir != (dx, dy):
                    turn_cost = turn_penalty 
                    
                new_g = current_g + move_cost + turn_cost
                neighbor_state = (neighbor_pos, (dx, dy))
                
                if neighbor_state not in g_cost_dict or new_g < g_cost_dict[neighbor_state]:
                    g_cost_dict[neighbor_state] = new_g
                    
                    # Heuristic 
                    h = abs(neighbor_pos[0] - goal[0]) + abs(neighbor_pos[1] - goal[1])
                    new_f = new_g + h
                    
                    heapq.heappush(open_list, (new_f, new_g, neighbor_pos, (dx, dy)))
                    came_from[neighbor_state] = (current_pos, current_dir)
                    
    return None


optimal_path = turn_penalized_astar(grid, start, goal, turn_penalty=5)

plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap='Greys', origin='upper')

if optimal_path:
    x = [p[1] for p in optimal_path]
    y = [p[0] for p in optimal_path]
    
    plt.plot(x, y, 'g-', linewidth=4, label="Turn-Penalized A* (Optimal UAV Path)")

plt.plot(start[1], start[0], 'go', markersize=12, label="Start")
plt.plot(goal[1], goal[0], 'r*', markersize=15, label="Goal")

plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)
plt.xticks(np.arange(-0.5, 10, 1), [])
plt.yticks(np.arange(-0.5, 10, 1), [])
plt.legend()
plt.title("Energy Optimized Path (Minimizing Turns)")
plt.show()


def count_turns(path):
    turns = 0
    for i in range(2, len(path)):
        d1 = (path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1])
        d2 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        if d1 != d2:
            turns += 1
    return turns

print(f"Path Distance: {len(optimal_path)-1}")
print(f"Total Turns: {count_turns(optimal_path)}")