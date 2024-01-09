https://paul.pub/a-star-algorithm/
### Introduction to A* Algorithm
- **First Publication**: 1968 by Peter Hart, Nils Nilsson, Bertram Raphael (Stanford Research Institute).
- **Characteristics**: Extension of Dijkstra's algorithm, guided by a heuristic function for enhanced performance and accuracy.
### Algorithm Overview
- **Algorithm Type**: Pathfinding and graph traversal.
- **Language Implementation**: Python with dynamic visualization using the matplotlib library.
### Preliminary Concepts
1. **Breadth-First Search**: Expands like a flood, traversing adjacent nodes and then moving outward.
2. **Dijkstra's Algorithm**: Finds shortest paths between nodes, considering varying movement costs.
3. **Best First Search**: Uses heuristic information (estimation of distance to endpoint) for faster pathfinding but may miss the shortest path due to obstacles.
### A* Algorithm Mechanics
- **Fundamental Equation**: 
  \[f(n) = g(n) + h(n)\]
   - Where \(f(n)\) is the total priority of node n.
   - \(g(n)\) is the cost from the start node to n.
   - \(h(n)\) is the heuristic estimated cost from n to the end node.
- **Priority Queue**: Nodes with the smallest \(f(n)\) are prioritized.
- **Sets Used**: `open_set` for nodes to explore and `close_set` for explored nodes.

### Algorithm Process
1. Initialize `open_set` and `close_set`.
2. Add starting point to `open_set` with priority 0.
3. While `open_set` is not empty:
    - Choose the node with the highest priority.
    - If the node is the endpoint:
        - Backtrack from the endpoint to the start via parent nodes.
        - Return the resulting path.
    - Else:
        - Move node from `open_set` to `close_set`.
        - For each adjacent node:
            - Skip if in `close_set`.
            - Set its parent to the current node and calculate priority if not in `open_set`.
            - Add to `open_set` if not already present.
### Heuristic Function Variations
- **Manhattan Distance**: For graphs allowing only orthogonal movement.
- **Diagonal Distance**: For graphs allowing eight-directional movement.
- **Euclidean Distance**: For unrestricted directional movement.
### Distance Calculations
1. **Manhattan Distance**:
   - Function:
   - `h(node) = D * (abs(node.x - goal.x) + abs(node.y - goal.y))`
   - Applicable when only four-directional movements are allowed.
2. **Diagonal Distance**:
   - Function: 
     ```
     h(node) = D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
     ```
     - Where `D2` is the diagonal movement cost.
3. **Euclidean Distance**:
   - Function: `heuristic(node) = D * sqrt(dx * dx + dy * dy)`
   - Used for any-direction movements.

### Python Implementation Snippet
```python
import matplotlib.pyplot as plt
import numpy as np

def a_star_algorithm(start_node, end_node):
    # Implementation of A* Algorithm
    pass

# Example usage with matplotlib for visualization
start = (0, 0)
end = (5, 5)
path = a_star_algorithm(start, end)

# Visualization code using matplotlib
x, y = zip(*path)
plt.plot(x, y, '-o')
plt.show()
```

### Advantages of A* Algorithm
- Flexibility in balancing speed and accuracy via the heuristic function.
- Capable of handling various graph structures and movement constraints.
- Optimized for scenarios where a clear endpoint exists but the shortest path is not straightforward.

---
- **Visualization**: Matplotlib library
- **Source Code**: Available on GitHub (paulQuei/a-star-algorithm)
- **Application**: Pathfinding on a two-dimensional grid graph
### Point and Map Classes in Python
- **Point Class** (`point.py`):
  - Represents points in the graph.
  - Attributes: `x`, `y`, `cost`.
  - Cost initialized to maximum system value.
  ```python
  import sys

  class Point:
      def __init__(self, x, y):
          self.x = x
          self.y = y
          self.cost = sys.maxsize
  ```

- **RandomMap Class** (`random_map.py`):
  - Describes map structure.
  - Contains obstacles and random elements.
  - Utilizes numpy for randomness.
  - Methods to check if a point is an obstacle.
  ```python
  import numpy as np
  import point

  class RandomMap:
      def __init__(self, size=50):
          self.size = size
          self.obstacle = size // 8
          self.GenerateObstacle()

      def GenerateObstacle(self):
          # Code to generate obstacles...
          # Implementation details...

      def IsObstacle(self, i, j):
          # Returns True if (i, j) is an obstacle
          pass
  ```

### Implementation of A* Algorithm
- **AStar Class** (`a_star.py`):
  - Manages the A* algorithm process.
  - Uses `BaseCost`, `HeuristicCost`, and `TotalCost` methods.
  - Validates points and checks their existence in sets.
  - Determines start and end points.

  ```python
  import sys
  import point
  import random_map

  class AStar:
      def __init__(self, map):
          self.map = map
          # Additional initialization...

      def BaseCost(self, p):
          # Code for base cost calculation
          pass

      def HeuristicCost(self, p):
          # Code for heuristic cost calculation
          pass

      def TotalCost(self, p):
          # Code for total cost calculation
          pass

      # Additional methods for point validation and list checking...
  ```

- **Algorithm Core Logic**:
  - Process nodes, select the highest priority node.
  - If endpoint, build the path.
  - If not endpoint, continue with neighbor nodes.
  - Utilize `matplotlib` for visual representation.

  ```python
  # Additional methods in AStar class

  def RunAndSaveImage(self, ax, plt):
      # Code to run the algorithm and save images
      pass

  def ProcessPoint(self, x, y, parent):
      # Code to process each point
      pass

  def SelectPointInOpenList(self):
      # Code to select a point from the open list
      pass

  def BuildPath(self, p, ax, plt, start_time):
      # Code to build the final path
      pass
  ```

### Entry Logic for the Program
- **Main Logic** (`main.py`):
  - Setup map and visual elements.
  - Instantiate `AStar` class and execute the algorithm.

  ```python
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle
  import random_map
  import a_star

  # Set up matplotlib figure
  # Initialize map and plot the starting and ending points
  # Instantiate AStar and run the algorithm

  map = random_map.RandomMap()
  a_star_algo = a_star.AStar(map)
  a_star_algo.RunAndSaveImage(ax, plt)
  ```
### Algorithm Variants and Extensions
##### ARA* (Anytime Repairing A* or Anytime A*)
- **Overview**: A flexible version of A* that can adapt to time constraints.
- **Key Feature**: Generates efficient solutions quickly and refines them over time.
- **Real-World Application**: Particularly useful in scenarios where the time to solve the problem is limited, such as in real-time systems or interactive applications.
- **Methodology**: Starts with a broad search using loose constraints for fast, though suboptimal solutions. Gradually tightens constraints for improved accuracy.
- **Efficiency**: Reuses previous search efforts for better performance compared to other anytime algorithms.
- **Interruptibility**: Unique in its ability to be paused and resumed, making it versatile for dynamic environments.
##### D* (Dynamic A*)
- **Overview**: An extension of A* where the cost values can change during the algorithm's execution.
- **Variants**: Includes the original D*, Focussed D*, and D* Lite.
  - **Original D***: Developed by Anthony Stentz for robotics navigation.
  - **Focussed D***: Improves upon the original by combining A* and D* principles.
  - **D* Lite**: Based on LPA* (Lifelong Planning A*), more efficient than its predecessors.
- **Use Case**: Commonly used in mobile robots and autonomous vehicles, where the environment may change dynamically.
- **Functionality**: Adjusts the search based on new information about the environment, enabling real-time path recalculations.
- **Application**: Suitable for scenarios where the terrain or obstacles are not static and can change unexpectedly.
##### Field D*
- **Overview**: An advanced form of D* that uses interpolation for path planning.
- **Technique**: Employs linear interpolation for generating efficient paths, reducing unnecessary turns and smoothing out the route.
- **Optimality**: Assumes linear interpolation for optimal paths in most practical scenarios.
- **Usage**: Implemented in various field robotic systems where navigation across uneven terrain or complex environments is required.
- **Advantage**: Particularly effective in handling real-world terrain with varying levels of complexity.
##### Block A*
- **Overview**: An adaptation of A* that processes multiple units as a single block.
- **LDDB (Local Distance Database)**: Utilizes a database containing distances between local neighborhood boundary points.
- **Heap Value**: Each block in the open set has a heap value determining its processing order.
- **Expansion**: Uses LDDB to calculate g-values for boundary cells during block expansion.
- **Efficiency**: Well-suited for large-scale maps or environments where processing individual units would be computationally intensive.
#### Additional Considerations for Algorithm Variants
- **ARA* and Time Management**: Can be tuned for different levels of performance depending on the available computational time.
- **D* Lite in Modern Systems**: Currently, the most widely used variant in autonomous navigation due to its balance of speed and adaptability.
- **Field D* in Interpolation**: Demonstrates how interpolation techniques can enhance pathfinding algorithms beyond grid-based methods.
- **Block A* and LDDB**: Highlights the importance of pre-computed data in optimizing pathfinding tasks.
### Additional Resources
- Wikipedia: A* Variants
- Papers on ARA*, D*, Field D*, and Block A*.
---
