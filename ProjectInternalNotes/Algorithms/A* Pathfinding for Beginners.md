https://www.gamedev.net/reference/articles/article2003.asp
#### Introduction
- **Article Focus**: Introduces the A* algorithm to true beginners.
- **Aim**: To explain the fundamentals of A* and prepare readers for advanced materials.
- **Language Agnostic**: Concepts can be adapted to any programming language.
#### The Search Area
- **Scenario**: Navigating from point A to point B around obstacles.
- **Grid Representation**: The search area is divided into a square grid.
  - **Walkable/Unwalkable Squares**: Indicate possible paths.
  - **Nodes**: Center points of squares; critical for pathfinding.
- **Simplification**: Converts the search area into a 2D array.
#### Starting the Search
- **Open List**: Squares to be considered for the path.
- **Closed List**: Squares that no longer need to be considered.
- **Process**: 
  1. Start from point A, add to the open list.
  2. Check adjacent squares, add walkable ones to the open list.
  3. Move the starting square to the closed list.
- **Path Scoring**: Determines the best path using the formula: **F = G + H**.
  - **G**: Cost from starting point A to a square.
  - **H**: Estimated cost from that square to the destination B (heuristic).
  - **F**: Total cost of a square (G + H).

#### Path Scoring Details
- **G Calculation**: 
  - Horizontal/Vertical Move: Cost = 10.
  - Diagonal Move: Cost = 14 (approximation of the square root of 2).
- **H Calculation (Manhattan Method)**: 
  - Calculate the total horizontal and vertical distance to target, ignoring obstacles.
  - Multiply the total by 10 (cost per square).
- **Choosing Squares**: Select the square with the lowest F score.

#### Pathfinding Example
1. **Starting Point**: Begin with the starting square, calculate F, G, H for adjacent squares.
2. **Expansion**: Move to squares with the lowest F score, update their G, H, F.
3. **Continuation**: Continue the process until the target square is reached.
4. **Path Tracing**: Trace back from the target square to the starting square using parent squares.

#### Additional Considerations
- **Heuristics**: Should be chosen carefully to ensure the shortest path.
- **Performance**: The accuracy of H affects the algorithm's speed.
- **Flexibility**: The method can be adapted to different grid layouts and cost definitions.
---
#### Sample Code Snippet (Pseudocode)

```pseudocode
function AStarPathfinding(start, target):
    openList = [start]
    closedList = []
    while openList is not empty:
        current = select square from openList with lowest F score
        if current is target:
            return constructPath(current)
        move current from openList to closedList
        foreach neighbor of current:
            if neighbor is not walkable or in closedList:
                skip to next neighbor
            if new path to neighbor is shorter or neighbor not in openList:
                set neighbor's parent to current
                update neighbor's G, H, F scores
                if neighbor not in openList:
                    add neighbor to openList
    return failure
```

### Search in A* Pathfinding
1. **Choose the Square with the Lowest F Score**:
   - From the open list, select the square with the lowest F cost. In case of a tie, you can choose any, but some implementations favor the latest squares added.

2. **Move the Chosen Square to the Closed List**:
   - Remove the chosen square from the open list and add it to the closed list. This indicates that this square has been fully processed.

3. **Examine Adjacent Squares**:
   - Inspect the squares adjacent to the chosen square. Ignore squares with obstacles (like walls) or squares already in the closed list.
   - For squares not in the open list, add them to the list, setting their parent to the current square and calculating their F, G, and H scores.
   - For squares already in the open list, check if the path to them from the current square is better (i.e., has a lower G score). If so, update their parent to the current square and recalculate their scores.

4. **Repeat the Process**:
   - Continue the above steps until the target square is added to the closed list, indicating that a path has been found, or until the open list is empty, meaning there's no path.

5. **Trace the Path**:
   - Once the destination square is on the closed list, the path can be traced back from the destination to the start by following the 'parent' pointers.

6. **Optimization Note**:
   - You might see variations where the search stops as soon as the destination is added to the open list. This can be faster but may not always yield the shortest path, especially when the cost to reach the final node varies significantly.

7. **A* Characteristics**:
   - A* pathfinding must include open and closed lists and use the F = G + H formula for scoring paths.
   - A* is recognized as one of the most efficient pathfinding algorithms, but other methods might be better under specific conditions.

### Key Points of A* Pathfinding
- **F, G, H Scores**: Crucial for determining the best path.
- **Open List**: Contains squares that are yet to be fully examined.
- **Closed List**: Contains squares that have been fully processed.
- **Parent Squares**: Used to trace the shortest path back to the start.
- **Efficiency**: A* is generally efficient but other algorithms may be more suited to particular scenarios.

### Visualization and Code Implementation
To implement A* pathfinding in a programming environment, one would typically use data structures like lists or priority queues for the open and closed lists, and a 2D array or a similar structure to represent the grid. Each grid square would store its F, G, and H scores, along with a reference to its parent square. The algorithm iteratively updates these lists and scores based on the steps outlined above, ultimately finding the shortest path or determining that no path exists. 

A* pathfinding's popularity in various applications, from game development to robotics, stems from its balance of efficiency and accuracy in finding optimal paths in a grid-based environment.

---
When implementing the A* Pathfinding algorithm, there are several additional considerations to keep in mind to enhance its effectiveness and efficiency. Here's a summary of important points to consider:

### 1. Collision Avoidance with Other Units
- **Static Units**: Treat the locations of static units as unwalkable in the pathfinding algorithm.
- **Moving Units**: For units adjacent to the pathfinding unit, penalize nodes along their paths to avoid collisions.
- **Collision Detection**: Implement collision detection to recalculate paths or wait for moving units to clear the path.
### 2. Variable Terrain Cost
- **Different Terrain Costs**: Incorporate varying movement costs for different terrains (e.g., swamps, hills) into the G cost calculation.
- **Influence Mapping**: Apply a points system to paths for AI purposes, penalizing areas with high risk or high enemy activity.
### 3. Handling Unexplored Areas
- Use a "knownWalkability" array for each player to record explored areas, treating unexplored areas as walkable until proven otherwise.
### 4. Smoother Paths
- Penalize nodes where there's a change of direction or refine the path post-calculation to avoid sharp turns for a visually smoother path.
### 5. Non-Square Search Areas
- Consider different shapes for the search area, like hexagons or irregular shapes, and adapt the adjacency and cost calculations accordingly.
### 6. Speed Optimization Tips
- **Limit Pathfinding Frequency**: Spread pathfinding calculations over several game cycles.
- **Map Size and Units**: Consider using a smaller map or fewer units.
- **Multi-Tier Pathfinding**: Use different search area sizes for different path lengths.
- **Precalculated Paths**: For longer paths, consider using hardwired paths.
- **Avoiding Redundant Searches**: Pre-process the map to identify inaccessible areas and avoid redundant pathfinding in those regions.
- **Dead-end Tagging**: Identify and tag dead-end nodes to avoid unnecessary path calculations.
### 7. Maintaining the Open List
- **Efficient Data Structures**: Use a binary heap or other efficient data structures for the open list to speed up finding the lowest F cost node.
- **Handling Data**: Prefer arrays over object-oriented data handling for faster performance, and efficiently reset data between pathfinding calls.
### 8. Dijkstra's Algorithm
- Use Dijkstra's algorithm, which is similar to A* but without a heuristic component, for scenarios where the destination is not predetermined.
### Further Reading and Resources
- **Amit's A* Pages**: Offers detailed explanations and advanced concepts in A* pathfinding.
- **Gamasutra's "Smart Moves: Intelligent Path-Finding"**: A comprehensive article on A* and its alternatives.
- **Dave Pottinger's "Terrain Analysis"**: Covers advanced concepts in AI and pathfinding.