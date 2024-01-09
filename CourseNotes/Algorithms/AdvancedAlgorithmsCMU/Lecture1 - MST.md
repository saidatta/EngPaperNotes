x# Advanced Algorithms: Lecture 1 - Minimum Spanning Trees (MST)

## Overview
This lecture covers the fundamental concepts and classic algorithms related to Minimum Spanning Trees (MSTs), a key concept in graph theory and algorithm design. The focus is on three classic algorithms for computing MSTs: Borůvka's, Prim's (originally called Jarník's), and Kruskal's algorithms. The lecture also delves into the history, development, and optimization of these algorithms.

## Minimum Spanning Trees (MST)
- **Definition:** An MST of an undirected graph `G` is a tree that connects all vertices together with the minimum possible total edge weight.
- **Properties:**
  - Graph `G` is undirected.
  - Graph `G` has `V` vertices and `E` edges.
  - Edge weights are non-negative.
  - Assumes a simple graph (no parallel edges or self-loops).
  - Edge weights are distinct (for simplicity in this context). 
	  - **note:** If weights are distinct, then unique MST.
### Key Algorithms
1. **Borůvka's Algorithm (1926):** 
   - Developed almost 100 years ago.
   - Initially presented in a complex manner but fundamentally simple.

2. **Prim's Algorithm (1930), originally developed by Jarník:**
   - A simplification of Borůvka's algorithm.
   - Known for its simplicity and efficiency.

3. **Kruskal's Algorithm (1956):**
   - Another classic approach to finding MSTs.
   - Known for its straightforward implementation and usage in many textbook examples.

### Performance and Running Time
- **General Running Time:** \( O(M \log N) \) for all three algorithms, where `M` is the number of edges and `N` is the number of vertices.
- **Historical Optimization:** 
  - In the 1970s, efforts were made to improve the running time.
  - Algorithms achieving \( O(M \log^* N) \) were developed. log^** - is number of times you apply log.
  - `log^* N` is a very slowly growing function, indicating the number of times logarithm needs to be applied to `N` before the result is less than or equal to 1.
### Advanced Concepts
- **Linear Time Randomized Algorithm (1990s):**
  - A significant development by David Karger, Klein, and Tarjan.
  - This algorithm uses randomization to achieve a linear expected running time.
  - Open problem: A deterministic linear-time algorithm for MST is yet to be found.
## Technical Details
### Cut and Cycle Rules
- **Cut Rule (Blue Rule):**
  - Given any partition of a graph's vertices into two non-empty subsets, the lightest edge crossing the partition is part of the MST.
- **Cycle Rule (Red Rule):**
  - For any cycle in the graph, the heaviest edge in the cycle cannot be part of the MST.
### Code Example (Kruskal's Algorithm in Python)
```python
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def KruskalMST(graph, V):  # V is the number of vertices
    result = []  # Store the resultant MST
    i, e = 0, 0  # An index variable, used for sorted edges
    graph = sorted(graph, key=lambda item: item[2])
    parent, rank = [], []

    for node in range(V):
        parent.append(node)
        rank.append(0)

    while e < V - 1:
        u, v, w = graph[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e += 1
            result.append([u, v, w])
            union(parent, rank, x, y)

    return result
```

## Summary
This lecture provides a comprehensive understanding of MSTs and their algorithms. It emphasizes the historical development of these algorithms and the ongoing quest to optimize their running time. The lecture also underscores the importance of understanding fundamental concepts like the cut and cycle rules, which form the basis of all MST algorithms.

---
### Proofs of MST Principles

#### Cut Rule
- **Concept:** The lightest edge crossing any cut of the graph is part of the MST.
- **Proof:**
  - Assume the lightest edge is not in the MST.
  - Adding this edge to the MST creates a cycle.
  - The cycle must cross the cut again. Replace the heavier edge in this cycle (across the cut) with the lighter edge.
  - The resultant tree has a lower total weight, contradicting the assumption that the original tree was an MST.

#### Cycle Rule
- **Concept:** The heaviest edge in any cycle of the graph is not part of the MST.
- **Proof:**
  - Assume the heaviest edge in a cycle is in the MST.
  - Removing this edge splits the MST into two parts.
  - The remaining cycle connects these two parts and must contain an edge lighter than the removed edge.
  - Adding this lighter edge forms a tree with a lower total weight, contradicting the assumption of the original being an MST.

### MST Algorithms

#### Borůvka's Algorithm
- **Process:**
  - Start with each vertex as an isolated component.
  - At each stage, for every component, choose the cheapest edge leading out and color it blue.
  - Repeat until all components are connected.
- **Time Complexity:**
  - Halves the number of components in each round, leading to \( O(\log N) \) rounds.
  - Each round takes \( O(M) \) time, leading to \( O(M \log N) \) overall complexity.

#### Kruskal's Algorithm
- **Process:**
  - Sort all edges by weight.
  - Iterate over edges and add an edge to the MST if it connects two distinct components.
- **Time Complexity:**
  - Sorting edges takes \( O(M \log N) \).
  - Union-find data structure used for maintaining components and checking if an edge connects two distinct components.
  - Overall complexity approximately \( O(M \log N) \).
### Technical Aspects

#### Union-Find Data Structure
- **Operations:**
  - `MakeSet(x)`: Creates a set with a single element `x`.
  - `Find(x)`: Returns the name of the set containing `x`.
  - `Union(x, y)`: Merges the sets containing `x` and `y`.
- **Runtime:**
  - For `M` operations on `N` elements, total runtime is at most \( O(M \log^* N) \) using efficient union-find structures.
  - The inverse Ackermann function `α(M)` further optimizes this, leading to near-constant time for practical purposes.

#### Ackermann Function
- **Description:**
  - A rapidly growing function used to illustrate the complexity of certain algorithms.
  - The inverse Ackermann function is very slow-growing, useful in analyzing the efficiency of some algorithms, like union-find.

### Summary
This lecture provides a deep understanding of the principles behind MSTs, proving the fundamental cut and cycle rules. It details Borůvka's and Kruskal's algorithms, emphasizing their efficiency and implementation. The lecture also introduces important data structures like union-find, crucial for understanding these algorithms' complexity.

---
## Topic: Minimum Spanning Trees (MST) - Prim's Algorithm and Advanced Concepts

### Prim's Algorithm
- **Basic Process:**
  - Starts with a single vertex (root).
  - Repeatedly adds the cheapest edge from the current component to the MST, following the cut rule.
- **Implementation Details:**
  - Utilizes a priority queue data structure (like a heap) for efficient edge selection.
  - Operations on the priority queue:
    - `Insert(element, weight)`: Inserts an element with a given weight.
    - `DeleteMin()`: Removes the element with the smallest weight.
    - `DecreaseKey(element, newWeight)`: Decreases the weight of an element if the new weight is smaller.
  - Algorithm maintains a heap for each vertex, tracking the cheapest edge extending from the current MST component.
  - Complexity Analysis:
    - Time complexity is \( O(M \log N) \) where `M` is the number of edges and `N` is the number of vertices.
    - Involves \( M \) decrease key operations and \( N \) delete min operations.

### Advanced Concepts in MST Algorithms

#### Use of Fibonacci Heaps
- **Improvements Offered:**
  - Makes `Insert` and `DecreaseKey` operations constant amortized time.
  - `DeleteMin` remains \( O(\log N) \) amortized time.
  - Leads to a slight improvement in Prim's algorithm complexity.

#### Redman and Tarjan's Approach
- **Concept:**
  - Improves Prim's algorithm by dynamically adjusting the size of the heap (parameter `K`).
  - The algorithm restarts from a new root when the neighborhood of the current component exceeds a size threshold (`K`), or it connects to an existing component.
- **Complexity Analysis:**
  - Balances the number of decrease key operations and delete min operations.
  - Leads to a more efficient implementation, especially for sparse graphs.

### Final Discussion
- **Industry Usage of Algorithms:**
  - The actual implementation in the industry could vary based on characteristics like parallelism and overhead considerations.
  - Fibonacci heaps are less preferred due to complexities in practical implementation.

#### Cardiff, Klein, and Tarjan's Randomized Algorithm
- **Overview:**
  - Utilizes randomization for more efficient MST computation.
  - The approach involves sampling a subset of edges, building an MST on this subset, and then coloring heavy edges that form cycles with the sampled MST.
  - The algorithm recursively processes the remaining graph.
- **Expected Runtime:**
  - This approach leads to a linear expected runtime for MST computation.

### Key Takeaways
- The lecture detailed Prim's algorithm, including its implementation using priority queues and Fibonacci heaps.
- It explored advanced concepts and optimizations for MST algorithms, particularly focusing on balancing operations in Prim's algorithm and introducing a randomized approach for improved efficiency.
- The discussion also highlighted practical aspects of algorithm selection and implementation in the industry, emphasizing the importance of balancing theoretical efficiency with practical considerations.