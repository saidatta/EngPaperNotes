#### Overview of Greedy Algorithms
- **Definition**: Greedy algorithms make the locally optimal choice at each step, aiming for a globally optimal solution.
- **Context**: Optimal for many problems, simpler and more efficient than dynamic programming in certain cases.
- **Scope**: Applicable to optimization problems; often used when dynamic programming is excessive.
#### Key Principles
1. **Local Optimization**: Makes the best immediate choice without considering the global problem.
2. **Global Goal**: Attempts to find a globally optimal solution through a series of local decisions.
3. **Problem Suitability**: Not always optimal for all problems; effectiveness depends on the problem structure.
#### 15.1 An Activity-Selection Problem
- **Problem Definition**: Schedule a maximum number of activities requiring exclusive use of a common resource.
- **Input**: Set of activities \( S = \{a_1, a_2, \ldots, a_n\} \) with start times \( s_i \) and finish times \( f_i \).
- **Goal**: Select a maximum-size subset of non-overlapping activities.
- **Greedy Choice**: Select the activity with the earliest finish time.
- https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/solutions/93786/this-is-actually-activity-selection-problem/
#### Activity-Selection Algorithm
1. **Sort Activities**: Order activities by increasing order of finish time.
2. **Iterative Approach**:
   - Start with the first activity.
   - Select the next activity with the start time after the current activity's finish time.
3. **Algorithm Pseudocode**:
   ```python
   GREEDY-ACTIVITY-SELECTOR(s, f, n):
       A = {a1}
       k = 1
       for m = 2 to n:
           if s[m] >= f[k]:
               A = A ∪ {am}
               k = m
       return A
   ```
   - Where `s` and `f` are the start and finish times, and `n` is the number of activities.
#### Greedy vs Dynamic Programming
- Greedy algorithms are simpler and more efficient for certain problems compared to dynamic programming.
- Greedy solutions often involve making a choice and solving one subproblem, whereas dynamic programming solves many subproblems and combines their solutions.
#### Other Applications of Greedy Algorithms
- **Huffman Coding**: Efficient for data compression.
- **Minimum Spanning Trees**: Used in network design, such as in Chapter 21.
- **Dijkstra's Algorithm**: For shortest paths in graphs, as discussed in Section 22.3.
- **Set Covering**: Greedy heuristics applicable in certain scenarios (Section 35.3).
#### Complexity and Optimization
- Greedy algorithms often have more straightforward implementations and lower time complexity compared to other methods.
- Optimal for problems where the greedy choice leads to an optimal global solution.
#### Practical Applications and Code Examples
- **Network Scheduling**: Scheduling tasks or activities with shared resources.
- **Resource Allocation**: Efficiently allocating limited resources among competing tasks.
- **Pathfinding in Graphs**: Finding shortest paths and minimal spanning trees.
#### Conclusion
Greedy algorithms provide effective solutions for a range of optimization problems by making the most advantageous choice at each step. While they do not guarantee a globally optimal solution for all problems, they are well-suited for problems where local optimization aligns with the global goal. Understanding when and how to apply greedy algorithms is crucial for designing efficient algorithms in various applications.