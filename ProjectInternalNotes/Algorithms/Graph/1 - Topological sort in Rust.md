Below is a **step-by-step** explanation of the Rust code that implements **topological sorting** using **Kahn’s algorithm**. We’ll walk through the data structures, the algorithm’s logic, and the tests verifying its correctness. This explanation is aimed at a level of depth appropriate for a PhD-level software engineer.

---
## 1. What is Topological Sorting?
A **topological sort** of a directed acyclic graph (DAG) is an ordering of its nodes such that for every directed edge \((u \to v)\), \(u\) appears **before** \(v\) in the ordering. Kahn’s algorithm is one standard method to achieve this ordering or to detect if a cycle exists (in which case no valid topological ordering exists).

---
## 2. Function Signature

```rust
pub fn topological_sort<Node: Hash + Eq + Copy>(
    edges: &Vec<(Node, Node)>,
) -> TopologicalSortResult<Node>
```

- **Generic Parameter**:  
  - `Node`: any type that implements `Hash` + `Eq` + `Copy`.  
  - `Hash` and `Eq` are required because we store `Node` in a `HashMap` and need to compare them.  
  - `Copy` indicates nodes can be trivially copied (they have no complex ownership semantics).
- **Parameters**:  
  - `edges: &Vec<(Node, Node)>`: A list of directed edges in the graph, each represented as `(source, destination)`.
- **Return Type**:  
  - `Result<Vec<Node>, TopoligicalSortError>` which is an alias `TopologicalSortResult<Node>`.  
  - If the graph is a DAG, returns `Ok(sorted_nodes)`.  
  - If any cycle is detected, returns `Err(TopoligicalSortError::CycleDetected)`.

---

## 3. Data Structures for Kahn’s Algorithm

### 3.1 `edges_by_source`

```rust
let mut edges_by_source: HashMap<Node, Vec<Node>> = HashMap::default();
```

- A mapping from **a node** → **a list of its direct successors**.  
- This represents the graph in an adjacency-list form: for each node, we can retrieve all its “children” or “neighbors” (nodes it has edges to).

### 3.2 `incoming_edges_count`

```rust
let mut incoming_edges_count: HashMap<Node, usize> = HashMap::default();
```

- A mapping from **a node** → **the count of incoming edges**.  
- The number of edges that “point into” each node.  
- If a node has `0` incoming edges, it can appear next in a topological ordering (i.e., it has no unfulfilled dependencies).

By combining these two structures:
- `edges_by_source[node]` gives the neighbors of `node`.
- `incoming_edges_count[node]` tells us how many edges still point to `node`.

---

## 4. Building the Graph Structures

```rust
for (source, destination) in edges {
    incoming_edges_count.entry(*source).or_insert(0);
    edges_by_source
        .entry(*source)
        .or_default()
        .push(*destination);

    *incoming_edges_count.entry(*destination).or_insert(0) += 1;
}
```

- **Step 1**: Ensure the `source` node appears in `incoming_edges_count` with an initial `0` if it was not already there.
- **Step 2**: Add the `destination` node to the adjacency list for `source`.
- **Step 3**: Increment the `incoming_edges_count` for `destination` by 1, because each edge \((source \to destination)\) adds one incoming edge to `destination`.

Effectively:
- We handle the adjacency list by populating `edges_by_source`.
- We handle the in-degree count by incrementing `incoming_edges_count[destination]`.

---

## 5. Initializing the Queue of Zero-Incoming-Edge Nodes

```rust
let mut no_incoming_edges_q = VecDeque::default();

for (node, count) in &incoming_edges_count {
    if *count == 0 {
        no_incoming_edges_q.push_back(*node);
    }
}
```

- We create a `VecDeque<Node>` to store the nodes with **no** current dependencies (i.e., `count == 0`).  
- These can be processed first in the topological order.

---

## 6. The Main Loop of Kahn’s Algorithm

```rust
let mut sorted = Vec::default();

while let Some(no_incoming_edges) = no_incoming_edges_q.pop_back() {
    sorted.push(no_incoming_edges);
    incoming_edges_count.remove(&no_incoming_edges);
    // for each dependent neighbor, reduce its incoming edge count
    for neighbour in edges_by_source.get(&no_incoming_edges).unwrap_or(&vec![]) {
        if let Some(count) = incoming_edges_count.get_mut(neighbour) {
            *count -= 1;
            if *count == 0 {
                incoming_edges_count.remove(neighbour);
                no_incoming_edges_q.push_front(*neighbour);
            }
        }
    }
}
```

We repeatedly extract a node with zero incoming edges and **append it to the result**:

1. **Pop from `no_incoming_edges_q`**  
   - `pop_back()` removes a node from the end of the queue. (The direction—back vs. front—is an implementation detail; it could be `pop_front` as well.)

2. **Add to `sorted`**  
   - Since it has no dependencies, it can safely appear **next** in the topological order.

3. **Remove it from `incoming_edges_count`**  
   - We no longer track its in-degree, because we’re about to remove it from the graph.

4. **Decrement in-degree of its neighbors**  
   - For every node `neighbour` that depends on `no_incoming_edges` (i.e., has an edge \((no_incoming_edges \to neighbour)\)), we do:
     ```rust
     *count -= 1;
     if *count == 0 { ... }
     ```
   - If `neighbour` now has zero incoming edges, we remove it from `incoming_edges_count` and **push it onto** `no_incoming_edges_q`.  
   - Thus, once a node’s in-degree is reduced to zero, it becomes a candidate to be placed in the topological order.

### 6.1 Why Remove the Node?

```rust
incoming_edges_count.remove(&no_incoming_edges);
```

- Removing the node from `incoming_edges_count` is a way of marking that it’s “fully processed” and no longer in the graph.  
- If, in the end, any nodes remain in `incoming_edges_count`, it means they were never reduced to zero incoming edges—a sign of a cycle.

---

## 7. Detecting Cycles

```rust
if incoming_edges_count.is_empty() {
    Ok(sorted)
} else {
    Err(TopoligicalSortError::CycleDetected)
}
```

- After the `while` loop finishes, if **all** nodes were processed (i.e., `incoming_edges_count` is empty), we succeeded in generating a valid topological order.  
- If `incoming_edges_count` is **not** empty, it indicates there are nodes whose in-degree was never reduced to zero—implying at least one cycle in the graph. We return an error variant `TopoligicalSortError::CycleDetected`.

---

## 8. Testing the Code

The test suite includes various scenarios:

1. **`it_works()`**  
   - A straightforward DAG: `(1->2), (1->3), (2->3), (3->4), (4->5), (5->6), (6->7)`  
   - Asserts a successful topological sort is returned, and that the ordering is correct.  
   - Compares the result to `vec![1, 2, 3, 4, 5, 6, 7]`.

2. **`test_wikipedia_example()`**  
   - Checks correctness on a known example from Wikipedia (common edges: `(5->11), (7->11), etc.`).  
   - Confirms no cycles are detected and the ordering is valid.

3. **`test_cyclic_graph()`**  
   - A graph with a cycle: `1->2->3->4->5` and also `4->2`.  
   - This introduces a cycle `(2->3->4->2)`.  
   - The algorithm should detect that some nodes never reach in-degree 0 and return `Err(TopoligicalSortError::CycleDetected)`.

### 8.1 The `is_valid_sort` Helper

```rust
fn is_valid_sort<Node: Eq>(sorted: &[Node], graph: &[(Node, Node)]) -> bool {
    for (source, dest) in graph {
        let source_pos = sorted.iter().position(|node| node == source);
        let dest_pos = sorted.iter().position(|node| node == dest);
        match (source_pos, dest_pos) {
            (Some(src), Some(dst)) if src < dst => {}
            _ => {
                return false;
            }
        };
    }
    true
}
```

- **`is_valid_sort`** checks whether, for every edge \((source \to dest)\), the index of `source` in `sorted` is **less** than the index of `dest`. This ensures a valid topological order.

---

## 9. Complexity & Summary

- **Time Complexity**: \(O(V + E)\) for building the adjacency list and running Kahn’s algorithm, where \(V\) is the number of unique nodes and \(E\) is the number of edges.
- **Space Complexity**: \(O(V + E)\) for the adjacency list and for storing in-degree counts and the result.

### Key Takeaways

- We use **Kahn’s algorithm**:
  1. Count in-degrees.
  2. Maintain a queue of nodes with in-degree \(0\).
  3. Repeatedly dequeue a node, add it to the result, and decrement in-degree of its neighbors.
  4. If any nodes remain with nonzero in-degree, a cycle exists.
- The code systematically builds adjacency lists, in-degree maps, and processes nodes in a loop.  
- The test coverage checks basic acyclic graphs, a known example, and a cycle detection scenario.  
- Returning `Err(TopoligicalSortError::CycleDetected)` upon leftover nodes is the mechanism to detect cycles.

Overall, this is a concise implementation of **topological sorting** in Rust, demonstrating both **successful DAG ordering** and **cycle detection** in a directed graph.