Below is a logical explanation of what this `NodeScheduler` class and its related methods do within the Trino codebase. It focuses on how `NodeScheduler` selects and assigns nodes (worker machines) to handle query splits, enforcing various constraints and optimizing for effective parallel execution.

---

### High-Level Purpose

`NodeScheduler` is responsible for deciding which worker nodes in the Trino cluster will run specific parts (splits) of a query. A "split" represents a piece of data and its associated processing task. The scheduler tries to place these splits on nodes in a way that balances load, respects resource limits, and takes advantage of locality (running data on nodes where it's most accessible).

---

### Key Responsibilities

1. **Node Selection & Filtering**:  
   The scheduler provides methods to:
   - Retrieve all nodes available for scheduling.
   - Filter out certain nodes if needed (e.g., exclude coordinators, remove blacklisted nodes).
   - Randomize node order to ensure fair distribution of work over time.

2. **Direct Node Assignments for Certain Splits**:  
   Some queries have strict mapping from splits to nodes (e.g., fault-tolerant or bucketed queries), and the scheduler must respect these mappings. If the bucket-to-node mapping exists, the scheduler places each split on its pre-assigned node.

3. **Limiting Splits per Node**:  
   The code ensures that a node doesn't get overloaded. There are constraints such as:
   - `maxSplitsWeightPerNode`: a cap on how much "split weight" a node can accept.
   - `minPendingSplitsWeightPerTask`: ensures that a node doesn't become a bottleneck by queuing too many splits waiting for free execution slots.

4. **Handling Blocked Nodes**:  
   If a node can't accept more splits at the moment (due to hitting the configured limits), the assignment is deferred. The scheduler keeps track of these blocked nodes and returns a future (`ListenableFuture`) that completes when the node frees enough resources (like when a running task finishes or acknowledges the splits).

5. **Integration with Existing Tasks**:  
   When adding new splits to already running tasks (`RemoteTask` instances), the scheduler checks if there's room in these tasks' split queues. If not, it creates a future that will be completed once the node's split queue has space again.

6. **Creating Scheduling Results (SplitPlacementResult)**:  
   After trying to assign all splits, the scheduler bundles the results:
   - Which splits ended up assigned to which nodes?
   - A `blocked` future that indicates if we must wait for nodes to free space before assigning more splits.

   Clients of the scheduler use this result to either proceed with the assigned splits or wait for the returned future if some splits could not be scheduled immediately.

---

### Important Methods Explained

- **`createNodeSelector`**: Given a session and an optional catalog, this method returns a `NodeSelector`, which encapsulates logic for choosing nodes for the splits related to that particular catalog and session.

- **`selectNodes(int limit, Iterator<InternalNode> candidates)`**: Chooses up to `limit` nodes from an iterator of candidates. Useful when you only need a fixed number of nodes.

- **`randomizedNodes(...)` / `filterNodes(...)`**: Provides subsets of nodes after filtering out coordinators or excluded nodes and randomizing their order for fair load distribution.

- **`selectExactNodes(...)`**: Tries to find nodes that match given host constraints (like a host address specified by a split). If no such exact nodes are found but the host is the coordinator, it may return the coordinator node to ensure progress.

- **`selectDistributionNodes(...)`**: The core logic for assigning splits to nodes in a bucketed or fault-tolerant execution scenario:
  1. Iterates over each split.
  2. Finds the node assigned by the bucket map.
  3. Checks if that node can accept the split (based on pending splits count, weights, etc.).
  4. Collects assignments in a `Multimap` from nodes to splits.
  5. If a node is full, that node is marked as blocked.
  6. Returns a `SplitPlacementResult` containing the assignments and a future that will be complete when any blocked node frees up space.

- **`canAssignSplitToDistributionNode(...)` and `canAssignSplitBasedOnWeight(...)`**: Check whether a node can still accept a split according to the resource limitations. This ensures that even if splits are large, at least one can run if the node is currently empty, allowing forward progress.

- **`toWhenHasSplitQueueSpaceFuture(...)`**: Builds a future that completes when one of the blocked nodes (or tasks) signals that it can take more splits. The scheduler or callers use this to pause adding more splits until capacity is available.

---

### Takeaways

- **Balancing Workload**: By applying constraints and randomization, the scheduler tries to prevent hotspots where a single node or task gets overloaded.
- **Asynchronous & Future-based**: The code doesn't block waiting for nodes to become free; it returns futures that complete when conditions change, enabling a reactive approach to scheduling.
- **Integration with Node and Task State**: This code interacts with `RemoteTask` and `NodeMap` to understand current cluster load, pending splits, and node availability.

---

In essence, `NodeScheduler` forms a crucial part of Trino’s distributed execution framework. It ensures splits are assigned intelligently and fairly, respects resource constraints, and provides hooks (via futures) to handle scenarios where immediate scheduling isn’t possible, thus maintaining efficiency and robustness in the scheduling layer.