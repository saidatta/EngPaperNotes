### Context: What Are Resource Groups in Trino?

In a large, multi-tenant or multi-workload Trino environment, resource groups help control how queries share and compete for limited system resources (like CPU and memory). They can impose concurrency limits, memory quotas, and scheduling policies to ensure fairness, prevent overload, and maintain predictable performance. A resource group may contain subgroups, forming a tree. Queries are only directly submitted to leaf groups, and parent groups aggregate usage and apply constraints across their descendants.

---
### Role of `InternalResourceGroup`
`InternalResourceGroup` is an internal data structure that:
1. **Organizes Queries in a Hierarchy:**  
   Each `InternalResourceGroup` represents a node in a resource group tree. A node can be a leaf (accepting queries directly) or an intermediate node (aggregating limits and states of its children).
2. **Tracks Query States and Limits:**  
   It keeps track of how many queries are running and queued, how much CPU time and memory they consume, and whether the group (and its ancestors) are allowed to run or queue more queries.
3. **Enforces Resource Limits:**  
   Limits include:
   - **Memory limits (soft/hard)**: Prevents running too many or too large queries once usage surpasses thresholds.
   - **Concurrency limits (soft/hard)**: Controls how many queries can run simultaneously.
   - **CPU usage and quota generation**: The group can generate CPU quota over time and enforce soft/hard CPU consumption limits, dynamically adjusting concurrency allowances.
   - **Queue size limits**: Restricts how many queries can be waiting if resources are not currently available.
4. **Implements Scheduling Policies:**  
   The class supports various scheduling strategies:
   - **FAIR**: Simple first-in-first-out with equal fairness.
   - **WEIGHTED & WEIGHTED_FAIR**: Allocates resources based on assigned weights, ensuring some groups get proportionally more slots if configured.
   - **QUERY_PRIORITY**: Orders queries based on priority settings, allowing higher priority queries to start before lower priority ones.

   Depending on the chosen scheduling policy, the group uses different internal data structures (e.g., a FIFO queue or a priority queue) to decide which queries get to start next.

5. **Dynamic Eligibility & Promotion of Subgroups:**
   Leaf groups submit queries to this structure. Intermediate groups aggregate the state from their children. If a subgroup has capacity to run more queries, it signals upwards, possibly making the parent aware that it can run more queries. Conversely, if limits are reached, it stops "promoting" its children, preventing new queries from starting.

6. **Asynchronous Execution and Start-Up of Queries:**
   When a query arrives:
   - The group checks if it can run immediately.
   - If not, it goes into a queue.
   - When resources free up, the group "dispatches" a queued query to start running by calling the query’s `startWaitingForResources()` method in a background thread.

7. **Updating and Tracking Resource Usage:**
   Each query reports CPU and memory usage. The `InternalResourceGroup` periodically updates its internal snapshots of total usage. This helps it decide if the group has exceeded soft or hard limits and needs to reduce concurrency or reject new queries.

8. **Integration with the Rest of the System:**
   The `InternalResourceGroup`:
   - Integrates with `ResourceGroupManager` for hierarchical organization and selection.
   - Works closely with `ManagedQueryExecution` objects that represent running queries.
   - Communicates with JMX or other monitoring tools to report metrics.

---

### Conceptual Flow

**When a query comes in:**
1. The query is submitted to a leaf `InternalResourceGroup`.
2. The group checks if it has room to run this query immediately:
   - If yes, it starts the query.
   - If no, it places the query in a queue.
3. The resource group and its ancestors track how many queries are queued and running.
4. As queries finish or as CPU quota regenerates over time, the group may promote queued queries to running.

**When usage changes:**
- The group updates CPU/memory usage.
- If usage exceeds certain thresholds, it may reduce concurrency for future queries.
- If usage drops, it may allow more queries to start from the queue.

**When asked for state or info:**
- The group can produce `ResourceGroupInfo` snapshots showing how many queries are running, queued, memory usage, CPU usage, etc.
- It can also produce a hierarchical view showing the entire subtree’s utilization.

---

### Key Takeaways

- **Hierarchical Resource Management:** `InternalResourceGroup` nodes form a tree that enforces resource constraints from the top down, ensuring that each level’s rules apply to its descendants.
- **Dynamic Scheduling:** By maintaining priority queues and usage counters, the group decides dynamically which queries to run next, achieving fairness, priority, or weighting as configured.
- **Scalable Control:** This design helps Trino serve many concurrent queries efficiently while ensuring that no single user or workload hogs the entire cluster.

In essence, `InternalResourceGroup` is a building block that helps Trino’s coordinator fairly allocate resources among multiple queries and workloads, maintaining stable, predictable performance in a shared environment.