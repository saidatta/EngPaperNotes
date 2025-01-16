Below is a logical explanation of what the `ClusterMemoryManager` class does within Trino’s memory management and query execution ecosystem:

### High-Level Overview

The `ClusterMemoryManager` is a coordinator-side component in Trino that monitors and manages the cluster’s memory usage. It helps ensure that queries do not exceed configured memory limits and, when necessary, terminates queries or tasks to prevent the cluster from running out of memory.

### Key Responsibilities

1. **Aggregating Memory Usage Information Across the Cluster**:  
   The `ClusterMemoryManager` collects memory usage statistics from all active nodes in the cluster. Each node provides a snapshot of its memory pools and running queries’ memory usage.

2. **Enforcing Global Memory Limits for Queries**:
   Trino allows configuring maximum memory limits at a global cluster level. This class monitors each query’s total and user memory consumption to ensure no query surpasses these global limits:
   - **User Memory Limit**: How much user memory a single query can use.
   - **Total (User + System) Memory Limit**: The total amount of memory (including overhead and system memory) that a query can consume.

   If a query exceeds these limits, `ClusterMemoryManager` will force it to fail, helping to maintain cluster stability.

3. **Detecting Out-of-Memory Conditions**:
   When the cluster is under memory pressure (e.g., one or more nodes are running at their memory capacity), the manager decides if the cluster is effectively “out of memory.” If so, it identifies which queries are safe to kill in order to free memory resources.

4. **Low Memory Killers**:
   The `ClusterMemoryManager` integrates with "LowMemoryKillers." These are strategies (injected services) that determine which queries or tasks should be terminated first when memory is scarce. The logic is:
   - First, try to kill tasks (more granular approach).
   - If that doesn’t help, kill entire queries.
   
   By plugging in different `LowMemoryKiller` implementations, the cluster can adopt different policies (such as killing the largest queries or those running the longest).

5. **Resource Overcommit Handling**:
   Some queries may have a special flag (`RESOURCE_OVERCOMMIT`) that allows them to bypass normal memory limits. `ClusterMemoryManager` still monitors these, and will only kill them if the cluster is genuinely out of memory. This provides a prioritized mechanism where certain queries run at higher priority but still must yield to system stability.

6. **Logging and Diagnostics**:
   If the manager decides to kill queries or tasks due to memory pressure, it logs diagnostic information about memory usage on each node. This helps administrators understand why certain queries were terminated and potentially adjust configuration or cluster resources.

7. **Integration with the MemoryPool**:
   The cluster maintains a `ClusterMemoryPool` that aggregates memory usage from all nodes. The `ClusterMemoryManager` updates this pool’s view periodically. Through listeners, external components (like UI dashboards or management tools) can be notified when memory conditions change.

8. **Memory Leak Detection**:
   The manager also works with a `ClusterMemoryLeakDetector` to identify if memory has leaked (queries that disappeared but memory reservations still linger). If detected, it helps keep the cluster stable despite potential bugs or issues.

9. **Monitoring and Reporting**:
   The manager exposes metrics such as:
   - Total user memory reservation
   - Total memory reservation (user + system)
   - How many queries and tasks have been killed due to OOM (Out-Of-Memory)
   - Number of queries suspected to have memory leaks

   These metrics are useful for cluster administrators to monitor health and performance.

### Process Flow Example

1. **Continuous Updates**:  
   Periodically, `ClusterMemoryManager` fetches fresh memory info from each node.

2. **Check Running Queries**:  
   It checks each running query’s memory usage against global and session-level limits.

3. **Out-Of-Memory Check**:  
   If any node signals memory pressure (e.g., running out of memory space), the manager considers the cluster OOM scenario.

4. **Choosing Victims to Free Memory**:  
   If needed, it consults the low memory killer policies to pick queries or tasks to kill, freeing memory and preventing a full cluster meltdown.

5. **Updating the Memory Pool and Notifying Listeners**:  
   After adjustments, it updates the `ClusterMemoryPool` and notifies any registered listeners, allowing real-time dashboards and monitoring systems to stay informed.

### Key Takeaways

- The `ClusterMemoryManager` doesn’t just look at one node; it aggregates information from all cluster nodes.
- It strictly enforces global memory limits to ensure no single query takes down the entire cluster.
- It provides a structured, configurable approach to handle OOM conditions, improving cluster resilience and stability.
- It acts as a safety valve, stepping in when memory is critically low to terminate offending queries or tasks.

In essence, the `ClusterMemoryManager` ensures that Trino’s distributed SQL engine remains stable and fair in how it allocates memory across all running queries.