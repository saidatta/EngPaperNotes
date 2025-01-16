Below is a logical explanation of what the `NodePartitioningManager` class does and how it fits into Trino’s query execution process. This class is crucial for mapping the conceptual partitioning of data into actual nodes and buckets that the query will use during execution.

---

### High-Level Purpose

In Trino, queries are often split into multiple tasks that run in parallel on different worker nodes. These tasks may be organized into “buckets” to ensure that data is properly distributed and parallelized. The `NodePartitioningManager` is responsible for:

1. Determining how many partitions (buckets) to use for a given partitioning scheme.
2. Mapping each bucket to a specific node in the cluster.
3. Providing a `PartitionFunction` or `BucketFunction` that decides how data rows or splits are assigned to buckets.
4. Handling both built-in system partitioning schemes (like hash distributions or single-node distributions) and connector-provided partitioning schemes.

---

### Key Responsibilities

1. **Obtaining Partition Functions**:
   For a given plan fragment, Trino must assign rows to partitions. The `NodePartitioningManager`:
   - Uses the `PartitioningHandle` from the plan to figure out which partitioning strategy to use.
   - If the partitioning is a built-in system type, it returns a specialized partition function.
   - Otherwise, it uses a connector-provided `BucketFunction` to partition data.

2. **Mapping Buckets to Nodes**:
   The manager obtains a `NodePartitionMap` that:
   - Lists the worker nodes assigned to each partition.
   - Provides an array mapping from each bucket to a partition index (and thus to a node).
   
   If the connector provides a `ConnectorBucketNodeMap`, it uses that mapping. If not, it creates a default mapping by evenly distributing buckets across available nodes.

3. **System vs. Connector Partitioning**:
   - **System Partitioning**: If the partitioning handle indicates a system partitioning (e.g., `FIXED_HASH_DISTRIBUTION`, `SINGLE`, `COORDINATOR_ONLY`), it uses internal logic to select nodes. For example, `FIXED` partitioning uses a fixed number of buckets and distributes them evenly across available nodes.
   - **Connector Partitioning**: If the partitioning handle points to a connector-specific implementation, it looks up the connector’s `ConnectorNodePartitioningProvider`. This provider can:
     - Return a `ConnectorBucketNodeMap` for fixed mappings.
     - Or indicate that no fixed mapping is available, in which case the `NodePartitioningManager` creates an arbitrary mapping.

4. **Handling Complex Partitioning Schemes**:
   For special partitioning handles like `MergePartitioningHandle`, the manager may recursively retrieve node mappings to ensure both sides of an insert/update operation use identical mappings. This recursion ensures consistent bucket-to-node assignments for complex operations.

5. **Default Bucket Counts and Fallbacks**:
   If the connector doesn't provide a bucket count, the manager chooses a default count based on session properties and the cluster environment. This ensures that the query has enough buckets to parallelize the data effectively but not so many that overhead becomes too large.

6. **Integrating with Node Scheduling**:
   The manager uses the `NodeScheduler` to get lists of nodes for arbitrary distributions. If the partitioning can't provide a fixed mapping, it shuffles and cycles through the available nodes to assign buckets in a balanced manner.

7. **Supporting Split-to-Bucket Assignments**:
   Finally, for reading data splits (pieces of data assigned to tasks), the manager provides a function (`getSplitToBucket`) that maps each `Split` to a bucket. This bucket index, combined with the bucket-to-node mapping, determines on which node the split will run.

---

### Example Flow

1. **Query Planning**: A plan fragment defines a partitioning handle indicating how data should be partitioned.
2. **Get Partition Function**: The execution engine calls `getPartitionFunction(...)` to get a `PartitionFunction` suitable for hashing rows.
3. **Get Node Partitioning Map**: If needed, the engine calls `getNodePartitioningMap(...)` to map buckets to nodes.
4. **Assigning Buckets and Splits**:
   - The returned map or bucket function is used at runtime to distribute data or assign splits.
   - If a connector provides a custom `BucketFunction` or `ConnectorBucketNodeMap`, these are used directly.
   - Otherwise, fallback logic distributes buckets or splits evenly across available workers.

---

### Key Takeaways

- **Flexibility**: The `NodePartitioningManager` provides a unified interface for both system and connector partitioning logic.
- **Scalability**: By handling large numbers of buckets and default mappings, it ensures queries can scale out across many nodes.
- **Adaptability**: It can handle a variety of partitioning strategies, from simple single-node distributions to complex hash or custom connector-defined distributions.
- **Integration**: It works closely with `NodeScheduler` and connector partitioning providers to produce the final bucket-to-node mapping and partition functions that Trino’s execution engine needs.

In summary, `NodePartitioningManager` is the central place where Trino decides how to map logical partitions (buckets) of data to actual cluster nodes and how to transform rows or splits into the right partition. This ensures that the distributed execution engine effectively leverages the cluster’s resources.