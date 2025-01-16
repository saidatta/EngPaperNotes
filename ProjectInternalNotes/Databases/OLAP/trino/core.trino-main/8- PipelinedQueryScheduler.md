Below is a logical explanation of what the `PipelinedQueryScheduler` class and its related internal classes do, focusing on how it orchestrates a query’s execution from multiple stages (some on the coordinator, some distributed across worker nodes) and handles error recovery and retries.

---

### High-Level Purpose

`PipelinedQueryScheduler` is a core part of Trino’s distributed query engine. It manages the scheduling of query stages that produce their output “pipelined”—i.e., as soon as data is processed, it’s streamed to subsequent stages without waiting for the entire stage to complete. This leads to more efficient and lower-latency execution. Additionally, this class handles complexities like:

- Coordination between stages that run on the coordinator node and those that run on worker nodes.
- Dynamic scheduling of data splits and tasks across nodes.
- Handling failures and, depending on the configured retry policy, retrying failed stages to increase query resilience.

In essence, `PipelinedQueryScheduler` sits at the heart of the scheduling process, ensuring that all parts of a query plan are executed in the right order, on the right nodes, and that the system can adapt to failures.

---

### Key Responsibilities

1. **Orchestrating Multiple Stages**:  
   The query plan is split into multiple stages. Some stages run on the coordinator node (for example, final aggregations or commit operations), and others run on worker nodes. This scheduler:
   - Sets up pipelines between these stages.
   - Ensures that once upstream (producing) stages have data ready, it can be consumed by downstream (consuming) stages.

2. **Coordinator Stages vs Distributed Stages**:
   - **Coordinator stages**: Must run on the coordinator node. They are not fault-tolerant and cannot be restarted if they fail. The code sets up a separate entity, `CoordinatorStagesScheduler`, to handle these stages. They create tasks on the coordinator itself.
   - **Distributed stages**: These are run across the cluster’s worker nodes. They are managed by the `DistributedStagesScheduler`. This part of the code can handle failures by reassigning tasks to different nodes or retrying entire query attempts if enabled.

3. **Retry Policy & Failure Recovery**:
   The `PipelinedQueryScheduler` supports different retry policies:
   - **NONE**: No retries—if a stage fails, the query fails.
   - **QUERY**: If a distributed stage fails, the entire distributed section of the query can be retried (starting a new “attempt”). This involves re-creating tasks and reassessing where to run them.
   
   The scheduler:
   - Monitors stages for failures.
   - Depending on the policy and error type, decides whether to fail the query or attempt a retry.
   - Implements a delay and exponential backoff for retries to avoid immediate repeated failures.

4. **Dynamic Adaptation During Execution**:
   The scheduler continuously monitors the state of stages:
   - Once a stage finishes, it notifies downstream stages so they can clean up or start finalizing.
   - If a stage is blocked (e.g., waiting for data or resources), the scheduler may wait for a future that completes when conditions change, then re-run the scheduling loop.
   
   The scheduling process is iterative and continues until all stages are finished or the query fails.

5. **Output Buffers & Data Flow**:
   The code sets up output buffers for each stage to ensure that data produced by a stage is directed to the correct downstream consumers. For coordinator-consumed stages, it sets up a single output partition. For distributed stages, it sets up partitions according to the plan’s partitioning handle.

6. **Integration with Other Components**:
   - **`ExecutionPolicy`**: Decides the order and manner in which stages are scheduled.
   - **`DynamicFilterService`**: Delays or prunes splits if certain dynamic filters are not yet satisfied, improving performance by skipping irrelevant data.
   - **`NodePartitioningManager` and `NodeScheduler`**: Assign nodes to tasks and obtain node-to-partition mappings for splits.
   - **`StageManager` and `StageExecution`**: Track and manage lifecycle of individual stages.
   - **`FailureDetector`**: Helps detect failed nodes or tasks so that the scheduler can respond appropriately.

7. **Finalization & Cleanup**:
   Once all stages have completed successfully, the scheduler transitions the query into a finishing state. If any stage fails and cannot be retried, it aborts all running tasks and transitions the query to a failed state. It also closes all `StageScheduler` instances to clean up resources.

---

### Important Internal Classes

- **`CoordinatorStagesScheduler`**: Manages stages that run on the coordinator. These are simpler but crucial stages (like final aggregations or commit tasks). They cannot be retried, so if they fail, the query fails.
- **`DistributedStagesScheduler`**: Manages the execution of stages across worker nodes. It supports retry logic (if configured) and can cancel, abort, or retry stages as needed.
- **`DistributedStagesSchedulerStateMachine`**: Tracks the state (PLANNED, RUNNING, FINISHED, etc.) of the distributed scheduling component. Allows transitions based on events like failures or completion.
- **`StageExecution`**: Represents the execution of a single stage. It holds tasks, tracks their states, and notifies listeners on state changes.
- **`StageScheduler`**: Responsible for actually assigning splits to tasks for a stage. Different implementations handle different partitioning schemes.

---

### Example Flow

1. **Initialization**:
   `PipelinedQueryScheduler` sets up the `StageManager`, creates `CoordinatorStagesScheduler` for coordinator-only stages, and `DistributedStagesScheduler` for worker stages.
   
2. **Starting Execution**:
   - The coordinator stages (if any) start first, possibly creating a single task on the coordinator.
   - The distributed stages scheduler is kicked off in parallel. It schedules tasks across worker nodes, taking input splits from various sources.

3. **Adaptive Scheduling**:
   If a stage gets stuck due to lack of resources or ongoing dynamic filters, the scheduler waits. Once conditions change (some tasks finish, freeing resources, or dynamic filters arrive), it attempts scheduling again.

4. **Failure Handling**:
   If a task or stage fails:
   - If retries are enabled and the error is considered retryable, the scheduler tears down the distributed stages and restarts them.  
   - If no retry is possible, the query fails and all remaining work is aborted.

5. **Completion**:
   Once all stages finish, the scheduler reports success to `queryStateMachine`, and any temporary resources are cleaned up.

---

### Key Takeaways

- `PipelinedQueryScheduler` is a complex orchestrator ensuring that distributed queries run efficiently and robustly.
- It manages a network of stages, deals with dynamic conditions (like node availability and runtime filtering), and can recover from certain failures.
- By separating coordinator-only scheduling from distributed stages scheduling, it can handle different classes of tasks differently, improving reliability and clarity.

In summary, `PipelinedQueryScheduler` is the powerhouse that turns a plan of interconnected stages into a running query that adapts dynamically to conditions in the cluster, tries to recover from failures, and ensures that results are eventually produced.