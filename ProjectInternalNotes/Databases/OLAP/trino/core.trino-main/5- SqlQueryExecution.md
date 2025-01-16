Below is a logical explanation of what the `SqlQueryExecution` class (and its factory) does within the Trino query lifecycle. This explanation ties into the previous overview of memory and query management, but focuses now on how a query is transformed from a SQL statement into an executable plan, scheduled, and ultimately run on a Trino cluster.

---

### What is `SqlQueryExecution`?

`SqlQueryExecution` represents the full lifecycle of a single SQL query in Trino. It manages every step from parsing and analyzing the query, through planning and optimization, to scheduling tasks across the cluster and ultimately producing results. It ties together multiple subsystems (planner, scheduler, execution policies, dynamic filters, etc.) into a coherent, end-to-end process for executing a single query.

---

### Key Responsibilities

1. **Initial Query Setup**:  
   When a query arrives (already parsed and “prepared” by the `PreparedQuery` step), the `SqlQueryExecution`:
   - Creates a `QueryStateMachine` for tracking query progress, statistics, and state changes.
   - Registers itself with various Trino services (like dynamic filtering).

2. **Analysis and Planning**:  
   - **Analysis**: Uses the `AnalyzerFactory` and `Analyzer` to understand the query’s semantics—resolving table names, checking column types, verifying permissions, and determining what data is referenced.
   - **Optimization**: Passes the analyzed query through a series of `PlanOptimizer`s. These rewrite and improve the query plan for efficiency, apply cost-based decisions, and finalize a logical plan.
   - **Fragmentation**: Once optimized, the logical plan is broken into “fragments” suitable for distributed execution. Each fragment will run on one or more worker nodes.
   - **Adaptive Planning (optional)**: Adaptive plan optimizers can refine the plan dynamically based on runtime feedback or advanced heuristics.

   After planning, the class has a `Plan` and a `SubPlan` representing the distribution of work across the cluster.

3. **Dynamic Filtering Registration**:  
   If dynamic filtering is enabled, the query is registered with the `DynamicFilterService`. Dynamic filters can prune irrelevant data early, reducing I/O and improving performance.

4. **Scheduling and Execution Policy**:  
   Using the chosen execution policy (defined by session properties and injected at runtime), the `SqlQueryExecution`:
   - Initializes a `QueryScheduler` (either `PipelinedQueryScheduler` or `EventDrivenFaultTolerantQueryScheduler`, depending on retry policy).
   - The scheduler assigns fragments to nodes, creates tasks, and schedules splits (data chunks) to be processed in parallel across the cluster.

5. **Running the Query**:  
   After planning and scheduling are complete and the query transitions to the "starting" state:
   - The scheduler begins creating tasks on worker nodes.
   - Tasks fetch their assigned splits, process the data, and produce intermediate results.
   - Results are combined, possibly through intermediate stages and exchanges, leading to a final result set that is returned to the client.

6. **Tracking and State Management**:  
   The `SqlQueryExecution` maintains:
   - The `QueryState` and transitions (from PLANNING → STARTING → RUNNING → FINISHED or FAILED).
   - Statistics like total CPU time, memory usage, and the final query completion time.
   - Listeners can be registered to be notified when the query transitions between states or completes.

7. **Error Handling and Cancellation**:  
   If something goes wrong (e.g., a node fails, or a memory limit is exceeded), it can mark the query as failed. If the user or system decides to cancel the query, `cancelQuery()` changes its state and halts execution. If a single stage or task fails, `failTask(...)` or `cancelStage(...)` is invoked accordingly.

8. **Finalization**:  
   When the query finishes:
   - Dynamic filtering registration is removed.
   - Any table execution contexts are cleaned up.
   - The scheduler is cleared, and final `QueryInfo` is produced for logs and reporting.
   - If the query produced a result, it notifies consumers that results are ready or that results were consumed.

---

### The `SqlQueryExecutionFactory`

The `SqlQueryExecutionFactory`:
- Knows how to create an instance of `SqlQueryExecution` for a given query.
- Injects all the necessary dependencies (planners, optimizers, schedulers, dynamic filter service, node allocators, etc.) configured in the Trino server.
- Ensures consistency and configurability, letting different policies and components be plugged in without changing core logic.

---

### Putting It All Together

**Example of a Query’s Journey**:
1. A user issues a SQL query to the Trino coordinator.
2. `SqlQueryExecutionFactory` creates a `SqlQueryExecution` object, providing it with session info, prepared query, and all other services.
3. `SqlQueryExecution` analyzes and optimizes the query, producing a finalized execution plan.
4. The query is registered for dynamic filtering if enabled.
5. The query scheduler is created and starts to allocate tasks across nodes.
6. Tasks run on multiple worker nodes; data flows through splits and stages.
7. The query completes; final results are delivered to the client. Cleanup occurs.

---

### Key Takeaways

- `SqlQueryExecution` is the orchestrator for a single query’s life in Trino.
- It integrates the query planner, the scheduling engine, dynamic filtering, memory management, and execution policies into a single workflow.
- By the time a query reaches `SqlQueryExecution`, it is ready to be turned into a distributed plan and executed across many nodes, benefiting from all the optimizations and policies Trino has in place.

This class is central to Trino’s modular architecture: it doesn’t do all the work itself but coordinates various components (analyzers, optimizers, schedulers, and executors) into a seamless pipeline to execute user queries at scale and with efficiency.