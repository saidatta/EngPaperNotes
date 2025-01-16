Below is a high-level, logical explanation of what the `DispatchManager` does within Trino’s internal architecture.

### What is the DispatchManager?

The `DispatchManager` is part of Trino’s coordinator logic. When a user issues a SQL query, Trino doesn't immediately start executing it. Instead, the query goes through a "dispatch" phase. The `DispatchManager` orchestrates this phase by preparing the query, assigning it to a resource group (for workload management), and eventually handing it off for execution. It is a key piece that connects the request from the client to the actual execution engine, ensuring proper initialization, validation, and queueing.

### Core Responsibilities

1. **Query Creation and Registration**:  
   When a new query arrives:
   - **Generate a QueryId**: Each query gets a unique identifier.
   - **Create a Session**: The `DispatchManager` uses the provided `SessionContext` to create a `Session` object, applying user and system defaults, and ensuring user permissions are respected.
   - **Prepare the SQL Statement**: It parses and plans the query text into a `PreparedQuery`.
   - **Check Permissions**: It uses `AccessControl` to ensure the user is allowed to run the query.
   - **Resource Group Assignment**: It finds the appropriate resource group for the query via the `ResourceGroupManager`, which determines concurrency and resource policies.

   If anything fails at this stage (e.g., permission denied, query too long), the `DispatchManager` registers a "failed dispatch query" so that the failure is tracked and reported consistently.

   **In short**: It safely transforms a raw SQL request into a well-defined, trackable query entity.

2. **Query Tracking**:  
   The `DispatchManager` keeps a `QueryTracker` to maintain a list of all queries currently known to the system. The `QueryTracker`:
   - Records query states (QUEUED, RUNNING, FINISHING, etc.).
   - Ensures queries are cleaned up after completion or failure.
   - Provides statistics and introspection into the cluster’s workload.

   This makes it possible to query the system for running queries, fetch their status, and enforce timeouts or memory limits.

3. **Non-Blocking Dispatch and Asynchronous Execution**:  
   The dispatch process is asynchronous. The `DispatchManager`:
   - Returns a future (`ListenableFuture`) that completes once the query is fully created and ready for execution dispatch.
   - Uses a dedicated `dispatchExecutor` to handle queries, ensuring that the main coordinator threads are not blocked by long-running tasks.

   By doing so, it can scale to handle many incoming queries efficiently.

4. **Monitoring and Statistics**:  
   The `DispatchManager` integrates with `QueryMonitor` and `QueryManagerStats`:
   - **`QueryMonitor`**: Fires events (like query creation and completion) for logging, auditing, or integration with external systems.
   - **`QueryManagerStats`**: Periodically updates execution statistics. This helps operators understand cluster health and diagnose performance issues.

5. **Cancellation and Failure Handling**:  
   The `DispatchManager` provides methods to:
   - **Cancel queries**: If a user decides to stop a query before it runs or mid-execution, this component handles that request.
   - **Fail queries**: If some unexpected error arises, it gracefully transitions the query to a failed state, and ensures consistent cleanup and notification.

6. **Handling Query Length and Validation**:  
   The code checks the query length against a configured maximum. This prevents overly large queries from bogging down the system. If the query is too long, it truncates and returns a `QUERY_TEXT_TOO_LARGE` error.

### Conceptual Flow

```
   Incoming Query (SQL text)
          |
          v
   DispatchManager:
       - Check query length
       - Create QueryId
       - Create Session
       - Access Control check
       - Prepare query (parse & plan)
       - Assign Resource Group
       - Create DispatchQuery object
       - Register query with QueryTracker
          |
          v
   If successful: Query is now known to the system and queued or dispatched
   If failure: Creates a failedDispatchQuery, logs event
```

Once the query is in the QueryTracker, the system can manage its lifecycle: waiting in a queue, running on a coordinator, and finally returning results to the client. The `DispatchManager` provides the initial scaffolding—turning a raw incoming request into a well-structured, managed query process.

### Key Takeaways

- The `DispatchManager` acts as the "front desk" for new queries. It reads the "instructions" (SQL), checks the credentials (AccessControl), assigns the "room" (Resource Group), and files the guest’s details (Registers in QueryTracker).
- It never leaves a query in a half-created state. Even on failure, it registers a "failed" query record so the system remains consistent and queries are properly accounted for.
- It supports asynchronous and non-blocking handling, helping Trino scale with many concurrent queries.

In essence, the `DispatchManager` ensures that every query entering the Trino system starts its journey in a controlled, validated, and well-organized manner.