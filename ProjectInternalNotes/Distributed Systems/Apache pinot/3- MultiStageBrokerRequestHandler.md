Below is a **technical explanation** of the **`MultiStageBrokerRequestHandler`** class in the Apache Pinot Broker module. This class extends `BaseBrokerRequestHandler` and thus inherits its foundational request-handling logic. However, `MultiStageBrokerRequestHandler` specifically focuses on **multi-stage execution** of SQL queries (i.e., the “v2” or distributed execution engine). It **compiles** a given query into **dispatchable** sub-plans, **routes** these sub-plans to worker servers, **collects partial results**, and **assembles** the final `BrokerResponse`.

---

## High-Level Overview

```
 ┌─────────────────────────────────────────────────────┐
 │          BaseBrokerRequestHandler (abstract)       │
 │    (Common security + parsing + quota + metrics)   │
 └─────────────────────────────────────────────────────┘
                   ▲                ▲
                   │ inherits       │
                   │                │ calls
                   │                ▼
 ┌──────────────────────────────────────────────────────────┐
 │       MultiStageBrokerRequestHandler (concrete)         │
 │ - Builds query execution plan for multi-stage engine    │
 │ - Dispatches plan fragments to workers using            │
 │   WorkerManager + QueryDispatcher                       │
 │ - Merges partial results into final BrokerResponse      │
 └──────────────────────────────────────────────────────────┘
```

1. **Inheritance from `BaseBrokerRequestHandler`**:
   - Uses the same **security checks**, **quota checks**, and **request context** logic.  
   - Overrides the `handleRequest(...)` method to focus on multi-stage query planning and execution.

2. **Core Responsibilities**:
   - **Plan** the query using `QueryEnvironment`.
   - **Dispatch** plan fragments to worker nodes via `QueryDispatcher`.
   - **Collect** partial results, handle timeouts and errors, and **synthesize** the final `BrokerResponse`.
   - **Generate** additional “explain” output if requested.

---

## Key Fields and Their Significance

```java
private final WorkerManager _workerManager;
private final QueryDispatcher _queryDispatcher;
private final boolean _explainAskingServerDefault;
private final MultiStageQueryThrottler _queryThrottler;
```

1. **`_workerManager` (`WorkerManager`)**:  
   - Knows how to locate or assign **workers** (the execution nodes) for each plan fragment.  
   - Tightly integrated with the routing manager to find the appropriate servers.

2. **`_queryDispatcher` (`QueryDispatcher`)**:  
   - Takes a **DispatchableSubPlan** and **submits** it to workers.  
   - Collects mailbox results, merges them, and **returns** them to the broker.  
   - Provides specialized methods like `submitAndReduce()` and `explain()`.

3. **`_explainAskingServerDefault` (`boolean`)**:  
   - Determines whether queries **by default** should produce a detailed “explain” that includes **server-level** plan data.

4. **`_queryThrottler` (`MultiStageQueryThrottler`)**:  
   - Controls concurrency or **throttles** the number of simultaneously running multi-stage queries.  
   - Uses something like a semaphore to limit how many queries can be executed at once.

---

## Constructor

```java
public MultiStageBrokerRequestHandler(
    PinotConfiguration config, String brokerId, BrokerRoutingManager routingManager,
    AccessControlFactory accessControlFactory, QueryQuotaManager queryQuotaManager,
    TableCache tableCache, MultiStageQueryThrottler queryThrottler) {

  super(config, brokerId, routingManager, accessControlFactory, queryQuotaManager, tableCache);
  ...
  _workerManager = new WorkerManager(hostname, port, _routingManager);
  _queryDispatcher = new QueryDispatcher(
      new MailboxService(hostname, port, config, tlsConfig), tlsConfig);
  ...
  _explainAskingServerDefault = config.getProperty(
      CommonConstants.MultiStageQueryRunner.KEY_OF_MULTISTAGE_EXPLAIN_INCLUDE_SEGMENT_PLAN,
      CommonConstants.MultiStageQueryRunner.DEFAULT_OF_MULTISTAGE_EXPLAIN_INCLUDE_SEGMENT_PLAN);
  _queryThrottler = queryThrottler;
}
```

- Calls **`super(...)`** to initialize the **base** fields (e.g., `_brokerId`, `_brokerMetrics`, etc.).  
- Instantiates `WorkerManager` for sub-plan distribution.  
- Creates a new `QueryDispatcher`, optionally configured with **TLS** if enabled (`tlsConfig`).  
- Reads flags like **`_explainAskingServerDefault`** from config for controlling explain plan specifics.  
- Hooks up the **`_queryThrottler`**.

---

## Lifecycle Methods

### `start()` and `shutDown()`

```java
@Override
public void start() {
  _queryDispatcher.start();
}

@Override
public void shutDown() {
  _queryDispatcher.shutdown();
}
```

- **Purpose**:  
  - Invoked when the broker starts up or shuts down.  
  - The `QueryDispatcher` may need to **start** its mailbox service or **stop** network connections.

---

## Overridden `handleRequest(...)`

The **heart** of the class:

```java
@Override
protected BrokerResponse handleRequest(
    long requestId, String query, SqlNodeAndOptions sqlNodeAndOptions,
    JsonNode request, @Nullable RequesterIdentity requesterIdentity,
    RequestContext requestContext, HttpHeaders httpHeaders, AccessControl accessControl) {
  ...
}
```

### Step-by-Step Logic

1. **Log the incoming query**:  
   ```java
   LOGGER.debug("SQL query for request {}: {}", requestId, query);
   ```

2. **Extract/Compute Query Options**:  
   - `queryOptions = sqlNodeAndOptions.getOptions()`.  
   - For instance, a `TIMEOUT_MS` or `EXPLAIN` flag might be in these options.

3. **Compile the Query**:  
   ```java
   long compilationStartTimeNs = System.nanoTime();
   QueryEnvironment.QueryPlannerResult queryPlanResult = queryEnvironment.planQuery(query, ...);
   ...
   long compilationTimeNs = (compilationEndTimeNs - compilationStartTimeNs)
       + sqlNodeAndOptions.getParseTimeNs();
   updatePhaseTimingForTables(tableNames, BrokerQueryPhase.REQUEST_COMPILATION, compilationTimeNs);
   ```
   - Uses **`QueryEnvironment`** to parse, validate, and build a **`DispatchableSubPlan`**.  
   - May also handle `EXPLAIN` queries differently:
     - If `EXPLAIN SELECT ...`, it either returns a textual plan or tries an “on-server” approach if `askServers` is `true`.

4. **Verify Table Authorization**:  
   ```java
   Set<String> tableNames = queryPlanResult.getTableNames();
   TableAuthorizationResult tableAuthorizationResult =
       hasTableAccess(requesterIdentity, tableNames, requestContext, httpHeaders);
   if (!tableAuthorizationResult.hasAccess()) { ... throw 403 ... }
   ```
   - Ensures the user can access these **tables**.

5. **QPS Quota Check**:  
   ```java
   if (hasExceededQPSQuota(database, tableNames, requestContext)) {
     return new BrokerResponseNative(QueryException.getException(...));
   }
   ```
   - Ensures we haven’t exceeded **per-database** or **per-table** QPS limits.

6. **Acquire Query Throttler**:  
   ```java
   if (!_queryThrottler.tryAcquire(queryTimeoutMs, TimeUnit.MILLISECONDS)) {
     // Timed out waiting to acquire concurrency slot
     return new BrokerResponseNative(QueryException.EXECUTION_TIMEOUT_ERROR);
   }
   ```
   - Uses a **semaphore** or similar mechanism to limit concurrent multi-stage queries.  
   - If we cannot acquire a slot within `queryTimeoutMs`, return a **timeout** response.

7. **Execute the Query** (`_queryDispatcher.submitAndReduce(...)`):
   ```java
   QueryDispatcher.QueryResult queryResults =
       _queryDispatcher.submitAndReduce(requestContext, dispatchableSubPlan, queryTimer.getRemainingTime(), queryOptions);
   ```
   - The broker dispatches plan fragments to multiple servers (workers).  
   - Waits for partial results via mailboxes.  
   - Merges the data into a single **`ResultTable`** or set of aggregates.

8. **Build the Response** (`BrokerResponseNativeV2`):
   ```java
   BrokerResponseNativeV2 brokerResponse = new BrokerResponseNativeV2();
   brokerResponse.setResultTable(queryResults.getResultTable());
   brokerResponse.setTablesQueried(tableNames);
   brokerResponse.setBrokerReduceTimeMs(queryResults.getBrokerReduceTimeMs());
   ...
   brokerResponse.setTimeUsedMs(totalTimeMs);
   augmentStatistics(requestContext, brokerResponse);
   ...
   ```
   - Attach **metadata**: time used, partial segments, exceptions, stage stats, etc.  
   - If **`DROP_RESULTS`** is requested, sets the result table to `null`.

9. **Log** the query stats using `_queryLogger.log(...)`.

10. **Return** the `BrokerResponse`.

11. **Finally** (in a `finally` block) **release** the throttler slot:  
   ```java
   _queryThrottler.release();
   ```

---

## Other Notable Methods

### `requestPhysicalPlan(...)`

```java
private Collection<PlanNode> requestPhysicalPlan(
    DispatchablePlanFragment fragment, RequestContext requestContext,
    long queryTimeoutMs, Map<String, String> queryOptions) {
  ...
  stagePlans = _queryDispatcher.explain(requestContext, fragment, queryTimeoutMs, queryOptions);
  ...
}
```

- For `EXPLAIN` queries with `askServers = true`, the broker instructs each server to return its **physical plan** for that fragment.  
- Used in the `EXPLAIN` flow to get a more **detailed** plan.

### `fillOldBrokerResponseStats(...)`

```java
private void fillOldBrokerResponseStats(
    BrokerResponseNativeV2 brokerResponse,
    List<MultiStageQueryStats.StageStats.Closed> queryStats,
    DispatchableSubPlan dispatchableSubPlan) {
  ...
}
```

- Populates the older (v1-like) broker response fields for backward compatibility.  
- Uses `MultiStageStatsTreeBuilder` to turn stage stats into JSON.

### `hasTableAccess(...)` Override

```java
private TableAuthorizationResult hasTableAccess(
    RequesterIdentity requesterIdentity, Set<String> tableNames,
    RequestContext requestContext, HttpHeaders httpHeaders) {
  ...
}
```

- Checks table-level authorization for a set of tables.  
- If the user lacks permission on **any** table, returns a combined denial result.

### `hasExceededQPSQuota(...)`

```java
private boolean hasExceededQPSQuota(
    @Nullable String database, Set<String> tableNames, RequestContext requestContext) {
  ...
}
```

- Verifies **database-level** QPS (if `database != null`) and **table-level** QPS.  
- If quotas are exceeded, sets an error code and returns `true`.

### `constructMultistageExplainPlan(...)`

```java
private BrokerResponse constructMultistageExplainPlan(String sql, String plan) {
  BrokerResponseNative brokerResponse = BrokerResponseNative.empty();
  List<Object[]> rows = new ArrayList<>();
  rows.add(new Object[]{sql, plan});
  ...
  brokerResponse.setResultTable(...);
  return brokerResponse;
}
```

- Builds a **`BrokerResponse`** that contains only 2 columns: `"SQL"`, `"PLAN"`.  
- Typically used to **return** a textual plan when `EXPLAIN` queries are run.
### Query Cancellation and Running Queries

- `getRunningQueries()` and `cancelQuery(...)` are **not** yet supported in the multi-stage engine:
  ```java
  throw new UnsupportedOperationException();
  ```
  - Future PRs may add advanced features to track multi-stage queries and cancel them mid-flight.
---
## Example Flow (ASCII)
**Request**: `SELECT city, COUNT(*) FROM events GROUP BY city;`
1. **Broker** calls `handleRequest(...)` in `MultiStageBrokerRequestHandler`.  
2. **Base** logic sets `requestId`, validates `sql`, checks security, etc.  
3. Multi-stage logic:
   1. **Compile** query with `QueryEnvironment.planQuery(...)` => yields a `DispatchableSubPlan`.  
   2. **Throttle** – ensure concurrency is within limits.  
   3. **Dispatch** sub-plans to workers via `_queryDispatcher.submitAndReduce(...)`.  
   4. Workers process partial data, send results back through mailboxes.  
   5. The broker merges partial results into a `ResultTable`.  
   6. **Return** the final `BrokerResponseNativeV2` JSON to the client.
---
## Conclusion
`MultiStageBrokerRequestHandler` is a **specialized** extension of `BaseBrokerRequestHandler` that:

1. **Integrates** with the new Pinot multi-stage (a.k.a. “v2”) query engine.  
2. **Dispatches** compiled plans to a set of worker servers (`QueryDispatcher` + `WorkerManager`).  
3. Handles **throttling** (`_queryThrottler`), **timeout** logic, **error** handling, and **result** merging.  
4. Optionally **explains** queries in detail via on-server physical plan retrieval.  

It **completes** the multi-stage pipeline by orchestrating the entire query lifecycle: parse, plan, authorize, dispatch, merge, and respond.