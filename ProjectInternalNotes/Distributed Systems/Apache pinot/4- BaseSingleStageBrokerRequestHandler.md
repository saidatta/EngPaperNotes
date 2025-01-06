Below is a **technical explanation** of the **`BaseSingleStageBrokerRequestHandler`** class in the Apache Pinot Broker module. This class extends `BaseBrokerRequestHandler` but targets the **single-stage** (a.k.a. “v1”) query engine. It incorporates **request parsing**, **rewriting**, **routing** to servers, **merging** partial responses, and **returning** a consolidated `BrokerResponseNative` to the user. Additionally, it manages certain advanced features like subquery handling, column name resolution (when case-insensitivity is enabled), query cancellation, and rewriting queries for approximate/optimized expressions.

---
## High-Level Overview

```
┌───────────────────────────────────────────────────────────────┐
│            BaseBrokerRequestHandler (abstract)               │
│   (Core logic for security, quotas, request context, etc.)   │
└───────────────────────────────────────────────────────────────┘
                   ▲
                   │ extends
                   │
┌───────────────────────────────────────────────────────────────────────┐
│    BaseSingleStageBrokerRequestHandler (abstract)                    │
│   (Single-stage logic: compile queries, route to servers, handle     │
│    case-insensitive columns, subqueries, server timeouts, etc.)      │
└───────────────────────────────────────────────────────────────────────┘
                                       ▲ 
                                       │ extends or is extended by
                                       │ (e.g., SingleConnectionBrokerRequestHandler)
                                       ▼
             ┌────────────────────────────────────────┐
             │  Concrete single-stage request handler │
             └────────────────────────────────────────┘
```

1. **Inheritance**:  
   - Inherits the **base** logic from `BaseBrokerRequestHandler` (authorization checks, quota checks, etc.).  
   - Defines single-stage specific logic such as:
     - Query rewriting (e.g., approximate function overrides).  
     - Multi-value column overrides.  
     - Routing to **offline** / **realtime** tables.  
     - Subquery expansions (e.g., rewriting `IN_SUBQUERY` to `IN_ID_SET`).  

2. **Abstract**:
   - The actual method to **dispatch** and **collect** from servers is left to a **subclass** in `processBrokerRequest(...)`.  
   - This separation allows Pinot to vary how single-stage queries are physically sent and merged.

---

## Notable Fields

```java
protected final QueryOptimizer _queryOptimizer = new QueryOptimizer();
protected final boolean _disableGroovy;
protected final boolean _useApproximateFunction;
protected final int _defaultHllLog2m;
protected final boolean _enableQueryLimitOverride;
protected final boolean _enableDistinctCountBitmapOverride;
protected final int _queryResponseLimit;
protected final Map<Long, QueryServers> _queriesById;
protected final boolean _enableMultistageMigrationMetric;
protected ExecutorService _multistageCompileExecutor;
protected BlockingQueue<Pair<String, String>> _multistageCompileQueryQueue;
```
1. **`_disableGroovy`** (`boolean`):
   - If set to `true`, **rejects** any queries that use Groovy transform functions.  
   - Some Pinot deployments disable Groovy for performance or security reasons.
2. **`_useApproximateFunction`** (`boolean`):
   - If `true`, rewrites exact aggregation functions (e.g. `DISTINCTCOUNT`) to approximate ones (e.g. `DISTINCTCOUNTSMART-HLL`).
3. **`_defaultHllLog2m`** (`int`):
   - If greater than 0, overrides `DISTINCTCOUNTHLL` function’s `log2m` parameter unless explicitly set.
4. **`_enableQueryLimitOverride`** / **`_queryResponseLimit`**:
   - Broker-level **limit** for query results (e.g. `LIMIT <= 30000`).  
   - If the user’s query sets a higher limit, we can clamp it to `_queryResponseLimit`.
5. **`_enableDistinctCountBitmapOverride`**:
   - If `true`, rewrites `distinctcount(...)` to `distinctcountbitmap(...)`.
6. **`_queriesById`** (`Map<Long, QueryServers>`):
   - **Tracks** running queries if **query cancellation** is enabled.  
   - Key: `requestId`, Value: object containing the query string and the set of servers.
7. **`_enableMultistageMigrationMetric`** / **`_multistageCompileExecutor`**:
   - Mechanism to see if the same query could be compiled using the “v2” engine.  
   - Allows metrics to assess readiness or coverage for multi-stage engine adoption.

---
## Constructor
```java
public BaseSingleStageBrokerRequestHandler(PinotConfiguration config, String brokerId, ...)
{
  super(config, brokerId, routingManager, accessControlFactory, queryQuotaManager, tableCache);

  _disableGroovy = _config.getProperty(Broker.DISABLE_GROOVY, ...);
  _useApproximateFunction = _config.getProperty(Broker.USE_APPROXIMATE_FUNCTION, false);
  _defaultHllLog2m = _config.getProperty(...);
  ...
  boolean enableQueryCancellation = Boolean.parseBoolean(
      config.getProperty(Broker.CONFIG_OF_BROKER_ENABLE_QUERY_CANCELLATION));
  _queriesById = enableQueryCancellation ? new ConcurrentHashMap<>() : null;

  _enableMultistageMigrationMetric = _config.getProperty(
      Broker.CONFIG_OF_BROKER_ENABLE_MULTISTAGE_MIGRATION_METRIC, ...);

  if (_enableMultistageMigrationMetric) {
    _multistageCompileExecutor = Executors.newSingleThreadExecutor();
    _multistageCompileQueryQueue = new LinkedBlockingQueue<>(1000);
  }

  LOGGER.info("Initialized {} with broker id: ...", getClass().getSimpleName(), ...);
}
```

- **Initializes** fields based on broker config.  
- If `_enableMultistageMigrationMetric` is `true`, spawns a background **compile** thread to see if queries can be run in multi-stage mode.  
- If `enableQueryCancellation` is `true`, sets up a `_queriesById` map to track in-flight queries.

---
## Lifecycle Methods: `start()` and `shutDown()`

```java
@Override
public void start() {
  if (_enableMultistageMigrationMetric) {
    _multistageCompileExecutor.submit(() -> { 
      // drains _multistageCompileQueryQueue & checks 
      // if queries are valid for multi-stage
    });
  }
}

@Override
public void shutDown() {
  if (_enableMultistageMigrationMetric) {
    _multistageCompileExecutor.shutdownNow();
  }
}
```

- **`start()`**: Kicks off a single-thread loop that **consumes** queued queries, checking if they can be compiled by the multi-stage engine.  
- **`shutDown()`**: Gracefully stops that thread.
---
## Core Method: `handleRequest(...)`
This is the **heart** of single-stage query handling. It orchestrates:
```java
@Override
protected BrokerResponse handleRequest(
    long requestId, String query, SqlNodeAndOptions sqlNodeAndOptions, JsonNode request,
    @Nullable RequesterIdentity requesterIdentity, RequestContext requestContext,
    @Nullable HttpHeaders httpHeaders, AccessControl accessControl) throws Exception {
  ...
}
```

1. **Compile** the query into a **`PinotQuery`**:  
   ```java
   pinotQuery = CalciteSqlParser.compileToPinotQuery(sqlNodeAndOptions);
   ```
   - If compilation fails, check if the query is actually a multi-stage-only query. If so, return an appropriate error message.

2. **Literal-Only Optimization**:  
   - If the query is only **constant expressions** (no columns or tables), we can directly compute the result in the broker.  

3. **Check** subqueries (e.g. `IN_SUBQUERY`):
   - If found, the broker recursively **executes** that subquery, obtains a result set (like an ID set), and **rewrites** it into a simpler form (e.g. `IN_ID_SET`).

4. **Resolve Table Name** and fix column names for **case-insensitive** clusters:
   ```java
   String tableName = getActualTableName(..., _tableCache);
   updateColumnNames(rawTableName, pinotQuery, ignoreCase, columnNameMap);
   ```

5. **HLL Log2m / Distinct** rewriting:
   - If `_defaultHllLog2m` > 0, override certain `distinctcountHLL(...)`.
   - If `_enableDistinctCountBitmapOverride` is `true`, turn `distinctcount(...)` -> `distinctcountbitmap(...)`.  
   - If the table has segment-partitioned columns, rewrite `distinctcount(col)` -> `segmentpartitioneddistinctcount(col)`.

6. **Apply Quotas**:
   - Database-level, table-level QPS checks.  

7. **Validate** or rewrite the request:
   - e.g. clamp `LIMIT` if `_enableQueryLimitOverride` is `true`.  
   - throw exceptions if something fails (like limit too large, groovy disabled, etc.).

8. **Derive** offline vs. realtime routing:
   - e.g., if table type is `HYBRID`, query can fan out to both offline and realtime.  
   - Retrieve `TimeBoundaryInfo` for offline portion if needed.  

9. **Compute** or retrieve the **routing tables** from `_routingManager`:
   - Produces a map of `ServerInstance -> {listOfSegments...}`.  
   - Tracks **unavailable segments**, sets partial results if needed.

10. **Attach** timeouts, max server response sizes, etc.

11. **Dispatch** to servers / gather responses:
   ```java
   BrokerResponseNative brokerResponse = processBrokerRequest(...);
   ```
   - This is an **abstract** method. Concrete classes handle the netty-based or single-connection approach to sending requests.

12. **Fill** metadata into the final `BrokerResponse`:
   - e.g. `setNumSegmentsPrunedByBroker(numPrunedSegmentsTotal)`.  
   - If subqueries or rewriting appended new exceptions, add them to the response.
13. **Record** final metrics, log the query.
---
## Subquery Handling

- The code looks for functions named `IN_SUBQUERY(...)`.  
- It runs that subquery first, obtains a single result (e.g. an ID set as a string), then **rewrites** the parent filter to `IN_ID_SET(..., 'serializedIDSet')`.  
- This allows a second pass that only references **literal** values, avoiding complex multi-round routing on the same query.

---

## Column Name Resolution and `*` Expansion

```java
if (hasStar) {
  // Expand `SELECT *` to all columns in the table
}
```

- If a query includes `SELECT *`, we **expand** it to every column in the schema (excluding special `$` columns).  
- If case-insensitivity is **enabled**, map the user’s column references to actual columns in `_tableCache`.

---

## Query Cancellation

- If **enabled** (`enableQueryCancellation = true`), the class populates `_queriesById` with a mapping of `requestId` to the servers being queried.  
- `cancelQuery(...)` sends an HTTP `DELETE` to those servers, instructing them to cancel if still running.

---

## Approximate Function Overrides

- If `_useApproximateFunction` is `true`, transforms:
  - `distinctcount(...)` -> `distinctcountsmarthll(...)`.  
  - `percentileXX(...)` -> `percentilesmarttdigest(...)`.

---

## Final Execution: `processBrokerRequest(...)`

```java
protected abstract BrokerResponseNative processBrokerRequest(
    long requestId, BrokerRequest originalBrokerRequest, BrokerRequest serverBrokerRequest,
    @Nullable BrokerRequest offlineBrokerRequest, @Nullable Map<ServerInstance, Pair<List<String>, List<String>>> offlineRoutingTable,
    @Nullable BrokerRequest realtimeBrokerRequest, @Nullable Map<ServerInstance, Pair<List<String>, List<String>>> realtimeRoutingTable,
    long timeoutMs, ServerStats serverStats, RequestContext requestContext) throws Exception;
```
- **Subclasses** (such as `SingleConnectionBrokerRequestHandler`) implement the network scatter/gather to servers:
  1. Build a netty or HTTP request with the `offlineBrokerRequest` and `realtimeBrokerRequest`.  
  2. Send them to all relevant servers in the offline and realtime routing tables.  
  3. Collect partial results, do final merges, and produce a single `BrokerResponseNative`.
---
## Conclusion
**`BaseSingleStageBrokerRequestHandler`** is the **foundation** for Pinot’s single-stage (v1) query processing. Its responsibilities include:
1. **Compiling** queries into Pinot’s internal representation (`PinotQuery`).  
2. **Rewriting** expressions for approximate or function overrides (HLL, distinct-count-bitmap, etc.).  
3. **Handling** subqueries in the filter.  
4. **Resolving** the correct offline/realtime table segments and computing a route plan.  
5. **Applying** broker-level policies (time-out, max server response size, etc.).  
6. **Delegating** actual scatter/gather to the (still abstract) `processBrokerRequest(...)`.  
Through this design, Pinot can easily adapt the single-stage engine’s front-end logic while keeping core broker security and quota checks centralized in the parent class `BaseBrokerRequestHandler`.