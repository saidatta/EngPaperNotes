Below is a **technical walk-through** of the **`PinotClientRequest`** class from the Apache Pinot Broker module. This class is essentially the **entry point** for handling various types of **SQL and time series queries** via RESTful endpoints. It leverages **JAX-RS annotations** (`@GET`, `@POST`, `@Path`, etc.) and asynchronous request handling** (`@ManagedAsync`) to process client queries. 

---
## High-Level Architecture

```
     ┌─────────────────────────────────────────────────┐
     │                PinotClientRequest              │
     │  (REST API endpoints for queries)             │
     └─────────────────────────────────────────────────┘
              │                   ↑
              │ calls             │ returns
              ▼                   │
     ┌────────────────────────────────────────┐
     │       BrokerRequestHandler            │
     │   (Core request handling logic)       │
     └────────────────────────────────────────┘
              │
              ▼
     ┌────────────────────────────────────────┐
     │        Execution Engine(s)           │
     │   (Single-stage or Multi-stage)      │
     └────────────────────────────────────────┘
```

1. **`PinotClientRequest`**:
   - Defines JAX-RS REST endpoints using `@GET`, `@POST`, `@DELETE`, etc.
   - Orchestrates the request flow. This includes request parsing, authentication/authorization (via custom logic in Pinot), optional tracing, and dispatch to the core broker request handling logic.

2. **`BrokerRequestHandler`** (injected via `@Inject`):
   - Responsible for the actual execution logic.  
   - Once a request is parsed and validated, `PinotClientRequest` calls `_requestHandler.handleRequest(...)` to get a `BrokerResponse`.

3. **Execution Engine(s)**:
   - Pinot has a **single-stage** (a.k.a. “v1” engine) for queries and a **multi-stage** (a.k.a. “v2” engine) introduced for more complex distributed queries.  
   - `PinotClientRequest` can route queries to the appropriate engine based on flags or forcibly by the `forceUseMultiStage` parameter.  
---
## Key Class Fields and Their Roles

```java
@Inject
SqlQueryExecutor _sqlQueryExecutor;

@Inject
private BrokerRequestHandler _requestHandler;

@Inject
private BrokerMetrics _brokerMetrics;

@Inject
private Executor _executor;

@Inject
private HttpClientConnectionManager _httpConnMgr;
```

- **`_sqlQueryExecutor`**: 
  - Handles **DML (Data Manipulation Language)** statements. Pinot recently gained the ability to do certain limited DML operations (e.g. insert, delete) in some contexts. This executor coordinates such statements.  

- **`_requestHandler`**:
  - The primary logic center for read-only (DQL) queries.  
  - Contains `handleRequest(...)`, which processes queries, merges partial results, and constructs the final `BrokerResponse`.

- **`_brokerMetrics`**:
  - Tracks broker-level metrics, such as **exceptions**, **latency**, etc.  

- **`_executor`**:
  - A `java.util.concurrent.Executor` for asynchronous tasks.  
  - Used with `CompletableFuture` to process queries in parallel (e.g. the `compare` endpoint).  

- **`_httpConnMgr`**:
  - Manages HTTP connections for communication (especially when canceling running queries on servers).

---

## REST Endpoints Breakdown

### 1. `/query/sql` (GET and POST)

```java
@GET
@ManagedAsync
@Produces(MediaType.APPLICATION_JSON)
@Path("query/sql")
public void processSqlQueryGet(...)
```

- **Purpose**:  
  Handles **SQL** (single-stage or default path) queries via **HTTP GET**.  
  By default, Pinot uses the single-stage engine unless forced or configured otherwise.  

- **Asynchronous**:  
  Annotated with `@ManagedAsync`, so the request completes in a non-blocking fashion.  

- **Steps**:
  1. Convert query to JSON:  
     ```java
     ObjectNode requestJson = JsonUtils.newObjectNode();
     requestJson.put(Request.SQL, query);
     ```
  2. **Trace** (if `traceEnabled != null`).
  3. Dispatch to `executeSqlQuery(...)`.
  4. Return the `Response` by calling `asyncResponse.resume(getPinotQueryResponse(brokerResponse));`.

- **Exception Handling**:
  - Catches `WebApplicationException` and logs more general errors.  
  - Metrics are updated on exceptions (`_brokerMetrics.addMeteredGlobalValue(...)`).

```java
@POST
@ManagedAsync
@Produces(MediaType.APPLICATION_JSON)
@Path("query/sql")
public void processSqlQueryPost(...)
```
  
- **Purpose**:
  - Same as the GET, but **accepts a POST payload** (JSON).  
- **Important detail**:
  - Expects the JSON payload to have a `sql` field, e.g. `{ "sql": "SELECT * FROM myTable" }`.  

### 2. `/query` (GET and POST) — Multi-Stage Engine

```java
@GET
@ManagedAsync
@Produces(MediaType.APPLICATION_JSON)
@Path("query")
public void processSqlWithMultiStageQueryEngineGet(...)
```

- **Purpose**:
  - Very similar to `/query/sql`, but it uses `executeSqlQuery(..., forceUseMultiStage=true)`.  
  - This explicitly triggers the **new multi-stage engine** pipeline in Pinot.  

- **Example**:
  - `GET /query?sql=SELECT+COUNT(*)+FROM+myTable`

```java
@POST
@ManagedAsync
@Produces(MediaType.APPLICATION_JSON)
@Path("query")
public void processSqlWithMultiStageQueryEnginePost(...)
```
  
- **Purpose**:
  - Again, POST version of the multi-stage query.  
  - Expects JSON, e.g. `{ "sql": "SELECT * FROM myTable" }`.  
  - Forwards it with `forceUseMultiStage=true`.

### 3. `/timeseries/api/v1/query_range` and `/timeseries/api/v1/query`

```java
@GET
@ManagedAsync
@Produces(MediaType.APPLICATION_JSON)
@Path("timeseries/api/v1/query_range")
public void processTimeSeriesQueryEngine(...)
```

- **Purpose**:
  - Exposes a **Prometheus-compatible** endpoint to perform time series queries in Pinot.  
  - This method calls `executeTimeSeriesQuery(...)`, returning a `PinotBrokerTimeSeriesResponse`.  

- **Implementation details**:
  - Query parameters can include language type, time range, etc.  
  - If there's an error in the underlying response, it returns an HTTP `500`.

```java
@GET
@ManagedAsync
@Produces(MediaType.APPLICATION_JSON)
@Path("timeseries/api/v1/query")
public void processTimeSeriesInstantQuery(...)
```
- **Purpose**:
  - Another stub for instant time series queries.  
  - **Currently not implemented** — returns an empty JSON object.

### 4. `/query/compare` (POST)

```java
@POST
@Produces(MediaType.APPLICATION_JSON)
@Path("query/compare")
public void processSqlQueryWithBothEnginesAndCompareResults(...)
```

- **Purpose**:  
  - **Runs the same query (or variants of the same query) on both**:
    1. The **single-stage engine** (v1).  
    2. The **multi-stage engine** (v2).  
  - Then **compares** results, returning a JSON diff.  

- **Flow**:
  1. Parse JSON request, check for `sql`, or use `sqlV1`/`sqlV2`.  
  2. Construct two `ObjectNode` payloads: one for v1, one for v2.  
  3. Use `CompletableFuture` to **execute both in parallel**:
     ```java
     CompletableFuture<BrokerResponse> v1Response = CompletableFuture.supplyAsync(..., _executor);
     CompletableFuture<BrokerResponse> v2Response = CompletableFuture.supplyAsync(..., _executor);
     ```
  4. `CompletableFuture.allOf(v1Response, v2Response).join()` ensures both complete.  
  5. `getPinotQueryComparisonResponse(...)` merges the results and provides a structured diff.  

- **Comparison**:
  - If an exception or mismatch is found, it’s added to `comparisonAnalysis`.  
  - Compares:
    - **Number of columns**, **column types**, **number of rows**,  
    - Aggregation values for simple aggregation-only queries.

### 5. `/query/{queryId}` (DELETE)

```java
@DELETE
@Path("query/{queryId}")
@Authorize(targetType = TargetType.CLUSTER, action = Actions.Cluster.CANCEL_QUERY)
@Produces(MediaType.APPLICATION_JSON)
public String cancelQuery(...)
```

- **Purpose**:
  - **Cancels** a running query with the given `queryId`.  
  - If that query doesn’t exist or is not running, returns `404`.  

- **Mechanics**:
  - Calls `_requestHandler.cancelQuery(...)` with the provided `queryId`.  
  - Optionally collects verbose server responses if `verbose=true`.  

### 6. `/queries` (GET)

```java
@GET
@Path("queries")
@Authorize(targetType = TargetType.CLUSTER, action = Actions.Cluster.GET_RUNNING_QUERY)
@Produces(MediaType.APPLICATION_JSON)
public Map<Long, String> getRunningQueries()
```

- **Purpose**:
  - Returns a **map of running queries** on this broker, where the **key** is `queryId`, and the **value** is the query string.  
  - Could be used by operational tools to see what’s running and optionally cancel.

---

## Core Helper Methods

### `executeSqlQuery(...)`

```java
private BrokerResponse executeSqlQuery(
    ObjectNode sqlRequestJson, HttpRequesterIdentity httpRequesterIdentity,
    boolean onlyDql, HttpHeaders httpHeaders, boolean forceUseMultiStage)
    throws Exception
{
    ...
}
```

1. **Parsing**:
   - Uses `RequestUtils.parseQuery(...)` to parse the `sql` text into a `SqlNodeAndOptions` object.  
   - If `forceUseMultiStage` is `true`, sets an option `Request.QueryOptionKey.USE_MULTISTAGE_ENGINE = "true"`.  

2. **Routing**:
   - Based on **query type** (DQL vs. DML):
     - **DQL** -> `_requestHandler.handleRequest(...)`.
     - **DML** -> `_sqlQueryExecutor.executeDMLStatement(...)`.  

3. **Tracing**:
   - Creates a **`RequestScope`** via Pinot’s `Tracing` utility.  
   - Tracks arrival time and any other metadata.  

4. **Error Handling**:
   - If query fails during parse or execution, it either returns an immediate `BrokerResponseNative` with an error or throws an exception.

### `executeTimeSeriesQuery(...)`

```java
private PinotBrokerTimeSeriesResponse executeTimeSeriesQuery(
    String language, String queryString, RequestContext requestContext)
{
    return _requestHandler.handleTimeSeriesRequest(language, queryString, requestContext);
}
```

- **Purpose**:  
  - Delegates to `_requestHandler.handleTimeSeriesRequest(...)`.  
  - Integrates with the new time series engine logic in Pinot.  

### `makeHttpIdentity(...)`

```java
private static HttpRequesterIdentity makeHttpIdentity(
    org.glassfish.grizzly.http.server.Request context)
{
    // Gathers all HTTP headers, populates them into a MultiMap
    ...
}
```

- **Purpose**:
  - Creates a wrapper object for passing along HTTP headers and the **URL**.  
  - Could be used for **auth** or **logging**.

### `getPinotQueryResponse(...)`

```java
static Response getPinotQueryResponse(BrokerResponse brokerResponse) throws Exception
{
    int queryErrorCodeHeaderValue = -1; // default if no error
    List<QueryProcessingException> exceptions = brokerResponse.getExceptions();
    if (!exceptions.isEmpty()) {
      queryErrorCodeHeaderValue = exceptions.get(0).getErrorCode();
    }
    return Response.ok()
        .header(PINOT_QUERY_ERROR_CODE_HEADER, queryErrorCodeHeaderValue)
        .entity((StreamingOutput) brokerResponse::toOutputStream)
        .type(MediaType.APPLICATION_JSON)
        .build();
}
```

- **Purpose**:  
  - **Wraps** the `BrokerResponse` in an HTTP `Response` object, setting `X-Pinot-Error-Code` header.  
  - Streams the JSON output directly from `brokerResponse.toOutputStream(...)`.  

### `getPinotQueryComparisonResponse(...)`

```java
static Response getPinotQueryComparisonResponse(
    String query, BrokerResponse v1Response, BrokerResponse v2Response)
{
    ObjectNode response = JsonUtils.newObjectNode();
    response.set("v1Response", JsonUtils.objectToJsonNode(v1Response));
    response.set("v2Response", JsonUtils.objectToJsonNode(v2Response));
    response.set("comparisonAnalysis",
        JsonUtils.objectToJsonNode(analyzeQueryResultDifferences(query, v1Response, v2Response)));

    return Response.ok()
        .header(PINOT_QUERY_ERROR_CODE_HEADER, -1)
        .entity(response)
        .type(MediaType.APPLICATION_JSON)
        .build();
}
```

- **Purpose**:
  - **Combines** the single-stage and multi-stage responses in one JSON.  
  - Includes an extra `"comparisonAnalysis"` field with any differences found.  

### `analyzeQueryResultDifferences(...)`

```java
private static List<String> analyzeQueryResultDifferences(
    String query, BrokerResponse v1Response, BrokerResponse v2Response)
{
    // Compares columns, data types, row count, simple aggregates, etc.
    // Returns a list of differences as strings.
}
```

- **Purpose**:
  - Looks for **structural** mismatches (e.g. different columns, row counts, data types).  
  - Also checks for simple **aggregation** mismatches in single-row scenarios.  
  - Returns a list of textual difference messages.

---

## Example Usage (ASCII Demo)

**Single-Stage Query**:  
```
1) Client sends an HTTP GET:

   GET /query/sql?sql=SELECT+COUNT(*)+FROM+myTable

2) PinotClientRequest.processSqlQueryGet(...) → executeSqlQuery(...)
   → BrokerRequestHandler.handleRequest(...)
   → Returns BrokerResponse e.g. { "resultTable": { ... }, ... }

3) PinotClientRequest forms an HTTP 200 OK response with the JSON payload
   including X-Pinot-Error-Code: -1 if success.
```

**Multi-Stage Query**:
```
1) Client sends an HTTP POST to /query with JSON:
   {
     "sql": "SELECT city, COUNT(*) FROM myTable GROUP BY city"
   }

2) PinotClientRequest.processSqlWithMultiStageQueryEnginePost(...)
   -> executeSqlQuery(..., forceUseMultiStage=true)
   -> BrokerRequestHandler.handleRequest(...) uses new multi-stage engine
   -> Returns BrokerResponse with potentially distributed plan results
```

**Compare Query**:
```
1) Client calls:

   POST /query/compare
   {
     "sql": "SELECT COUNT(*) FROM myTable"
   }

2) PinotClientRequest.processSqlQueryWithBothEnginesAndCompareResults(...)
   -> Runs query on v1 and v2 in parallel
   -> Compares columns, row counts, aggregates
   -> Returns a combined JSON with "v1Response", "v2Response",
      and "comparisonAnalysis" that highlights differences.
```

---

## Conclusion

- The **`PinotClientRequest`** class serves as the **main REST layer** for the Pinot Broker. It:
  - Defines endpoints for **query ingestion** (SQL, multi-stage SQL, time series).  
  - **Processes** the incoming requests (parsing, authorization, error handling, tracing).  
  - **Dispatches** them to either the single-stage or multi-stage **execution engine**.  
  - **Handles** specialized features like **DML** statements, **time series** queries, and **canceling** running queries.  
  - **Provides** utility methods for structured **response** construction (especially relevant for asynchronous and comparison-based workflows).

This orchestration class is a critical piece of the Broker’s HTTP interface, ensuring that external clients can interact seamlessly with Pinot’s query engines.