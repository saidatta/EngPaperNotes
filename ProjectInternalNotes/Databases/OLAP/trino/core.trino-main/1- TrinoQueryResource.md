Below is a logical explanation of what the given Trino code snippet does in the context of how Trino handles queries internally. We’ll break it down step-by-step, using conceptual metaphors, examples, and ASCII diagrams to help clarify the flow:

---
### High-Level Purpose

Trino’s query engine involves multiple steps from the moment a user sends a SQL query until the query results are retrieved. The code you see here is part of Trino's front-end API layer that manages "queued" queries. It handles:
1. **Receiving a new SQL query** (via a POST request).
2. **Registering the query** and enqueuing it so it can be dispatched (executed) by the engine.
3. **Providing endpoints** for polling the query’s status, retrieving partial or final results.
4. **Handling cancellation** and cleanup of queries.

This code primarily deals with the early lifecycle of a query (the "queued" state before execution), how the client is informed about the query’s progress, and how the system ensures queries that never properly start are abandoned and cleaned up.

---
### Logical Flow of a Query

1. **Posting a Query** (`POST /v1/statement`):
   - A user sends a SQL statement to Trino.
   - The system checks if the statement is empty.
   - If valid, it creates a `Query` object.
     - **SessionContext**: Identifies who is running the query and with what settings.
     - **QueryId**: A unique identifier for this query.
     - **Dispatcher**: The query isn’t executed immediately. Instead, it’s registered with a `DispatchManager` to eventually be run on a coordinator, and possibly worker nodes.

   **ASCII:**
   ```
   User's SQL ---> [POST /v1/statement] ---> Query registered ---> Returns initial QueryResults (QUEUED)
   ```

   The immediate response to the user is a JSON structure (`QueryResults`) that indicates the query is queued and not yet running.

2. **Query Registration**:
   - The `registerQuery` method creates a `Query` object and records it in the `QueryManager`.
   - The `QueryManager` tracks all queries currently known to the system. While the query is in QUEUED state, it doesn’t have actual running tasks yet.
   - The `DispatchManager` is responsible for "dispatching" the query to run on the actual Trino cluster (e.g., deciding which coordinator or cluster resources will handle it).

   **ASCII:**
   ```
   QueryManager (in memory) holds: [QueryId -> Query object]

   DispatchManager (handles the logic to start the query execution)
   ```

3. **Polling for Status** (`GET /v1/statement/queued/{queryId}/{slug}/{token}`):
   - After submitting the query, the client periodically checks if the query has started running or completed.
   - The `GET` endpoint retrieves the current `Query` state and returns updated `QueryResults`.
   - If the query is still not dispatched, it returns QUEUED status.  
   - If dispatched, it might return a redirect URL to a different endpoint (`/v1/statement/executing/...`) where execution results can be fetched.
   - This endpoint uses a "token" to ensure that the client and server stay in sync about which "page" or "state" of the query’s output they are looking at.

   **ASCII:**
   ```
   Client --> GET /.../queued/{queryId}/{slug}/{token} --> Server checks Query state
        If still queued: returns updated QUEUED state with next polling URL
        If dispatched: redirects client to executing endpoint for actual results
   ```

   The `maxWait` parameter allows for a "long-polling" approach: the client’s request can wait briefly for state changes, reducing the number of polling requests needed.

4. **Cancelling a Query** (`DELETE /v1/statement/queued/{queryId}/{slug}/{token}`):
   - If the client decides to stop the query before it runs, they call DELETE.
   - The system cancels the queued query so it won’t start (or if it’s in the process of starting, it’s aborted).

   **ASCII:**
   ```
   Client ---> DELETE /.../queued/{queryId} --> Query is marked as cancelled --> No further execution
   ```

5. **Query Lifecycle Management & Cleanup**:
   - The `QueryManager` periodically checks all registered queries.
   - If a query has not been properly dispatched after a certain timeout (it’s stuck or the client never returned to start it), it’s abandoned.
   - Abandoned or finished queries are removed from the `QueryManager`, and resources are cleaned up.

   **ASCII:**
   ```
   QueryManager background thread:
       Check queries -> If submission timed out or done -> remove query -> free resources
   ```

6. **Internal Query Object**:
   Inside the `Query` class:
   - The query holds the original SQL statement, the user session, and a `Slug` (a security token-like structure used to ensure that URLs are valid and not guessable).
   - `submitIfNeeded()` starts the actual creation and dispatch of the query in the `DispatchManager`.
   - The query tracks tokens to synchronize client requests with the server’s state transitions:
     - Every time the client requests results with a certain token, the server moves to the next token once it responds. This ensures a stateful, incremental exchange.
   - When the query eventually is dispatched, the next response redirects to the appropriate “executing” endpoint.

   **ASCII:**
   ```
   Query:
     - queryId: unique ID
     - slug: used to create secure URLs
     - sessionContext: user/session info
     - dispatchManager: triggers actual execution
     - keeps track of lastToken, so each client request increments this token
   ```

7. **QueryResults and Redirections**:
   - Initially, `QueryResults` show a state of `QUEUED`.
   - Once the query is ready to run, the returned `QueryResults` will contain a `nextUri` pointing to a different endpoint (like `/v1/statement/executing/...`).
   - If the query fails, `QueryResults` will contain an `error` block with details.
   - If the query finishes successfully, future requests might show `QueryResults` with data or indicate that the query completed.

   **ASCII:**
   ```
   QueryResults:
     state: QUEUED | FAILED | (later, RUNNING/FINISHED)
     nextUri: URL to poll for next state or data
     error: optional if something went wrong
   ```

---

### Putting It All Together

- **User Submits Query**: They POST a SQL statement. The server registers the query, returns QUEUED state.
- **User Polls for Status**: They GET a status endpoint. If the query is still queued, it returns a QUEUED `QueryResults`. Once the query moves forward, it redirects them to the actual execution endpoint.
- **Timeouts & Cleanup**: If the user disappears and never proceeds, or if something goes wrong and the query never gets dispatched, the system eventually abandons the query and frees resources.
- **Cancellation**: If the user decides to cancel at any point, a DELETE request stops the query.

This code snippet shows how Trino decouples the initial query submission phase (QUEUED) from the eventual execution phase. It provides a REST-ful interface for managing query states, ensuring the client and server stay in sync, and prevents orphaned queries from living forever.

---

### In Summary

This part of the code handles the initial "queued" stage of a Trino query’s lifecycle via HTTP endpoints. It manages:

- **Registration of new queries** with an internal manager (`QueryManager`).
- **Long-polling for query state** so the client can see when the query moves from queued to executing.
- **Redirection and token mechanism** to ensure proper state transitions and avoid stale requests.
- **Safe cleanup and cancellation** to maintain system hygiene and resource management.

----
Below is an ASCII diagram showing an end-to-end flow of the described Trino query submission and retrieval process. This includes what happens when a user submits a query, how it transitions through the queued state, and how the client obtains results or a redirect to the executing state.

```
     +-------------------------+
     |         Client          |
     | (e.g. curl, JDBC, etc.)|
     +-----------+-------------+
                 |
                 | POST /v1/statement with SQL query
                 v
       +---------+----------+
       |    Trino HTTP     |
       |    Endpoint       |
       |  (StatementResource)
       +---------+----------+
                 |
                 | Create Query object and SessionContext
                 | Register query with QueryManager
                 v
         +-------+--------+
         |   QueryManager |
         |  (Holds queries|
         |   in memory)   |
         +-------+--------+
                 |
                 | Creates QueryId, stores Query
                 | Query is not executed yet, "queued"
                 v
        +--------+----------+
        | DispatchManager   |
        | (Manages dispatch |
        | of queries to     |
        | coordinators)     |
        +-------------------+

             (No actual dispatch yet; query is in QUEUED state)

                 |
                 | Returns initial QueryResults (QUEUED)
                 |
                 v
       +---------+----------+
       |    Trino HTTP     |
       |    Endpoint       |
       +---------+----------+
                 |
                 | Respond to client with:
                 |  { "id": "queryId",
                 |    "state": "QUEUED",
                 |    "nextUri": "/v1/statement/queued/{queryId}/{slug}/{token}"
                 |  }
                 v
     +-----------+-------------+
     |         Client          |
     +-----------+-------------+
                 |
                 | GET /v1/statement/queued/{queryId}/{slug}/{token}
                 v
       +---------+----------+
       |    Trino HTTP     |
       |    Endpoint       |
       +---------+----------+
                 |
                 | Checks Query state in QueryManager
                 | If still queued and not dispatched:
                 |   returns updated QUEUED results (with nextUri)
                 | else if dispatched:
                 |   returns redirect nextUri -> /v1/statement/executing/...
                 v
         +-------+--------+
         |   QueryManager | 
         |                |
         +-------+--------+
                 |
                 | If enough time passes and
                 | query not dispatched or client silent:
                 |   QueryManager abandons the query
                 |   and removes it
                 |
                 | Otherwise, DispatchManager eventually
                 | dispatches the query
                 v
        +--------+----------+
        | DispatchManager   |
        | (Now decides on   |
        | coordinator and   |
        | starts execution) |
        +--------+----------+
                 |
                 | Once dispatched, query moves to RUNNING state
                 | DispatchManager updates Query's nextUri to an executing endpoint
                 v
       +---------+----------+
       |    Trino HTTP     |
       |    Endpoint       | <--- If client polls again (GET /queued/...),
       +---------+----------+      it now returns a redirect to "/executing/..."
                 |
                 | Client follows nextUri to "/executing"
                 |
                 v
            (EXECUTING ENDPOINTS)
                 |
                 | Client now fetches results pages, final results, etc.
                 |
                 v
     +-----------+-------------+
     |         Client          |
     +-------------------------+

```

**Key Points in the Diagram**:
- The client first POSTs a query, and gets back a QUEUED state with a polling URL.
- The client keeps polling the `/queued/...` endpoint until the query transitions from QUEUED to either EXECUTING or FAILING.
- Once dispatched, the server provides a `nextUri` to a different endpoint (like `/executing/...`) for actual data retrieval.
- If the client never comes back or too much time passes, the `QueryManager` purges the query.
- Deleting the query (`DELETE`) also returns the query to a cancelled state, stopping further actions.

This ASCII sketch shows the lifecycle and interaction between the client and the Trino server components handling a query’s queued stage until it’s dispatched.