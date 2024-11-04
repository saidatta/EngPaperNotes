## Overview

In this chapter, we delve into the concept of **streaming databases**, which represent the next evolution in data processing architectures. We'll explore how streaming databases unify stream processing and database capabilities, simplifying architectures and reducing complexity.

---

## Key Concepts

### Turning the Database Inside Out

- **Martin Kleppmann's Insight**: In *Designing Data-Intensive Applications*, Kleppmann compares databases to spreadsheets where formulas automatically recalculate when inputs change.
- **Materialized Views in Stream Processing**:
  - **Traditional Databases**: Materialized views refresh periodically (minutes to hours).
  - **Stream Processing Platforms**: Materialized views update continuously with each new change.
- **Challenge**: The "inside-out" approach introduces complexity by requiring multiple systems (streaming platform, stream processor, external database).

### Moving Towards Streaming Databases

- **Goal**: Consolidate components to reduce complexity.
- **Approach**: Merge the stream processor and database, eliminating the need for intermediary topics and multiple systems.
- **Result**: A **streaming database** that handles both stream processing and data storage/querying.

---

## Components of a Streaming Architecture

![Stream Processing Parts](stream_processing_parts.png)

### Components Explained

1. **Database**:
   - Types: OLTP, RTOLAP, internal state stores in stream processors.
   - Role: Stores data; how data is stored and queried varies by type.

2. **Topic**:
   - Mimics the Write-Ahead Log (WAL) in an OLTP database.
   - Publishes streams of data to other databases and stream processors.

3. **Stream Processor**:
   - Transforms streams of data.
   - Maintains an internal **state store**.

4. **Materialized View**:
   - Precomputes results and stores them.
   - Can be created in a database or stream processor.
   - Requires a persistence layer.

### Data Flows

- **Typical Real-Time Analytical Flow**:

  ```plaintext
  OLTP → Topic → Stream Processor → Topic → RTOLAP
  ```

- **Alternate Flow with Same OLTP as Destination**:

  ```plaintext
  OLTP → Topic → Stream Processor → Topic → OLTP
  ```

### Materialized View Location Ambiguity

- Materialized views span across the stream processor's state store, output topic, and destination database.
- None of these components alone serve as a first-class materialized view.

---

## Consolidating Components into a Streaming Database

![Consolidated Streaming Database](consolidated_streaming_database.png)

- **Consolidation**:
  - Merge the stream processor and database.
  - Eliminate the need for an intermediary topic.
  - The stream processor's state store and the database become one.

- **Benefits**:
  - Simplifies architecture.
  - Enables both **push** and **pull** queries within a single system.
  - A unified SQL engine supports both streaming and querying.

---

## Push and Pull Queries

- **Push Queries**:
  - Run asynchronously in the background.
  - Similar to continuous stream processing.

- **Pull Queries**:
  - Executed on demand by end-users.
  - Traditional database queries.

- **Implication**:
  - A streaming database must support both query types using a unified SQL engine.

---

## Column-Based Streaming Databases

![Clickstream with Column-Based Streaming Database](clickstream_column_based.png)

- **Use Case**: Clickstream analytics.
- **Architecture**:
  - Streaming database serves as both the stream processor and the RTOLAP data store.
  - Users can author both push and pull queries using SQL in one place.

### Example: Creating a Materialized View (Push Query)

```sql
CREATE MATERIALIZED VIEW CUSTOMER_CLICKS AS
SELECT *
FROM CLICK_EVENTS E
JOIN CUSTOMERS C ON C.ip = E.ip;
```

- **Explanation**:
  - Continuously updates as new click events and customer data arrive.
  - Stores the result in the streaming database.

### Example: Application Pull Query

```sql
SELECT *
FROM CLICK_EVENTS E
JOIN CUSTOMERS C ON C.ip = E.ip
WHERE C.id = '123';
```

- **Explanation**:
  - Retrieves enriched click events for a specific customer.
  - Executes with low latency due to precomputed materialized view.

---

## Row-Based Streaming Databases

![Row-Based Streaming Database](row_based_streaming_database.png)

- **Characteristics**:
  - Optimized for transactional workloads (CRUD operations).
  - Not ideal for analytical queries due to performance issues.

- **Use Case**:
  - Implementing **Command Query Responsibility Segregation (CQRS)**.

### Command Query Responsibility Segregation (CQRS)

- **Definition**:
  - Architectural pattern separating read (query) and write (command) operations.
  - Allows optimization of read and write models independently.

- **Implementation with Streaming Database**:

  ![CQRS with Streaming Database](cqrs_streaming_database.png)

  - **Write Side**: OLTP database handles transactional writes.
  - **Read Side**: Streaming database provides read-only access for queries.
  - **Data Flow**:
    - Changes in the OLTP database are captured via CDC and streamed to the streaming database.
    - The streaming database maintains a materialized view for fast read access.

---

## Edge Streaming Databases

- **Definition**:
  - Databases positioned closer to the application layer (web edge).
  - Handle data processing at the edge for faster response times.

- **Example**:
  - Using a streaming database directly within a microservice to process real-time data.

- **Benefit**:
  - Reduces latency by processing data closer to where it's consumed.

---

## Challenges in Merging SQL Engines

### SQL Expressivity

- **Definition**:
  - The ability of the SQL engine to represent complex data manipulations succinctly.

- **Challenges in Merging SQL Engines**:

  1. **Performance Mismatch**:
     - Stream processors optimize for real-time data.
     - OLAP databases optimize for complex analytical queries.
     - Merging may lead to suboptimal performance for one or both workloads.

  2. **Latency**:
     - Stream processing requires low latency.
     - OLAP queries may prioritize throughput over latency.

  3. **Resource Allocation**:
     - Conflicting resource demands between streaming and analytical workloads.

  4. **Data Modeling Differences**:
     - Stream processors handle raw or semi-structured data.
     - OLAP databases require structured, preprocessed data.

  5. **Data Consistency**:
     - Ensuring consistency between data in motion (streams) and data at rest (stored data).

  6. **Complexity**:
     - Increased system complexity can impact maintainability and stability.

  7. **Data Volume and Retention**:
     - Streams may have high data volume with short retention.
     - OLAP databases store historical data over longer periods.

  8. **Query Optimizations**:
     - OLAP databases offer advanced optimization techniques not present in stream processors.

  9. **Schema Evolution**:
     - Stream processors may handle evolving schemas more flexibly.

  10. **Maintenance and Updates**:
      - Managing a combined SQL engine increases operational complexity.

### Merging OLTP and Stream Processor SQL Engines

- **Easier Integration** due to:

  - **Data Format Alignment**:
    - Both use row-based storage models.

  - **Real-Time Nature**:
    - Both handle real-time data to some extent.

  - **Transaction Handling**:
    - Shared focus on data consistency and updates.

  - **Event-Driven Architecture**:
    - Stream processors' event-driven nature aligns with real-time updates in OLTP.

- **Challenges Remain**:
  - Despite easier integration, careful architectural planning is necessary.

---

## Benefits of Streaming Databases

- **Reduced Complexity**:
  - Fewer systems to manage and integrate.
  - Simplified data pipelines.

- **Unified SQL Engine**:
  - Easier development and debugging.
  - Consistent SQL semantics across streaming and querying.

- **Improved Performance**:
  - Lower latency due to consolidation.
  - Efficient resource utilization.

- **Flexibility**:
  - Support for both push and pull queries.
  - Ability to handle both data in motion and data at rest.

---

## Mathematical Explanation

### Materialized Views in Streaming Databases

- **Definition**: A continuously updated result of a query stored for fast access.
- **Mathematical Representation**:

  - **Let**:
    - \( D \) = Set of all data records.
    - \( Q \) = Query applied to data.
    - \( V = Q(D) \) = Materialized view.

- **Incremental Updates**:

  - When a new data record \( d \) arrives:
    - **Update Materialized View**:
      \[
      V_{\text{new}} = V_{\text{old}} \oplus Q(d)
      \]
      Where \( \oplus \) represents the incremental update operation.

- **Challenge**:
  - Efficiently computing \( V_{\text{new}} \) without reprocessing entire data \( D \).

### Stream Processing vs. Batch Processing

- **Stream Processing**:
  - Data processed as it arrives.
  - **Latency (\( L_s \))**: Low latency per record.
  - **Throughput (\( T_s \))**: High, depends on event rate.

- **Batch Processing**:
  - Data processed in large chunks.
  - **Latency (\( L_b \))**: Higher, depends on batch size.
  - **Throughput (\( T_b \))**: Potentially higher for aggregate operations.

- **Goal of Streaming Databases**:
  - Achieve low \( L_s \) while handling complex queries traditionally suited for batch processing.

---

## Code Examples

### Creating a Materialized View in a Streaming Database

**SQL Query**:

```sql
CREATE MATERIALIZED VIEW enriched_clicks AS
SELECT
  e.event_time,
  e.click_id,
  c.customer_id,
  c.customer_name,
  p.product_id,
  p.product_name
FROM click_events e
JOIN customers c ON e.customer_id = c.customer_id
JOIN products p ON e.product_id = p.product_id;
```

- **Explanation**:
  - Continuously joins `click_events`, `customers`, and `products`.
  - Updates the materialized view as new data arrives.

### Pull Query Example

**SQL Query**:

```sql
SELECT *
FROM enriched_clicks
WHERE customer_id = '123'
  AND event_time >= NOW() - INTERVAL '1' DAY;
```

- **Explanation**:
  - Retrieves enriched click events for customer '123' from the last day.
  - Benefits from the precomputed materialized view for fast response.

---

## Best Practices

1. **Understand Workload Characteristics**:
   - Determine if your use case benefits more from a row-based or column-based streaming database.

2. **Unified Schema Management**:
   - Maintain consistent schemas across streaming and stored data to simplify integration.

3. **Resource Planning**:
   - Allocate resources considering the combined workloads of stream processing and querying.

4. **Optimize SQL Queries**:
   - Leverage the capabilities of the unified SQL engine for both streaming transformations and queries.

5. **Monitor Performance**:
   - Continuously monitor system performance to identify bottlenecks introduced by merging systems.

6. **Plan for Schema Evolution**:
   - Implement strategies to handle evolving data schemas gracefully.

---

## Conclusion

Streaming databases represent a significant step forward in simplifying data architectures by unifying stream processing and database capabilities. By consolidating components and providing a unified SQL engine, they reduce complexity and improve developer productivity. However, careful planning and consideration are necessary to navigate the challenges associated with merging different system paradigms.

---

## Additional Notes

### Edge Streaming Databases

- **Emerging Trend**:
  - Databases like **EdgeDB** aim to process data closer to where it's generated and consumed.
  - **Benefit**:
    - Reduces latency and bandwidth usage.

### SQL Engine Considerations

- **Expressivity vs. Performance**:
  - A more expressive SQL engine may introduce performance overhead.
  - Balance expressivity with the performance requirements of your application.

- **Advanced Features**:
  - Some streaming databases may not support all SQL features (e.g., complex window functions).
  - Evaluate the SQL capabilities of the streaming database against your needs.

---

## References

- **Designing Data-Intensive Applications** by Martin Kleppmann.
- **Apache Kafka Documentation**: [Kafka Streams](https://kafka.apache.org/documentation/streams/)
- **ksqlDB Documentation**: [ksqlDB](https://ksqldb.io/)
- **Materialize Documentation**: [Materialize](https://materialize.com/docs/)
- **Command Query Responsibility Segregation**: [Microsoft Docs](https://docs.microsoft.com/en-us/azure/architecture/patterns/cqrs)

---

## Tags

#StreamingDatabases #MaterializedViews #StreamProcessing #Databases #SQL #CQRS #DataEngineering #RealTimeAnalytics #StaffPlusNotes #DistributedSystems

---

## Footnotes

1. **Complexity of "Turning the Database Inside Out"**:
   - Requires operating multiple systems (streaming platform, stream processor, external database).
   - Increases operational overhead and demands specialized expertise.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.