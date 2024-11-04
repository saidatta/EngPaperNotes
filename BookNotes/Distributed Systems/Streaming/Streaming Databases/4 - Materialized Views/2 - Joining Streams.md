# 

## Overview

In modern data processing systems, **joining streams** is a crucial operation, especially when dealing with real-time data. This involves combining data from different sources to enrich or augment the data streams for further analysis.

- **Transformation operations** are performed on or between **tabular constructs** that hold:
  - **Change Streams** (Materialized Views)
  - **Append-Only Streams**

Understanding how these constructs interact and how to effectively join them is essential for building efficient streaming applications.

---

## Key Concepts

### Append-Only Streams

- **Definition**: Streams where only **insertions** are allowed; no updates or deletions.
- **Characteristics**:
  - Ever-growing data.
  - Represented as **append tables** in stream processors.
  - Not backed by a **state store**.
  - Example: Click events from a website.

### Change Streams

- **Definition**: Streams that include **inserts**, **updates**, and **deletes**; often sourced from Change Data Capture (CDC) systems.
- **Characteristics**:
  - Represent changes to data over time.
  - Represented as **change tables** (materialized views) in stream processors.
  - Backed by a **state store**.
  - Example: Updates to customer profiles.

---

### Tabular Constructs in Stream Processors

#### Append Tables

- Hold **append-only streams**.
- Do **not** rely on state stores.
- Represent data that **passes through** the stream processor.

#### Change Tables

- Represent **materialized views**.
- Backed by a **state store**.
- Store the **latest state** of data based on changes.

---

### Topics in Streaming Platforms

#### Append Topics

- Contain **append-only data**.
- Correspond to append-only streams.

#### Change Topics

- Contain **change events** or **CDC events**.
- Correspond to change streams.
- Sometimes called **table topics** by Kafka engineers.

---

## Apache Calcite

### What is Apache Calcite?

- **Definition**: A dynamic data management framework that provides advanced SQL processing and optimization capabilities.
- **Purpose**: Allows the creation of custom databases and data processing systems using **relational algebra**.

### Features

- **Relational Algebra Implementation**: Provides a mathematical foundation for SQL operations.
- **Query Optimization**: Offers a cost-based optimizer for efficient query execution plans.
- **Adapter Architecture**: Can connect to various data sources and storage engines.

### Limitations

- **Does Not Include**:
  - Storage of data.
  - Algorithms for processing data.
  - Repository for storing metadata.

> "Calcite intentionally stays out of the business of storing and processing data... this makes it an excellent choice for mediating between applications and one or more data storage locations and data processing engines. It is also a perfect foundation for building a database: just add data."

### Use Cases

- **Building Databases from Scratch**: Acts as a foundational building block.
- **Integration**: Used by systems like Apache Flink, Apache Pinot, Apache Druid, and others.

---

## Joining Streams with SQL

### Example: Joining Table Topics

```sql
CREATE SINK clickstream_enriched AS
SELECT
  E.*,
  C.*,
  P.*
FROM CLICK_EVENTS E
JOIN CUSTOMERS C ON C.ip = E.ip
JOIN PRODUCTS P ON P.product_id = E.product_id
WITH (
  connector = 'kafka',
  topic = 'click_customer_product',
  properties.bootstrap.server = 'kafka:9092',
  type = 'upsert',
  primary_key = 'id'
);
```

#### Explanation:

- **CLICK_EVENTS (E)**:
  - **Type**: Append table.
  - **Source**: Append topic.
  - **Content**: Click events from the application.

- **CUSTOMERS (C)**:
  - **Type**: Change table.
  - **Source**: Change topic.
  - **Content**: Customer data from CDC.

- **PRODUCTS (P)**:
  - **Type**: Change table.
  - **Source**: Change topic.
  - **Content**: Product data from CDC.
  - **Note**: Assumes `product_id` is extracted from the click URL.

---

### Joining Logic

- **Inner Join**: Combines records that have matching values in both tables.
- **Output**: A materialized view resulting from the join operation.
- **Benefits**:
  - Enriches click events with customer and product information.
  - Offloads complex processing from downstream systems.

---

## Properties of Different Types of Table Joins

### Append Table to Append Table

- **Requirement**: **Windowing** is mandatory.
- **Reason**: Without windowing, the state store will eventually run out of space due to unbounded data growth.
- **Use Case**: Joining two streams of events, such as merging logs from two different sources.

### Change Table to Change Table

- **Requirement**: **Windowing** is not required.
- **Reason**: Since change tables represent the latest state, the join result can fit into the state store if appropriately sized.
- **Use Case**: Merging two dimension tables, like customers and addresses.

### Change Table to Append Table

- **Requirement**: **Windowing** is necessary.
- **Reason**: The append table introduces unbounded growth, necessitating windowing to limit state store size.
- **Use Case**: Enriching event streams with the latest state from a dimension table.

---

### Importance of Windowing

- **Definition**: A mechanism to group data within a specific time frame.
- **Purpose**: Limits the amount of data held in the state store.
- **Types of Windows**:
  - **Tumbling Windows**: Fixed-size, non-overlapping.
  - **Sliding Windows**: Fixed-size, overlapping.
  - **Session Windows**: Based on periods of inactivity.

---

## Left Join in Stream Processing

### How Left Joins Work

- **Definition**: Returns all records from the left table and matched records from the right table.
- **In Stream Processing**:
  - The **left stream** drives the result.
  - The join continuously and dynamically updates as new events arrive.

### SQL Syntax

```sql
SELECT ...
FROM append_table_stream
LEFT JOIN change_table_stream ON join_condition;
```

- **append_table_stream**: The primary stream (append table).
- **change_table_stream**: The secondary stream (change table).
- **join_condition**: The condition on which to join the streams.

### Example with Clicks and Customers

```sql
SELECT
  k.product_id,
  c.customer_name
FROM click k
LEFT JOIN customers c ON k.customer_id = c.customer_id;
```

- **Explanation**:
  - For every click event (`k`), retrieve the corresponding customer's name (`c.customer_name`).
  - If there's no matching customer, the customer fields will be `NULL`.

---

### Continuous and Dynamic Joins

- **Real-Time Updates**: As new data arrives in either stream, the join output is updated.
- **State Management**: Requires maintaining state for the duration of the window (if windowed) or indefinitely (if unbounded).

---

## Clickstream Use Case

### Data Flow Diagram (Conceptual Steps)

1. **Customer Update**:
   - A customer updates their information.
   - The update is saved in the OLTP database.

2. **CDC Process**:
   - A CDC connector captures changes from the OLTP database.
   - Changes to the `CUSTOMERS` table are written to a **CDC topic** (change topic).

3. **Customer Clicks**:
   - The customer clicks on a product in the e-commerce application.
   - The click event is written to a **click events topic** (append topic).

4. **Stream Processor**:
   - Reads from both the CDC topic and the click events topic.
   - Stores `CUSTOMERS` data in a **state store** (change table).
   - Processes `CLICK_EVENTS` data as an **append table**.

5. **Join Operation**:
   - Performs a **left join** between `CLICK_EVENTS` and `CUSTOMERS`.
   - Enriches click events with customer information.

6. **Output Topics**:
   - Writes enriched click events to an **output topic**.
   - May also write the latest customer data to another topic (optional/redundant).

7. **Real-Time OLAP Ingestion**:
   - The RTOLAP data store consumes the enriched data from the output topic.
   - Stores data optimized for analytical queries.

8. **User Queries**:
   - Users query the RTOLAP for real-time analytics.
   - Benefit from pre-joined, enriched data for low-latency responses.

---

### Diagram Explanation

While we cannot display images, here's a step-by-step representation:

1. **Customer Updates**:
   - **Event**: Customer updates profile.
   - **Action**: OLTP database records the update.

2. **CDC Capturing**:
   - **Event**: CDC connector captures the update.
   - **Action**: Writes to `CUSTOMERS` change topic.

3. **Click Events**:
   - **Event**: Customer clicks a product.
   - **Action**: Click event written to `CLICK_EVENTS` append topic.

4. **Stream Processing**:
   - **Action**: Stream processor reads both topics.
   - **Stores**:
     - `CUSTOMERS` data in a state store.
     - `CLICK_EVENTS` data as pass-through.

5. **Joining Streams**:
   - **Operation**: Left join on `CLICK_EVENTS` and `CUSTOMERS` based on customer ID.
   - **Result**: Enriched click events with customer data.

6. **Output to Topics**:
   - **Enriched Clicks**: Written to an output topic.
   - **Optional**: Latest `CUSTOMERS` data written to another topic.

7. **RTOLAP Ingestion**:
   - **Action**: RTOLAP consumes enriched data.
   - **Benefit**: Optimized for fast query performance.

8. **User Interaction**:
   - **Action**: Users query RTOLAP.
   - **Experience**: Low-latency responses due to pre-processed data.

---

### Implementation Choices

- **Pre-Join in Stream Processor**:
  - Pros:
    - Offloads processing from RTOLAP.
    - Improves query performance for end-users.
  - Cons:
    - Increases complexity in stream processing logic.

- **Join in RTOLAP at Query Time**:
  - Pros:
    - Simplifies stream processing.
  - Cons:
    - Higher query latency.
    - Increased load on RTOLAP system.

---

## Reducing Complexity with Streaming Databases

### Challenges in Traditional Architecture

- **Multiple Systems**: Managing stream processors, streaming platforms, and RTOLAP databases separately.
- **Data Movement**: Data must be moved and transformed across different systems.
- **Latency**: Additional overhead from data movement and processing delays.
- **Maintenance Overhead**: Requires expertise in multiple technologies.

### Solution: Streaming Databases

- **Definition**: Databases that integrate stream processing capabilities with traditional database features.
- **Benefits**:
  - **Unified Platform**: Combines storage, processing, and querying.
  - **Simplified Architecture**: Reduces the number of systems to manage.
  - **Low Latency**: Directly operates on streaming data.
  - **Materialized Views**: Provides real-time, continuously updated views.

### How Streaming Databases Help

- **In-Memory Processing**: Handle data in real-time without disk I/O bottlenecks.
- **Stateful Stream Processing**: Maintain state across events for operations like joins and aggregations.
- **SQL Support**: Use familiar SQL syntax for both stream processing and querying.
- **Integrated Storage**: Data persists within the same system, reducing data movement.

---

## Summary

> "I'm gonna make a very bold claim [that] all the databases you've seen so far are streaming databases."  
> — **Mihai Budiu**, "Building a Streaming Incremental View Maintenance Engine with Calcite," March 2023

### Key Takeaways

- **Materialized Views Bridge the Gap**:
  - They blur the lines between stream processing and databases.
  - Enable precomputed, persistent summaries of streaming data.
  
- **Advantages of Materialized Views**:
  - **Efficient Real-Time Analytics**: Faster query responses without reprocessing entire datasets.
  - **Seamless Integration**: Combine streaming and batch processing paradigms.
  - **Simplified Architecture**: Reduce complexity by unifying data processing models.

### Constructs in OLTP Databases Related to Streaming

1. **Write-Ahead Log (WAL)**:
   - Captures changes to database tables.
   - Serves as a source for CDC systems.

2. **Materialized Views**:
   - Asynchronous queries that preprocess and store data.
   - Enable low-latency queries by serving precomputed results.

### "Turning the Database Inside Out"

- **Concept by Martin Kleppmann**:
  - **Traditional View**: Databases are central, and streams are derived.
  - **Inside Out**: Streams are primary, and databases are views over the streams.

### Our Journey

- **Externalizing the WAL**:
  - Published changes to a streaming platform (e.g., Kafka).

- **Mimicking Materialized Views in Stream Processing**:
  - Created stateful stream processing platforms that handle complex transformations.
  - Offloaded transformation logic from OLTP databases.

---

## Mathematical Concepts

### Relational Algebra in Stream Processing

- **Selection (\( \sigma \))**:
  - Filters rows based on a predicate.
  - Example: \( \sigma_{customer\_id = 123}(Customers) \)

- **Projection (\( \pi \))**:
  - Selects specific columns.
  - Example: \( \pi_{customer\_name, email}(Customers) \)

- **Join (\( \Join \))**:
  - Combines rows from two tables based on a related column.
  - Example: \( Clicks \Join_{Clicks.customer\_id = Customers.customer\_id} Customers \)

- **Aggregation (\( \gamma \))**:
  - Performs calculations on sets of rows.
  - Example: \( \gamma_{COUNT(click\_id)}(Clicks) \)

### State Store Size Considerations

- **Unbounded Streams**:
  - Without windowing, state stores can grow infinitely.
  - **Memory Usage (\( M \))** over time (\( t \)):
    \[
    M(t) = M(0) + r \times t
    \]
    Where:
    - \( M(0) \): Initial memory usage.
    - \( r \): Rate of incoming events.

- **Windowed Streams**:
  - State store size is limited to the window duration.
  - **Maximum Memory Usage (\( M_{\text{max}} \))**:
    \[
    M_{\text{max}} = r \times W
    \]
    Where:
    - \( W \): Window size (time duration).

---

## Code Examples

### Implementing a Join in Apache Flink SQL

#### Table Definitions

```sql
-- Define the CLICK_EVENTS table (append table)
CREATE TABLE CLICK_EVENTS (
  event_time TIMESTAMP(3),
  customer_id BIGINT,
  product_id BIGINT,
  -- other columns...
  WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
  'connector' = 'kafka',
  'topic' = 'click_events',
  'properties.bootstrap.servers' = 'kafka:9092',
  'format' = 'json'
);

-- Define the CUSTOMERS table (change table)
CREATE TABLE CUSTOMERS (
  customer_id BIGINT,
  customer_name STRING,
  email STRING,
  -- other columns...
  PRIMARY KEY (customer_id) NOT ENFORCED
) WITH (
  'connector' = 'upsert-kafka',
  'topic' = 'customers_cdc',
  'properties.bootstrap.servers' = 'kafka:9092',
  'key.format' = 'json',
  'value.format' = 'json'
);
```

#### Join Query

```sql
INSERT INTO ENRICHED_CLICKS
SELECT
  c.event_time,
  c.customer_id,
  c.product_id,
  cust.customer_name,
  cust.email
FROM CLICK_EVENTS c
LEFT JOIN CUSTOMERS FOR SYSTEM_TIME AS OF c.event_time AS cust
ON c.customer_id = cust.customer_id;
```

#### Explanation

- **CLICK_EVENTS**:
  - Defined with a **watermark** for event time.
  - Consumed as an append-only stream.

- **CUSTOMERS**:
  - Defined with a **primary key**.
  - Consumed as a change-log stream using **upsert-kafka** connector.

- **Join Logic**:
  - Performs a **temporal table join** using `FOR SYSTEM_TIME AS OF`.
  - Ensures that the customer data corresponds to the event time of the click.

---

## Best Practices

1. **Understand Stream Types**:
   - Know whether your streams are append-only or contain changes.
   - Choose appropriate tabular constructs (append tables vs. change tables).

2. **Use Windowing Wisely**:
   - Apply windowing when joining with append-only streams.
   - Prevents unbounded state growth.

3. **Leverage Stream Processing Engines**:
   - Utilize engines like Apache Flink with SQL support.
   - Simplify complex stream transformations.

4. **Optimize State Stores**:
   - Monitor and manage state store sizes.
   - Use key expiration or windowing to control memory usage.

5. **Consider Streaming Databases**:
   - Evaluate if a streaming database fits your use case.
   - Benefits include simplified architecture and lower latency.

---

## Conclusion

- **Materialized Views** are powerful tools that enable efficient real-time analytics by bridging stream processing and databases.
- **Joining Streams** requires careful consideration of stream types and the implications on state management.
- **Streaming Databases** offer a promising solution to reduce complexity and improve performance in real-time data processing systems.

---

# Tags

- #Streaming
- #MaterializedViews
- #StreamProcessing
- #ApacheCalcite
- #ApacheFlink
- #SQL
- #DataEngineering
- #RealTimeAnalytics
- #StaffPlusNotes
- #DistributedSystems

---

# References

- **Apache Calcite Documentation**: [Calcite](https://calcite.apache.org/)
- **Apache Flink SQL Documentation**: [Flink SQL](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/dev/table/sql/overview/)
- **Martin Kleppmann's Talk**: ["Turning the Database Inside Out"](https://www.confluent.io/kafka-summit-lon18/turning-the-database-inside-out-with-apache-samza/)
- **Mihai Budiu's Presentation**: ["Building a Streaming Incremental View Maintenance Engine with Calcite"](https://www.youtube.com/watch?v=VIDEO_LINK)

---

# Footnotes

1. **Windowing in Stream Processing**: A window defines a finite set of data over which calculations are performed. Without windows, operations over unbounded streams could consume infinite memory.

2. **Temporal Table Joins**: In Flink SQL, temporal joins allow you to join a stream with a table while considering the time attribute, ensuring you get the correct version of the data at the event time.

---

# Appendix

## Additional Mathematical Formulas

### Calculating State Store Size with Windowing

- **Assuming**:
  - **Event Rate** (\( \lambda \)): Number of events per second.
  - **Window Size** (\( W \)): Duration of the window in seconds.
  - **Average Event Size** (\( S \)): Size of each event in bytes.

- **Total State Store Size** (\( M \)):

  \[
  M = \lambda \times W \times S
  \]

- **Example**:
  - \( \lambda = 1000 \) events/second
  - \( W = 60 \) seconds
  - \( S = 500 \) bytes

  \[
  M = 1000 \times 60 \times 500 = 30,000,000 \text{ bytes } = 28.6 \text{ MB}
  \]

---

## Sample Data Formats

### Click Events (JSON)

```json
{
  "event_time": "2023-07-20T12:34:56.789Z",
  "customer_id": 12345,
  "product_id": 9876,
  "session_id": "abcde12345",
  "ip_address": "192.168.1.1"
}
```

### Customer Updates (JSON)

```json
{
  "customer_id": 12345,
  "customer_name": "Jane Doe",
  "email": "jane.doe@example.com",
  "phone": "+1234567890",
  "address": "123 Main St"
}
```

---

Remember, when dealing with real-time data and stream processing, always consider the trade-offs between complexity, performance, and resource utilization. Understanding these concepts deeply will empower you to design and implement efficient streaming architectures suitable for high-demand applications.