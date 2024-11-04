## Overview

Materialized views are a critical concept in understanding streaming databases and real-time data processing systems. They enable efficient query performance by precomputing and storing the results of complex queries, which can then be incrementally updated as new data arrives.

- **First Introduced**: Early 1990s.
- **Purpose**: Improve query performance by avoiding repeated execution of expensive queries.
- **In Stream Processing**: Materialized views are continuously and incrementally updated in the background, reflecting real-time changes.

---

## Traditional Views vs. Materialized Views

### Traditional Views

- **Definition**: A virtual table defined by a SQL query.
- **Characteristics**:
  - Results are **not stored**; computed on-the-fly when queried.
  - Increases latency because the query executes every time.
- **Analogy**: A smart chipmunk named **Simon** who counts nuts every time you ask.

#### Example:

- **Scenario**: You ask Simon, "How many nuts are in the yard?"
- **Process**:
  - Simon runs out, counts the nuts, returns with the count.
  - Each time you ask, he repeats the process.
- **Mathematical Representation**:

  \[
  \text{Count} = \sum_{i=1}^{N} 1
  \]

  Where \( N \) is the total number of nuts in the yard.

### Materialized Views

- **Definition**: A physical table storing the results of a query.
- **Characteristics**:
  - Results are **precomputed and stored**.
  - Updated asynchronously as underlying data changes.
- **Analogy**: Simon writes the count on a piece of paper and stores it in a box. Another chipmunk, **Alvin**, reads the count from the box when asked.

#### Example:

- **Scenario**: Simon monitors changes (incremental updates) and updates the count in the box.
- **Process**:
  - Alvin retrieves the count from the box without recomputing.
- **Mathematical Representation**:

  \[
  X_{\text{new}} = X_{\text{current}} + \Delta X
  \]

  Where:
  - \( X_{\text{current}} \) is the current count.
  - \( \Delta X \) is the incremental change (nuts added or removed).

---

## Incremental Updates

### Concept

- **Incremental Changes**: Small, targeted updates to data without recomputing from scratch.
- **Benefits**:
  - Reduces computational overhead.
  - Keeps data consistent and up-to-date in real-time.

### Mathematical Representation

- **State Update Function**:

  \[
  X_{t+1} = X_t + \Delta X_t
  \]

  Where:
  - \( X_t \) is the state at time \( t \).
  - \( \Delta X_t \) is the change observed at time \( t \).

### In the Chipmunk Analogy

- **Simon** continuously observes the yard for changes (\( \Delta X \)) and updates the stored count \( X \) accordingly.

---

## Change Data Capture (CDC) and Materialized Views

### Change Data Capture (CDC)

- **Definition**: Technique to capture and track changes (inserts, updates, deletes) in a database over time.
- **Method**: Reads the **Write-Ahead Log (WAL)** of an OLTP database.
- **Purpose**: Allows downstream systems to react to data changes in real-time.

### Relationship with Materialized Views

- **Materialized Views**:
  - Consume CDC streams to maintain up-to-date query results.
  - Use incremental changes to update views without full recomputation.

### Example Scenario

- **Nuts with Attributes**:
  - **Attributes**: Color, Location (latitude, longitude).
  - Nuts can change color or location.
- **Process**:
  - Simon captures changes to nuts (CDC events).
  - Updates the materialized view (the list of nuts in the box).
  - Alvin serves the latest list to clients.

### Diagram Explanation

![Replication using Incremental Changes](replication_incremental_changes.png)

- **Components**:
  1. **OLTP Database**: Source of truth with WAL capturing changes.
  2. **CDC Connector**: Reads WAL and writes changes to a streaming platform.
  3. **Streaming Platform**: Publishes changes to topics.
  4. **Stream Processor**: Consumes topics, maintains materialized views.
  5. **Materialized Views**: Represent the latest state of data.
  6. **Clients**: Query materialized views for up-to-date information.

---

## Push vs. Pull Queries

### Definitions

- **Push Queries**:
  - **Asynchronous**: Run in the background.
  - **Data Flow**: Data is pushed to the client when changes occur.
  - **Analogy**: Simon (smart chipmunk) updates the count proactively.
- **Pull Queries**:
  - **Synchronous**: Executed upon request.
  - **Data Flow**: Client requests data and waits for the result.
  - **Analogy**: Alvin (less smart chipmunk) retrieves the count from the box when asked.

### Trade-offs

- **Latency**:
  - **Push Queries**: Lower latency as data is updated in real-time.
  - **Pull Queries**: May have higher latency due to on-demand computation.
- **Flexibility**:
  - **Push Queries**: Less flexible; predefined computations.
  - **Pull Queries**: More flexible; clients can specify queries ad-hoc.

### Balancing Latency and Flexibility

![Latency vs Flexibility](latency_vs_flexibility.png)

- **Low Latency Applications**:
  - Benefit from push queries.
  - Examples: Real-time dashboards, monitoring systems.
- **High Flexibility Applications**:
  - Require pull queries for ad-hoc analysis.
  - Examples: Data exploration, complex analytics.

### Combining Push and Pull Queries

- **Goal**: Achieve both low latency and high flexibility.
- **Approach**:
  1. Client submits a **push query** to create a materialized view.
  2. Client subscribes to changes in the materialized view.
- **Benefits**:
  - Single SQL query for real-time updates.
  - Reduced latency as incremental changes are pushed.
  - Flexibility to define custom queries.

---

## Limitations of Current Systems

- **Separation of Concerns**:
  - Push and pull queries are executed in **separate systems** (stream processor vs. OLAP).
  - Authored by **different teams** (data engineers vs. analysts).
- **Challenges**:
  - **Complexity**: Managing and coordinating multiple systems.
  - **Redundancy**: Duplicate efforts in data processing.
  - **Latency**: Additional overhead in data movement between systems.

---

## Streaming Databases

### Characteristics

- **Unified System**: Combines stream processing and database capabilities.
- **Features**:
  - **Materialized Views**: Continuously updated with real-time data.
  - **Exposure of Views**: Ability to expose changes to topics (like a WAL).
  - **Optimal Storage**: Efficient data storage for serving queries.
  - **Serving Methods**: Support for both synchronous (pull) and asynchronous (push) queries.
- **Benefits**:
  - Simplifies architecture by consolidating systems.
  - Reduces latency and complexity.
  - Enhances flexibility and performance.

---

## Common Solutions for Real-Time Analytics

### Traditional Architecture

![Common Solution for Real-Time Analytics](common_real_time_analytics.png)

- **Components**:
  1. **OLTP Database**: Operational data store.
  2. **CDC Connector**: Captures changes and writes to streaming platform.
  3. **Streaming Platform**: Distributes data streams.
  4. **Stream Processor**: Processes data, creates materialized views.
  5. **RTOLAP Data Store**: Stores optimized data for analytical queries.
  6. **Clients**: Pull data from RTOLAP for analysis.

### Data Flow Steps

1. **Data Generation**: Applications write to OLTP database.
2. **CDC**: Changes captured and sent to streaming platform.
3. **Stream Processing**: Materialized views created and updated.
4. **Data Ingestion**: RTOLAP consumes processed data.
5. **Query Serving**: Clients execute pull queries against RTOLAP.

### Challenges

- **Separate Systems**: Push queries (stream processor) and pull queries (RTOLAP) are decoupled.
- **Optimization**: End-users have limited ability to influence data processing logic.
- **Coordination**: Requires collaboration between data engineers and analysts.

---

## Upsert Operations in CDC Streams

### Definition

- **Upsert**: Combination of "update" and "insert".
  - **Logic**: If a record exists, update it; otherwise, insert it.
- **Purpose**:
  - Simplify handling of CDC streams containing inserts, updates, and deletes.
  - Maintain accurate and up-to-date replicas of source tables.

### Handling Deletes

- **Deletes**: More complex to handle.
  - Some systems flag records as deleted.
  - Others remove records entirely.
- **Considerations**:
  - Consistency between source and replica.
  - Impact on query results.

### Implementing Upsert Logic

- **Options**:
  1. **In RTOLAP System**:
     - RTOLAP reads CDC topics directly.
     - Applies upsert logic internally.
  2. **In Stream Processor**:
     - Stream processor handles upserts.
     - Outputs data to RTOLAP-ready format.

### Example Workflow

1. **Data Change**: Application updates a product's inventory.
2. **WAL Update**: Change recorded in OLTP database's WAL.
3. **CDC Snapshot**:
   - Initial snapshot taken to seed downstream replicas.
   - Necessary to rebuild table state from incremental changes.
4. **Stream Processing**:
   - Materialized view of the Products table created.
   - Upsert logic applied to maintain accurate state.
5. **Data Output**:
   - Stream processor writes to output topic or directly to RTOLAP.
6. **RTOLAP Ingestion**:
   - RTOLAP consumes data.
   - Applies additional transformations or indexing as needed.
7. **Client Queries**:
   - End-users query RTOLAP for analytical insights.

### Challenges

- **Complexity**: Multiple systems handling similar logic.
- **Data Consistency**: Ensuring replicas accurately reflect source data.
- **Latency**: Additional processing steps can increase end-to-end latency.

---

## Code Examples

### Upsert Implementation in Apache Flink

#### Schema Definition

```java
public class Product {
    public Long id;
    public String name;
    public String color;
    public Long barcode;
    // Constructors, getters, setters
}
```

#### Stream Processing with Upsert Logic

```java
DataStream<Product> cdcStream = env.addSource(new FlinkKafkaConsumer<>(
    "products_cdc_topic",
    new ProductDeserializationSchema(),
    properties
));

KeyedStream<Product, Long> keyedStream = cdcStream.keyBy(product -> product.id);

DataStream<Product> upsertedStream = keyedStream.process(new KeyedProcessFunction<Long, Product, Product>() {
    private ValueState<Product> state;

    @Override
    public void open(Configuration parameters) {
        ValueStateDescriptor<Product> descriptor = new ValueStateDescriptor<>(
            "productState",
            Product.class
        );
        state = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(Product value, Context ctx, Collector<Product> out) throws Exception {
        // Apply upsert logic
        state.update(value);
        out.collect(value);
    }
});
```

---

## Mathematical Concepts

### Incremental State Updates

- **State Transition Function**:

  \[
  S_{t+1} = S_t + \Delta S_t
  \]

  Where:
  - \( S_t \) is the state at time \( t \).
  - \( \Delta S_t \) is the incremental change at time \( t \).

- **Aggregate Functions**:

  - **Count**:

    \[
    \text{Count}_{t+1} = \text{Count}_t + \Delta \text{Count}_t
    \]

  - **Sum**:

    \[
    \text{Sum}_{t+1} = \text{Sum}_t + \Delta \text{Sum}_t
    \]

  - **Average**:

    \[
    \text{Average}_{t+1} = \frac{\text{Sum}_{t+1}}{\text{Count}_{t+1}}
    \]

---

## Best Practices

1. **Consolidate Systems**:
   - Use streaming databases to unify stream processing and data storage.
2. **Leverage Materialized Views**:
   - Utilize materialized views for low-latency query serving.
3. **Implement Upsert Logic Appropriately**:
   - Decide where upsert logic should reside based on system capabilities.
4. **Optimize Data Flow**:
   - Minimize data movement between systems to reduce latency.
5. **Collaborate Across Teams**:
   - Encourage collaboration between data engineers and analysts.

---

## Conclusion

Materialized views play a pivotal role in real-time data processing and analytics. By understanding the trade-offs between push and pull queries and the complexities of handling CDC data, we can design systems that provide both low latency and high flexibility. Streaming databases offer a promising solution by consolidating stream processing and database functionalities, simplifying architectures, and improving performance.

---

## Further Reading

- **"Turning the Database Inside-Out" by Martin Kleppmann**: [YouTube Video](https://www.youtube.com/watch?v=fU9hR3kiOK0)
- **Apache Flink Documentation**: [Stateful Stream Processing](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/dev/datastream/state/)
- **Change Data Capture (CDC)**: [Wikipedia](https://en.wikipedia.org/wiki/Change_data_capture)
- **Materialized Views in Databases**: [Database Systems Concepts](https://www.db-book.com/)

---

## Footnotes

1. **Upsert**: The term combines "update" and "insert" operations. It's used when you want to insert a new record or update an existing one based on a primary key.
2. **Handling Deletes**: Deletes are more complex in CDC streams. Some systems handle them explicitly, while others may ignore or flag them for potential recovery.

---

# Tags

- #StreamingDatabases
- #MaterializedViews
- #RealTimeAnalytics
- #ChangeDataCapture
- #StreamProcessing
- #PushVsPullQueries
- #UpsertOperations
- #DataEngineering
- #DistributedSystems
- #StaffPlusNotes

---

This comprehensive overview should provide you with a solid understanding of materialized views in the context of streaming databases and real-time data processing systems. The included code examples, mathematical representations, and detailed explanations aim to enhance your grasp of the concepts and their practical applications.