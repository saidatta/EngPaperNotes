## Overview

Stream processors are software platforms or tools designed to process continuous streams of data in real-time. A key feature of these processors is their ability to perform **stateful transformations**, thanks to built-in state stores. This capability is essential for consumers to derive analytical insights from event data.

---

## Popular Stream Processors

### Apache Kafka Streams

- **Type**: Stream processing library for JVM-based languages.
- **Integration**: Part of the Apache Kafka project.
- **Features**:
  - Allows building real-time applications and microservices.
  - Consumes, processes, and produces data streams to and from Kafka.
  - Supports **materialized views** via `KTable` and `GlobalKTable`.

### Apache Flink

- **Type**: Standalone stream processor.
- **Processing**: Supports both batch and stream processing.
- **Connectors**: Integrates with Kafka, Pulsar, Kinesis, MongoDB, Elasticsearch, etc.
- **Philosophy**: Treats batch processing as a special case of streaming (bounded data).
- **Deployment**: Runs on its own cluster.

### Spark Structured Streaming

- **Type**: Component of Apache Spark for stream processing.
- **Connectors**: Supports various connectors.
- **Processing Model**: Uses **micro-batching** rather than native streaming.
- **Philosophy**: Views streaming as a special case of batch processing.
- **Deployment**: Cluster-based.

### Apache Samza

- **Type**: Stream processor developed by LinkedIn.
- **Connectors**: Supports Kafka, Azure Event Hubs, Kinesis, HDFS.
- **Deployment**: Cluster-based.
- **Features**:
  - Provides **materialized views** via `Table`.

### Apache Beam

- **Type**: Unified programming model and SDKs.
- **Purpose**: Abstracts data processing pipelines.
- **Runners**: Can execute on Flink, Spark, Samza, Google Cloud Dataflow, etc.

---

## Newer Stream Processors

### Quix Streams

- **Language**: C# and Python.
- **Type**: Stream processing library.
- **Comparison**: Similar to Kafka Streams but for non-JVM languages.
- **Features**:
  - Supports **streaming data frames** akin to Pandas or Spark DataFrames.
  - Data frames are incrementally updated under the hood.

### Bytewax

- **Language**: Python.
- **Type**: Stream processing library.
- **Engine**: Built on Timely Dataflow.
- **Comparison**: Similar to Kafka Streams.

### Pathway

- **Language**: Python.
- **Type**: Stream processing library.
- **Engine**: Based on Differential Dataflow.
- **Features**:
  - Supports **materialized views** via `Table`.

### Estuary Flow

- **Type**: Stream processor.
- **Features**:
  - Supports a wide variety of connectors.
  - Cluster-based deployment.

---

## Materialized Views in Stream Processors

### Importance

Materialized views are essential for implementing real-time analytics use cases. They allow stream processors to represent data in a database-like structure, enabling efficient querying and aggregation.

### Support in Stream Processors

- **Supported**:
  - **Kafka Streams**: Uses `KTable` and `GlobalKTable`.
  - **Samza**: Provides `Table`.
  - **Pathway**: Offers `Table`.
- **Not Directly Supported**:
  - **Apache Flink**: Does not natively support materialized views.
  - **Apache Spark**: Does not have built-in materialized views but offers alternatives.

---

## Emulating Materialized Views in Apache Spark

While Apache Spark doesn't natively support materialized views, you can achieve similar functionality through:

### Caching Mechanism

- **Method**: Cache intermediate or final computation results.
- **Benefits**:
  - Speeds up subsequent queries.
  - Data can be cached in memory or on disk.

### Reusable Views with DataFrames/Datasets

- **Method**: Define reusable transformations.
- **Benefits**:
  - Can be saved and reused across multiple applications.
  - Provides data abstraction and potential optimization.

### Integration with External Data Stores

- **Method**: Use systems like Apache Hive, HBase, or other databases that support materialized views.
- **Benefits**:
  - Leverages the strengths of specialized databases.
  - Spark handles data processing while the external store manages views.

---

## Two Types of Streams

Understanding the types of streams is crucial for effective stream processing:

### 1. Append-Only Streams

- **Characteristics**:
  - Contains discrete, distinct events.
  - Events are only added (appended) and not modified or deleted.
- **Examples**:
  - Clickstream data: Each click is a unique event.
  - Sensor readings: Each measurement is a new event.
- **Usage**:
  - Ideal for events that are inherently unique and timestamped.
- **Data Flow**:

  ![Append-Only Streams](append_only_streams.png)

### 2. Change Streams

- **Characteristics**:
  - Represents changes (inserts, updates, deletes) to records.
  - Often sourced from Change Data Capture (CDC) mechanisms.
- **Examples**:
  - Updates to customer profiles.
  - Changes in product information.
- **Usage**:
  - Captures the evolving state of data entities.
- **Data Flow**:

  ![Change Streams](change_streams.png)

---

## Ingesting Data from Kafka into Stream Processors

### Creating Source Tables

#### Append-Only Stream Example

```sql
-- Ingest clickstream data as an append-only stream
CREATE SOURCE TABLE click_events (
  id INTEGER,
  ts TIMESTAMP,       -- Event timestamp
  url VARCHAR,        -- Contains product ID to be parsed
  ip_address VARCHAR, -- IP address identifying the customer
  session_id VARCHAR,
  referrer VARCHAR,
  browser VARCHAR
) WITH (
  connector = 'kafka',
  topic = 'clicks',
  bootstrap_servers = 'kafka:9092',
  scan_startup_mode = 'earliest'
) FORMAT JSON;
```

#### Change Stream (CDC) Example for Products

```sql
-- Ingest product data from Debezium CDC stream
CREATE SOURCE TABLE products (
  before ROW<id BIGINT, name VARCHAR, color VARCHAR, barcode BIGINT>,
  after ROW<id BIGINT, name VARCHAR, color VARCHAR, barcode BIGINT>,
  op VARCHAR,       -- Operation type: 'c' for create, 'u' for update, 'd' for delete
  source ROW<...>   -- Metadata about the source
) WITH (
  connector = 'kafka',
  topic = 'products',
  bootstrap_servers = 'kafka:9092',
  scan_startup_mode = 'earliest'
) FORMAT JSON;
```

---

## Materialized Views

### Concept

- **Definition**: A precomputed, stored view that contains the results of a query.
- **Purpose**:
  - Improves query performance.
  - Maintains the latest state of data for fast access.
- **In Stream Processing**:
  - Continuously updated as new data arrives.
  - Essential for joining change streams with append-only streams.

### Creating Materialized Views for CDC Data

#### Products Materialized View

```sql
-- Create a materialized view for the latest product data
CREATE MATERIALIZED VIEW latest_products AS
SELECT after.id AS id, after.name AS name, after.color AS color, after.barcode AS barcode
FROM products
WHERE op IN ('c', 'u');  -- Include only create and update operations
```

#### Customers Materialized View

```sql
-- Create a materialized view for the latest customer data
CREATE MATERIALIZED VIEW latest_customers AS
SELECT after.id AS id, after.name AS name, after.email AS email, after.ip_address AS ip_address
FROM customers
WHERE op IN ('c', 'u');  -- Include only create and update operations
```

---

## Joining Streams for Data Enrichment

### Purpose

- **Goal**: Enrich click events with the latest product and customer information.
- **Challenges**:
  - Dealing with out-of-order events.
  - Ensuring the latest state is used in joins.

### Performing the Join

```sql
-- Enrich click events by joining with latest product and customer data
CREATE SINK enriched_clicks AS
SELECT
  c.id AS click_id,
  c.ts AS timestamp,
  c.url,
  c.ip_address,
  cust.name AS customer_name,
  cust.email AS customer_email,
  prod.name AS product_name,
  prod.color AS product_color,
  prod.barcode AS product_barcode
FROM click_events c
JOIN latest_customers cust ON c.ip_address = cust.ip_address
JOIN latest_products prod ON c.product_id = prod.id
WITH (
  connector = 'kafka',
  topic = 'enriched_clicks',
  bootstrap_servers = 'kafka:9092',
  format = 'json'
);
```

- **Notes**:
  - Extract `product_id` from the `url` in `click_events`.
  - Ensure that only the latest records are used from `latest_products` and `latest_customers`.

---

## Handling Duplicate Records

### Issue

- CDC streams can contain multiple records for the same ID due to updates.
- Joining without reduction can lead to duplicates.

### Solution

- Use materialized views to **reduce** the change stream to only the latest record per ID.
- This ensures that joins are performed with the most recent data.

### Example Reduction

Suppose the `products` table has the following records:

| ID | Name    | Color       | Barcode |
|----|---------|-------------|---------|
| 1  | T-shirt | Green       | 123456  |
| 1  | T-shirt | Greenish    | 123456  |
| 1  | T-shirt | Lime Green  | 123456  |

After creating a materialized view, only the latest record is retained:

| ID | Name    | Color      | Barcode |
|----|---------|------------|---------|
| 1  | T-shirt | Lime Green | 123456  |

---

## Advantages of Early Transformation

- **Efficiency**: Reduces processing load on OLAP systems.
- **Timeliness**: Provides real-time enriched data to consumers.
- **Resource Optimization**: Frees up OLAP resources for analytical queries rather than data transformation tasks.

---

## Push vs. Pull Queries

- **Push Queries**:
  - Continuously push updates as data changes.
  - Suitable for streaming data and real-time dashboards.
- **Pull Queries**:
  - Retrieve data on-demand.
  - Used for ad-hoc queries and historical data analysis.

---

## Summary

- **Stateful Transformations**: Essential for complex event processing in stream processors.
- **Materialized Views**: Enable efficient querying and joining of streaming data.
- **Stream Processors**: Must support state management and materialized views for advanced analytics.
- **ETL vs. ELT**:
  - **ETL**: Extract, Transform, Load; suits streaming by transforming data before loading into the OLAP.
  - **ELT**: Extract, Load, Transform; less suitable for streaming unless the target system is a stream processor.
- **Data Enrichment**: Combining append-only streams with change streams provides valuable context for analytics.

---

## Mathematical Concepts

### Stream Joins

- **Equi-Join Condition**:
  \[
  \text{Join Condition: } A \Join B \text{ where } A.\text{key} = B.\text{key}
  \]

- **Temporal Join**:
  - Considers event time to handle late or out-of-order data.
  - Uses windowing to manage state size and processing time.

### Windowing Functions

- **Tumbling Windows**: Non-overlapping, fixed-size intervals.
- **Sliding Windows**: Overlapping windows that slide over time.
- **Session Windows**: Dynamic windows based on activity periods.

---

## Code Examples

### Windowed Aggregation in Apache Flink

```java
DataStream<Tuple2<String, Integer>> aggregatedStream = inputStream
    .keyBy(event -> event.getKey())
    .timeWindow(Time.minutes(5))
    .sum("value");
```

### Stateful Map Function in Kafka Streams

```java
KStream<String, Long> statefulStream = inputStream
    .groupByKey()
    .aggregate(
        () -> 0L,
        (key, newValue, aggValue) -> aggValue + newValue,
        Materialized.with(Serdes.String(), Serdes.Long())
    )
    .toStream();
```

---

## Best Practices

- **Early Transformation**: Perform data transformations as early as possible in the pipeline.
- **Use Materialized Views**: To maintain the latest state and enable efficient joins.
- **Optimize State Stores**: Ensure they are appropriately sized to hold necessary state without impacting performance.
- **Separate Concerns**:
  - Keep transformation logic separate from analytical querying.
  - Use stream processors for data enrichment and OLAP systems for serving queries.

---

## Further Reading

- **Apache Kafka Streams Documentation**: [Kafka Streams](https://kafka.apache.org/documentation/streams/)
- **Apache Flink Documentation**: [Flink Documentation](https://nightlies.apache.org/flink/flink-docs-release-1.15/)
- **Streaming 101: An Introduction to Stream Processing**: Explains the fundamentals of stream processing.
- **Materialized Views in Stream Processing**: Discusses the role and implementation of materialized views.

---

## Conclusion

Stream processors are integral to modern data pipelines, enabling real-time data processing and analytics. By leveraging stateful transformations and materialized views, organizations can enrich data streams and provide immediate insights. Understanding the capabilities and limitations of different stream processors is crucial for designing efficient and scalable data architectures.

---