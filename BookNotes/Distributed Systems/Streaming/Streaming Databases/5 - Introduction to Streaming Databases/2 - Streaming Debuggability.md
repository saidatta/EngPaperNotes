## Overview
In modern data processing architectures, **streaming databases** have emerged as powerful tools that combine the capabilities of stream processors and databases. One of the significant advantages of streaming databases is the **improved debuggability** of data pipelines and materialized views. This document delves into the benefits of debugging in streaming databases, the challenges associated with SQL in streaming contexts, various streaming database implementations, and the architectural considerations when working with streaming databases.

---
## Advantages of Debugging in Streaming Databases

### 1. Familiar SQL Interface

- **Ease of Use**: Streaming databases often provide a SQL-like query language for defining stream processing operations.
- **Benefit**: Data engineers can leverage their existing SQL knowledge, making it easier to write, debug, and maintain streaming queries.

### 2. Simpler Logic

- **High-Level Abstraction**: Streaming databases offer higher-level abstractions that simplify complex stream processing tasks.
- **Benefit**: Simplified logic leads to easier debugging and reduced chances of errors.

### 3. Integrated Ecosystem

- **Unified Platform**: By combining a stream processor and a database into one system, streaming databases provide better integration with data tools and monitoring solutions.
- **Benefit**: Provides a holistic view of the data pipeline, aiding in end-to-end debugging.

### 4. Built-in Optimizations

- **Performance Enhancements**: Streaming databases come with built-in optimizations for common stream processing patterns.
- **Benefit**: Improves performance and reliability, reducing the need for complex debugging in performance-critical scenarios.

### 5. Easier Deployment

- **Simplified Setup**: Designed for ease of deployment, streaming databases reduce potential deployment-related issues.
- **Benefit**: Simplifies the debugging process by minimizing environment-specific problems.

---

## SQL Is Not a Silver Bullet

While SQL provides a powerful and familiar interface for querying data, it has limitations in streaming contexts.

### Limitations

- **Abstraction Level**: SQL is highly abstract, which can sometimes obscure the underlying processing logic.
- **Expressiveness**: Certain complex stream processing tasks may require more expressive capabilities than SQL offers.
- **Debugging Complexity**: Observing the actual logical execution plan derived by the stream processing system can be challenging.

### Mitigations

- **Lower-Level DSLs**: Tools like Kafka Streams (Streams DSL and Processor API) and Flink (DataStream API) offer more expressive DSLs for complex scenarios.
- **User-Defined Functions (UDFs)**: Extend SQL capabilities by writing custom functions, though they may not cover all use cases.
- **Execution Plan Inspection**: Tooling for inspecting execution plans is evolving and can aid in understanding and debugging complex queries.

---

## Streaming Database Implementations

Below is a comparison of some popular streaming databases, their licenses, state store implementations, and typical use cases.

### Table: Existing Streaming Databases

| Name         | License                       | State Store Implementation               | Use Cases                                    |
|--------------|-------------------------------|------------------------------------------|----------------------------------------------|
| **ksqlDB**   | Confluent Community License   | RocksDB (LSM tree key-value storage)     | CQRS, push queries                           |
| **RisingWave** | Apache 2                    | Row-based                                | CQRS, push queries, single row lookups       |
| **Materialize** | Business Source License (BSL) | Row-based                                | CQRS, push queries, single row lookups       |
| **Timeplus (Proton)** | Apache 2             | Column-based                             | Analytical push and pull queries             |

### Notes:

- **ksqlDB**:
  - Uses **RocksDB** for state storage.
  - Supports indexing only by primary keys.
  - Complex queries may require full table scans.

- **RisingWave and Materialize**:
  - Implement more database-like persistence layers.
  - Support flexible indexing schemes.
  - Efficiently serve a wide range of pull queries.

- **Timeplus (Proton)**:
  - Utilizes a columnar storage approach (e.g., based on ClickHouse).
  - Optimized for analytical queries.

---

## Streaming Database Architecture

### Overview

In a streaming database, **materialized views** (change tables) are held in the **state store**. **Append tables** pass through or are placed into a topic-like construct within the streaming database.

### Data Flow Diagram

![Streaming Database Architecture](streaming_database_architecture.png)

*Note: Since images cannot be displayed here, refer to Figure 5-9 in the original text for visual representation.*

### Steps in Data Processing

#### 1. Ingesting Click Events

- **Source**: Append topic in a streaming platform (e.g., Kafka).
- **Action**: Create a **source table** in the streaming database to ingest click events.

##### Example: Creating a Source Table for Click Events

```sql
CREATE SOURCE click_events (
  id INTEGER,
  ts BIGINT,         -- Timestamp
  url VARCHAR,       -- Contains product ID to be parsed out
  ip_address VARCHAR, -- Identifies a customer
  session_id VARCHAR,
  referrer VARCHAR,
  browser VARCHAR
)
WITH (
  connector = 'kafka',
  topic = 'clicks',
  properties.bootstrap.server = 'kafka:9092',
  scan.startup.mode = 'earliest'
)
ROW FORMAT JSON;
```

#### 2. Ingesting Customer CDC Data

- **Source**: Append-only table (initially).
- **Action**: Create a **source table** for customers using the Debezium CDC format.

##### Example: Creating a Source Table for Customers

```sql
CREATE SOURCE customers (
  before ROW<id BIGINT, name VARCHAR, email VARCHAR, ip_address VARCHAR>,
  after ROW<id BIGINT, name VARCHAR, email VARCHAR, ip_address VARCHAR>,
  op VARCHAR,      -- Operation type: insert, update, delete
  ts TIMESTAMP,    -- Timestamp of the change
  source <...>
)
WITH (
  connector = 'kafka',
  topic = 'customers',
  properties.bootstrap.server = 'kafka:9092',
  scan.startup.mode = 'earliest'
)
ROW FORMAT JSON;
```

#### 3. Creating a Materialized View for Customers

- **Purpose**: Materialize the latest state of each customer record.
- **Action**: Use a Common Table Expression (CTE) and windowing to create the materialized view.

##### Example: Creating a Materialized View for Customers

```sql
CREATE MATERIALIZED VIEW customers_mv AS
WITH ranked_customers AS (
  SELECT
    c.after AS customer_data,
    c.op,
    c.ts,
    ROW_NUMBER() OVER (
      PARTITION BY c.after.id
      ORDER BY c.ts DESC
    ) AS rn
  FROM customers AS c
)
SELECT customer_data.*
FROM ranked_customers
WHERE rn = 1 AND op != 'D';  -- Exclude deleted records
```

- **Explanation**:
  - **Windowing Function**: `ROW_NUMBER()` assigns a sequential number to each row within a partition of customers, ordered by timestamp descending.
  - **Filtering**: Selects the latest record (`rn = 1`) and excludes deleted records (`op != 'D'`).

#### 4. Alternative: Using a Debezium Connector for CDC Data

- **Action**: Use a custom connector to directly create a materialized view from Debezium CDC data.

##### Example: Creating a Materialized View with Debezium Connector

```sql
CREATE SOURCE customers_mv (
  id BIGINT PRIMARY KEY,
  name VARCHAR,
  email VARCHAR,
  ip_address VARCHAR
)
WITH (
  connector = 'kafka-debezium-cdc',
  topic = 'customers',
  properties.bootstrap.server = 'kafka:9092',
  scan.startup.mode = 'earliest'
)
ROW FORMAT JSON;
```

- **Note**: The custom connector handles CDC data and maintains the materialized view automatically.

#### 5. Enriching Click Events with Customer Data

- **Action**: Create a materialized view that joins click events with customer data.

##### Example: Creating an Enriched Click Events View

```sql
CREATE MATERIALIZED VIEW click_events_enriched AS
SELECT e.*, c.*
FROM click_events e
JOIN customers_mv c ON e.ip_address = c.ip_address;
```

#### 6. Further Enrichment with Product Data

- **Action**: Join with the `products` table to include product information.

##### Example: Enriching with Product Data

```sql
CREATE MATERIALIZED VIEW click_events_fully_enriched AS
SELECT e.*, c.*, p.*
FROM click_events e
JOIN customers_mv c ON e.ip_address = c.ip_address
JOIN products_mv p ON e.product_id = p.product_id;
```

- **Assumption**: A materialized view `products_mv` has been created similarly to `customers_mv`.

#### 7. Serving Analytical Pull Queries

- **Action**: End-users or applications execute pull queries against the enriched materialized views.

##### Example: Analytical Query

```sql
SELECT
  c.customer_name,
  COUNT(*) AS click_count
FROM click_events_enriched e
JOIN customers_mv c ON e.ip_address = c.ip_address
WHERE e.ts >= NOW() - INTERVAL '1 DAY'
GROUP BY c.customer_name
ORDER BY click_count DESC
LIMIT 10;
```

- **Purpose**: Retrieve the top 10 customers by click count in the last day.

---

## CDC Connectors

- **Purpose**: Specialized connectors that handle Change Data Capture (CDC) streams, automatically maintaining materialized views.
- **Variations**: Different streaming databases may have their own CDC connectors with specific names and configurations.
- **Note**: Always refer to the streaming database's documentation for accurate connector configurations.

### Bypassing the Streaming Platform

- Some streaming databases offer connectors that connect directly to the OLTP database, bypassing the streaming platform.
- **Advantage**: Simplifies the pipeline by reducing dependencies.
- **Disadvantage**: Limits the ability to replicate CDC data to multiple targets.

---

## Architectural Considerations

- **Materialized Views**: Central to streaming databases, enabling efficient querying of real-time data.
- **State Store**: The internal storage where materialized views (change tables) are maintained.
- **Append Tables**: Data that passes through or is stored in topic-like constructs, not held in the state store.
- **Storage Type**: The choice between row-based and columnar storage affects query performance and suitability for different use cases.

---

## Mathematical Concepts

### Window Functions and Partitioning

- **ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)**:
  - Assigns a unique sequential integer to rows within a partition, ordered by a specified column.
- **Use Case**: Identifying the latest record for each entity (e.g., customer) based on a timestamp.

### Example Calculation

Given customer updates:

| id  | name     | ts                   |
|-----|----------|----------------------|
| 1   | Alice    | 2023-07-20 10:00:00  |
| 1   | Alice B. | 2023-07-20 12:00:00  |
| 2   | Bob      | 2023-07-20 11:00:00  |

Applying:

```sql
ROW_NUMBER() OVER (
  PARTITION BY id
  ORDER BY ts DESC
) AS rn
```

Results:

| id  | name     | ts                   | rn |
|-----|----------|----------------------|----|
| 1   | Alice B. | 2023-07-20 12:00:00  | 1  |
| 1   | Alice    | 2023-07-20 10:00:00  | 2  |
| 2   | Bob      | 2023-07-20 11:00:00  | 1  |

- Selecting records where `rn = 1` gives the latest state for each customer.

---

## Using Multiple Streaming Databases

- **Scenario**: Applications may require both row-based and columnar streaming databases to serve different query patterns.
- **Consideration**: This setup requires both databases to build replicas of the tables.
- **Solution**: Utilize the streaming platform (e.g., Kafka) to publish CDC data, enabling multiple consumers to build their own replicas.

---

## Best Practices

1. **Understand Data Flow**: Clearly map out how data moves through the streaming database, from ingestion to materialized views.
2. **Leverage CDC Connectors**: Use specialized connectors to simplify the ingestion and materialization of change data.
3. **Optimize Queries**: Tailor queries to the storage type (row-based vs. columnar) for optimal performance.
4. **Monitor State Store Size**: Be mindful of the state store's size, especially when dealing with high-velocity streams.
5. **Plan for Scalability**: Design the architecture to accommodate future growth and additional use cases.

---

## Conclusion

Streaming databases enhance debuggability by consolidating stream processing and data storage, providing a unified platform for defining, executing, and querying data pipelines. By utilizing materialized views and leveraging familiar SQL interfaces, data engineers can more easily develop, debug, and optimize streaming applications. However, it's essential to be aware of the limitations of SQL in streaming contexts and consider the architectural implications when deploying streaming databases.

---

## References

- **Debezium Documentation**: [Debezium](https://debezium.io/)
- **ksqlDB Documentation**: [ksqlDB](https://ksqldb.io/)
- **Materialize Documentation**: [Materialize](https://materialize.com/docs/)
- **Apache Kafka Documentation**: [Kafka](https://kafka.apache.org/)
- **Window Functions in SQL**: [PostgreSQL Documentation](https://www.postgresql.org/docs/current/tutorial-window.html)

---

## Tags

#StreamingDatabases #Debugging #MaterializedViews #StreamProcessing #SQL #CDC #DataEngineering #RealTimeAnalytics #StaffPlusNotes #DistributedSystems

---

## Footnotes

1. **Data Definition Language (DDL)**: A subset of SQL used to define or modify database structures, such as creating or altering tables and views.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.