## Overview

In this section, we explore how **Extract, Load, Transform (ELT)** pipelines can support real-time use cases when the destination database is a **streaming database**. Traditionally, ELT pipelines are associated with batch processing, but streaming databases bridge the gap between data in motion and data at rest, enabling real-time data transformations and analytics.

---
## Traditional ELT Limitations

- **ELT Pipelines**: Involves extracting data, loading it into a destination database, and then transforming it.
- **Limitation**: Transformation occurs in the destination database, which traditionally processes data at rest.
- **Consequence**: Forces batch semantics for downstream processing, unsuitable for real-time use cases.

---

## Real-Time ELT with Streaming Databases

- **Streaming Databases**: Combine stream processing and database functionalities.
- **Integration**:
  - The "loading" and "transformation" steps are mediated by a **topic** on a streaming platform (e.g., Kafka).
  - The streaming database consumes data from this topic.
- **Benefits**:
  - Supports real-time data processing in ELT pipelines.
  - Existing ELT tools (e.g., **dbt**) can now be used for real-time transformations.
  - ELT jobs can be moved from batch-oriented data warehouses to the real-time streaming layer.
  
---

## Advantages of Streaming Databases in ELT

1. **Convergence of Streaming and Batch Processing**:
   - Marries "streaming" (data in motion) with "databases" (data at rest).
   - Enables continuous data transformation and analysis.

2. **Reintroduction of Database Constructs**:
   - **Write-Ahead Log (WAL)** and **Materialized Views** are brought back into the database context.
   - Materialized views run asynchronously in the background, updating with new data.

3. **Unified SQL Engine**:
   - Existing SQL engines can process both data at rest and data in motion.
   - Simplifies development and maintenance.

4. **Support for Push and Pull Queries**:
   - **Push Queries**: Executed in the "streaming" part, processing data as it arrives.
   - **Pull Queries**: Executed in the "database" part, querying stored data.

---

## Types of Streaming Databases

### 1. **Row-Based Streaming Databases**

- **Characteristics**:
  - Utilize row-oriented storage.
  - Optimized for transactional operations and simple lookups (point queries).
- **Examples**:
  - **ksqlDB**: Uses RocksDB for state storage and primary key indexing.

### 2. **Column-Based Streaming Databases**

- **Characteristics**:
  - Utilize columnar storage.
  - Optimized for analytical queries, including fast aggregations.
- **Examples**:
  - **Timeplus**: Employs a columnar persistence layer, suitable for complex analytics.

---

## Pull Queries and Storage Types

- **Row-Based Databases**:
  - Efficient for **simple lookups**.
  - Pull queries are typically point queries.
  
- **Columnar-Based Databases**:
  - Efficient for **analytical queries**.
  - Pull queries can include aggregations and complex computations.

---

## Spectrum of Streaming Databases

![Spectrum of Streaming Databases](spectrum_streaming_databases.png)

*Figure: A spectrum illustrating the range from row-based to column-based streaming databases.*

- **Left Side**:
  - **Row-Based Streaming Databases**.
  - Pull queries invoked by applications (event-driven).
  - No human intervention required.

- **Right Side**:
  - **Column-Based Streaming Databases**.
  - Pull queries invoked by humans or dashboards.
  - Suitable for ad-hoc analytical queries.

---

## Consistency in Streaming Databases

- **Definition**:
  - Ensures data remains valid and adheres to predefined rules and constraints.
  - Transactions bring the database from one consistent state to another.
  
- **Importance**:
  - **Data Integrity**: Prevents violations of integrity rules.
  - **Reliability**: Maintains accurate and trustworthy data.
  
---

## Mathematical Concepts

### Convergence of Streaming and Batch Processing

- **Data in Motion**:
  - Represented as a continuous function \( f(t) \) where \( t \) is time.
  - **Streaming Processing**: Processes data as it arrives.

- **Data at Rest**:
  - Represented as a dataset \( D \) stored in a database.
  - **Batch Processing**: Processes data in fixed intervals.

- **Convergence**:
  - **Streaming Databases** enable processing over \( D(t) \), where \( D(t) \) is the dataset at time \( t \).
  - **Materialized Views** update continuously: \( V(t) = Q(D(t)) \), where \( Q \) is a query.

### Push and Pull Queries

- **Push Queries**:
  - **Definition**: Queries that continuously push updates to subscribers.
  - **Mathematical Representation**:
    \[
    \text{Push\_Result}(t) = Q_{\text{push}}(D(t))
    \]
  
- **Pull Queries**:
  - **Definition**: Queries executed on demand.
  - **Mathematical Representation**:
    \[
    \text{Pull\_Result} = Q_{\text{pull}}(D(t_0))
    \]
    Where \( t_0 \) is the time of query execution.

---

## Code Examples

### Example 1: Real-Time ELT with Streaming Database

**Step 1**: Define Source Table from Streaming Platform

```sql
CREATE SOURCE click_events (
  id INTEGER,
  ts TIMESTAMP,
  url VARCHAR,
  ip_address VARCHAR,
  session_id VARCHAR,
  referrer VARCHAR,
  browser VARCHAR
) WITH (
  connector = 'kafka',
  topic = 'clicks',
  properties.bootstrap.server = 'kafka:9092',
  scan.startup.mode = 'earliest'
) ROW FORMAT JSON;
```

**Step 2**: Create Materialized View for Transformation

```sql
CREATE MATERIALIZED VIEW enriched_clicks AS
SELECT
  e.id,
  e.ts,
  e.url,
  e.ip_address,
  c.name AS customer_name,
  c.email AS customer_email
FROM click_events e
JOIN customers c ON e.ip_address = c.ip_address;
```

- **Explanation**:
  - **Data Ingestion**: Data is extracted and loaded from Kafka into the streaming database.
  - **Transformation**: Performed within the streaming database via the materialized view.

### Example 2: Pull Query on Materialized View

```sql
SELECT
  customer_name,
  COUNT(*) AS click_count
FROM enriched_clicks
GROUP BY customer_name
ORDER BY click_count DESC
LIMIT 10;
```

- **Purpose**: Retrieve the top 10 customers by click count.
- **Benefit**: Low-latency query due to precomputed materialized view.

---

## Best Practices

1. **Leverage Existing ELT Tools**:
   - Tools like **dbt** can be integrated with streaming databases for real-time transformations.

2. **Understand Storage Implications**:
   - Choose the appropriate storage type (row-based vs. column-based) based on query requirements.

3. **Ensure Consistency**:
   - Design transactions and materialized views to maintain data consistency.

4. **Optimize Queries**:
   - Tailor push and pull queries to leverage the strengths of the streaming database.

5. **Monitor Performance**:
   - Continuously monitor the system to identify bottlenecks and optimize resource utilization.

---

## Conclusion

Streaming databases revolutionize the traditional ELT pipelines by enabling real-time data transformations and analytics. By converging stream processing and database functionalities, they bridge the gap between data in motion and data at rest. This convergence allows for the utilization of existing SQL engines to process data asynchronously, transforming databases into streaming databases.

Understanding the differences between row-based and column-based streaming databases is crucial for optimizing query performance. Consistency remains a critical property to ensure data integrity and reliability in streaming environments.

---

## References

1. **Martin Kleppmann**, *Designing Data-Intensive Applications*.
2. **dbt (Data Build Tool)**: [dbt Official Website](https://www.getdbt.com/)

---

## Footnotes

1. **Connector Middleware**: If the downstream database doesn't support reading from the streaming platform directly, tools like Kafka Connect, Striim, StreamSets, or HVR are required to bridge the gap.

2. **dbt (Data Build Tool)**: An open-source tool that enables data engineers and analysts to transform, test, and deploy data transformations using SQL and Python.

---

## Tags

- #StreamingDatabases
- #ELT
- #RealTimeData
- #MaterializedViews
- #StreamProcessing
- #DataEngineering
- #SQL
- #PushPullQueries
- #Consistency
- #StaffPlusNotes

---

Feel free to reach out for any questions or further clarifications on these topics.