## Overview

In **Chapter 2**, we transformed data using a stream processing platform and placed the preprocessed data into a **sink topic** within a streaming platform like Kafka. Now, the preprocessed data resides in the **analytical plane**, ready to be served to consumers.

In this chapter, we'll explore:

- **Real-Time Expectations**: Understanding the SLAs for serving real-time analytics.
- **Choosing an Analytical Data Store**: Selecting appropriate data stores to meet real-time SLAs.
- **Sourcing from a Topic**: The role of sink topics and consumer-specific data preparation.
- **Ingestion Transformations**: Optimizing data before it reaches persistent storage.
- **OLTP vs. OLAP**: Comparing transactional and analytical data stores.
- **Row-Based vs. Column-Based Optimization**: How data storage formats impact performance.

![Preprocessed Data in the Analytical Plane](preprocessed_data_analytical_plane.png)

*Figure 3-1: The preprocessed data is available in the sink topic for real-time analytical serving to user-facing dashboards or applications.*

---

## Real-Time Expectations

To effectively serve real-time analytics to consumers (both humans and applications), we need to define and adhere to specific **Service-Level Agreements (SLAs)**. Key metrics include:

### 1. Latency

- **Definition**: The time it takes for an analytics query or computation to complete and return results.
- **Importance**: Low latency is crucial for near-instantaneous insights.
- **SLA Example**:
  - **Average Response Time**: ≤ 100 milliseconds.
  - **Maximum Response Time**: ≤ 500 milliseconds.

### 2. Throughput and Concurrency

- **Definition**: The number of analytics queries or computations processed within a given time frame.
- **Importance**: Indicates system capacity to handle concurrent requests, vital for high-volume scenarios.
- **SLA Example**:
  - **Queries per Second (QPS)**: ≥ 1,000 QPS.
  - **Concurrent Users**: Support for ≥ 500 concurrent users.

### 3. Data Freshness

- **Definition**: How up-to-date the analytics results are in relation to the underlying data streams.
- **Importance**: Ensures users have access to the most recent information.
- **SLA Example**:
  - **Maximum Data Delay**: ≤ 5 seconds from data generation to availability.

### 4. Accuracy

- **Definition**: The correctness and precision of analytics results.
- **Importance**: Critical for decision-making and maintaining trust in the system.
- **SLA Example**:
  - **Error Rate**: ≤ 0.1%.
  - **Confidence Interval**: 99% confidence in statistical computations.

---

### Other SLA Metrics

While the focus is on real-time metrics, other SLAs are important:

- **Availability**: System uptime, e.g., 99.99% availability.
- **Consistency**: Data consistency across different queries and replicas.
- **Scalability**: Ability to handle increasing loads.
- **Security and Privacy**: Compliance with regulations like GDPR, data encryption, access controls.

---

## Choosing an Analytical Data Store

To meet the stringent SLAs for real-time analytics, especially in use cases like clickstream analysis, we need to select data stores optimized for:

- **Low Latency**
- **High Throughput**
- **Real-Time Data Ingestion and Querying**

### Suitable Data Stores

1. **In-Memory Databases**

   - **Examples**: Redis, SingleStore (formerly MemSQL), Hazelcast, Apache Ignite.
   - **Characteristics**:
     - Store data entirely in memory for ultra-fast access.
     - Provide low-latency reads and writes.
   - **Use Cases**:
     - Real-time leaderboards.
     - Fast session storage.

2. **Real-Time OLAP (RTOLAP) Data Stores**

   - **Examples**: Apache Pinot, Apache Druid, ClickHouse, StarRocks, Apache Doris.
   - **Characteristics**:
     - Column-oriented storage for efficient analytical queries.
     - Distributed architecture for scalability.
     - Support for real-time data ingestion from streaming platforms.
   - **Use Cases**:
     - Real-time dashboards.
     - Ad-hoc analytical queries.

3. **Hybrid Transactional/Analytical Processing (HTAP) Databases**

   - **Examples**: TiDB, SingleStore.
   - **Characteristics**:
     - Combine OLTP and OLAP capabilities.
     - Reduce data movement between systems.
     - Provide consistent data for both transactional and analytical workloads.
   - **Use Cases**:
     - Systems requiring both real-time analytics and transactional integrity.

### Unsuitable Options

- **Traditional Data Warehouses**: Not optimized for real-time data; higher latency.
- **Data Lakes and Lakehouses**: Designed for batch processing and large-scale storage, not low-latency querying.

> **Note**: While the term "database" often refers to OLTP systems, we may use "data store" interchangeably to encompass both OLTP and OLAP systems.

---

## Sourcing from a Topic

In our architecture, the preprocessed data is placed into a **sink topic** in a streaming platform like Kafka.

![Sink Topic Consumption](sink_topic_consumption.png)

*Figure 3-2: The sink topic can be consumed by multiple domain consumers requiring differing timestamp formats.*

### Challenges with Consumer-Specific Data Preparation

- **Scenario**:
  - **Domain Consumer 1**: Requires timestamps in seconds.
  - **Domain Consumer 2**: Requires timestamps in `YYYY-MM-DD` format.
- **Problem**:
  - If the data in the sink topic provides timestamps in milliseconds, each consumer must transform the timestamp as part of their queries.
- **Consequences**:
  - **Increased Query Latency**: Additional transformation logic slows down queries.
  - **Resource Overhead**: Each query performs redundant computations.
  - **Potential SLA Violations**: Latency may exceed acceptable thresholds.
  
![Transformation Impact](transformation_impact.png)

*Figure 3-3: On the left, the timestamp transformation is done per query request; on the right, the timestamp transformation is done only once during ingestion.*

---

## Ingestion Transformations

### Definition

**Ingestion transformations** are operations applied to data **during** the ingestion process, before it reaches persistent storage.

- **Stateless Transformations**: Do not require maintaining any state between events.
  - **Examples**: Data type conversions, format changes, basic calculations.
- **Stateful Transformations**: Require maintaining state across events.
  - **Examples**: Aggregations, joins, windowed computations.

### Benefits

- **Performance**: Reduces per-query computation overhead.
- **Consistency**: Ensures all consumers receive data in the desired format.
- **Resource Optimization**: Offloads transformation logic from query time to ingestion time.

### Example: Timestamp Transformation

- **Ingestion Time**:
  - Convert timestamps from milliseconds to `YYYY-MM-DD` format.
  - Apply transformation once per data point during ingestion.
- **Query Time**:
  - Consumers can directly use the formatted timestamp without additional processing.

### Implementing Ingestion Transformations in RTOLAP

#### Apache Pinot Example

Apache Pinot allows ingestion-time transformations using **Ingestion Transforms** in the table configuration.

**Sample Transformation Config**:

```json
{
  "tableIndexConfig": {
    "ingestionConfig": {
      "transformConfigs": [
        {
          "columnName": "formatted_timestamp",
          "transformFunction": "dateTimeConvert(timestamp_in_ms, '1:MILLISECONDS:EPOCH', '1:DAYS:SIMPLE_DATE_FORMAT:yyyy-MM-dd', '1:DAYS')"
        }
      ]
    }
  }
}
```

- **Explanation**:
  - **`timestamp_in_ms`**: Original timestamp in milliseconds.
  - **`formatted_timestamp`**: New column with the formatted date.
  - **`dateTimeConvert`**: Built-in function to convert date formats.

#### Stateless vs. Stateful Transformations

- **Stateless**:
  - Can be easily implemented during ingestion.
  - Do not require maintaining state or context.
- **Stateful**:
  - More complex; may require support from the data store.
  - Examples include pre-aggregations using **star-tree indexes** in Pinot.

### Denormalized Views

- **Definition**: Combining data from multiple tables into one, reducing the need for joins at query time.
- **Advantages**:
  - **Performance**: Eliminates costly join operations during queries.
  - **Simplified Queries**: Easier for users to write and understand.
- **Trade-offs**:
  - **Data Redundancy**: Increases storage requirements.
  - **Data Integrity**: Must manage updates carefully to maintain consistency.

> **Note**: Denormalized views are particularly useful in OLAP systems where read performance is prioritized over storage efficiency.

---

## OLTP vs. OLAP

Understanding the difference between OLTP and OLAP systems is crucial for designing efficient data architectures.

### OLTP (Online Transaction Processing)

- **Purpose**: Manage transactional data for day-to-day operations.
- **Characteristics**:
  - **Row-Based Storage**: Optimized for fast reads and writes of individual records.
  - **Normalized Schema**: Reduces data redundancy.
  - **ACID Compliance**: Ensures data integrity and consistency.
- **Use Cases**:
  - Order processing systems.
  - Inventory management.
  - User account management.

### OLAP (Online Analytical Processing)

- **Purpose**: Support analytical queries for decision-making.
- **Characteristics**:
  - **Column-Based Storage**: Optimized for scanning large volumes of data.
  - **Denormalized Schema**: Facilitates complex queries without joins.
  - **Optimized for Read Performance**: Designed for high-throughput analytical queries.
- **Use Cases**:
  - Business intelligence dashboards.
  - Ad-hoc data analysis.
  - Real-time analytics applications.

---

### ACID Properties in OLTP Systems

1. **Atomicity**

   - **Definition**: Transactions are all-or-nothing.
   - **Example**: Transferring funds between accounts; both debit and credit must succeed or fail together.

2. **Consistency**

   - **Definition**: Transactions move the database from one valid state to another.
   - **Example**: Ensuring foreign key constraints are maintained after an insert.

3. **Isolation**

   - **Definition**: Concurrent transactions do not interfere with each other.
   - **Example**: Reading uncommitted data is prevented to avoid dirty reads.

4. **Durability**

   - **Definition**: Once a transaction is committed, it remains so, even in the event of a system crash.
   - **Example**: Completed transactions are saved to disk or replicated for persistence.

---

## Row-Based vs. Column-Based Optimization

### Data Storage Formats

#### Row-Based Storage

- **Organization**: Stores data row by row.
- **Ideal For**: OLTP systems where transactions involve reading or writing entire rows.
- **Advantages**:
  - Efficient for insert, update, and delete operations.
  - Minimal overhead for retrieving full records.
- **Disadvantages**:
  - Inefficient for analytical queries that access only a few columns.

#### Column-Based Storage

- **Organization**: Stores data column by column.
- **Ideal For**: OLAP systems where queries involve aggregations over columns.
- **Advantages**:
  - **Compression**: Similar data types within columns allow for better compression ratios.
  - **Query Performance**: Reads only necessary columns, reducing I/O.
- **Disadvantages**:
  - Less efficient for transactional workloads involving entire rows.

### Implications for Query Performance

#### Analytical Queries

- **Example Query**:

  ```sql
  SELECT SUM(sales_amount)
  FROM sales_data
  WHERE region = 'North America'
  ```

- **Columnar Storage Benefits**:
  - Reads only the `sales_amount` and `region` columns.
  - Skips irrelevant columns, reducing disk I/O.
  - Compression reduces data size, speeding up reads.

#### Transactional Queries

- **Example Query**:

  ```sql
  UPDATE user_profiles
  SET last_login = NOW()
  WHERE user_id = 12345
  ```

- **Row-Based Storage Benefits**:
  - Efficiently accesses the specific row for update.
  - Minimizes overhead for write operations.

### Compression Techniques

#### Column-Based Compression

- **Run-Length Encoding (RLE)**:
  - Stores sequences of the same data value as a single value and count.
  - **Example**: `[AAAAABBBCCDAA]` becomes `[(A,5), (B,3), (C,2), (D,1), (A,2)]`.

- **Dictionary Encoding**:
  - Replaces actual values with short codes.
  - **Example**: `{A:1, B:2, C:3}`; data `[A,B,C,A]` stored as `[1,2,3,1]`.

- **Bit-Packing**:
  - Stores data using the minimal number of bits required.

#### Impact on Performance

- **Reduced Storage**: Smaller data sizes lead to faster disk reads.
- **Improved Cache Utilization**: More data fits into memory caches.
- **Faster Query Execution**: Less data to scan and process.

---

## Mathematical Explanation: Compression Ratios

Given:

- **Uncompressed Data Size**: \( S_u \)
- **Compressed Data Size**: \( S_c \)
- **Compression Ratio**: \( R = \frac{S_u}{S_c} \)

A higher compression ratio \( R \) implies better compression.

**Example Calculation**:

- **Uncompressed Size**: \( S_u = 1 \text{ GB} \)
- **Compressed Size**: \( S_c = 200 \text{ MB} \)
- **Compression Ratio**:

  \[
  R = \frac{1 \text{ GB}}{200 \text{ MB}} = \frac{1000 \text{ MB}}{200 \text{ MB}} = 5
  \]

This means the data is compressed to **1/5th** of its original size.

---

## Code Examples

### Implementing Ingestion Transformation in Apache Pinot

**Ingestion Transformation Config**:

```json
{
  "tableIndexConfig": {
    "ingestionConfig": {
      "transformConfigs": [
        {
          "columnName": "product_id",
          "transformFunction": "parseProductId(url)"
        },
        {
          "columnName": "event_date",
          "transformFunction": "dateTimeConvert(timestamp_in_ms, '1:MILLISECONDS:EPOCH', '1:DAYS:SIMPLE_DATE_FORMAT:yyyy-MM-dd', '1:DAYS')"
        }
      ]
    }
  }
}
```

**Custom Transform Function**:

```java
public class ParseProductIdFunction implements TransformFunction {
    @Override
    public String transform(Object[] input) {
        String url = (String) input[0];
        // Logic to extract product ID from URL
        return extractProductId(url);
    }
}
```

### Querying Denormalized Data

Assuming we have denormalized data in our RTOLAP:

```sql
SELECT
  event_date,
  product_name,
  SUM(click_count) AS total_clicks
FROM
  clickstream_denormalized
WHERE
  event_date = '2023-07-15'
GROUP BY
  event_date,
  product_name
ORDER BY
  total_clicks DESC
LIMIT 10;
```

- **Explanation**:
  - Efficiently retrieves top 10 products by clicks on a specific date.
  - No joins required due to denormalization.

---

## Best Practices

1. **Perform Ingestion Transformations**:

   - Apply necessary data formatting during ingestion to reduce per-query overhead.

2. **Use Denormalized Schemas for OLAP**:

   - Optimize read performance by reducing the need for joins.

3. **Select Appropriate Data Stores**:

   - Choose data stores that align with your SLA requirements and data patterns.

4. **Monitor SLA Metrics**:

   - Continuously monitor latency, throughput, data freshness, and accuracy.

5. **Optimize Storage Formats**:

   - Use columnar storage and compression in OLAP systems for analytical queries.

---

## Conclusion

Serving real-time data effectively requires careful consideration of SLAs, data storage formats, and system architecture. By understanding the differences between OLTP and OLAP systems, leveraging ingestion transformations, and optimizing data storage, we can meet the stringent demands of real-time analytics applications.

In the next chapter, we'll explore different ways of serving the transformed data to consumers, delving deeper into materialized views and the role of streaming databases.

---

## Additional Notes

### Asynchronous vs. Synchronous Processes

- **Asynchronous Processes**:
  - Run in the background without direct user interaction.
  - Examples: Ingestion transformations, background data processing.

- **Synchronous Processes**:
  - Require user interaction and wait for results.
  - Examples: User-initiated queries, real-time dashboards.

Understanding the nature of processes helps in designing systems that optimize resource utilization and meet user expectations.

### Push vs. Pull Queries

- **Push-Based Systems**:
  - Data is pushed to consumers as it becomes available.
  - Ideal for real-time updates and streaming data.

- **Pull-Based Systems**:
  - Consumers request data when needed.
  - Common in traditional query-response models.

In real-time analytics, a combination of both may be used to balance freshness and resource usage.

---

## Mathematical Concepts

### Query Performance and Data Volume

**Time Complexity**:

- **Row-Based Scans**: \( O(N) \), where \( N \) is the total number of rows.
- **Column-Based Scans**: \( O(M) \), where \( M \) is the number of columns accessed.

**Example**:

- **Total Rows**: \( N = 1,000,000 \)
- **Total Columns**: \( C = 100 \)
- **Columns Accessed**: \( M = 5 \)

**Row-Based Scan Time**:

\[
T_{\text{row}} = k \times N \times C
\]

**Column-Based Scan Time**:

\[
T_{\text{column}} = k \times N \times M
\]

Since \( M \ll C \), column-based storage significantly reduces scan time.

---

## References

- **Apache Pinot Documentation**: [Pinot Docs](https://pinot.apache.org/docs/)
- **Understanding OLAP and OLTP Systems**: [Link](https://www.oracle.com/database/what-is-oltp/)
- **Data Warehousing Concepts**: [Link](https://docs.oracle.com/cd/B10500_01/server.920/a96520/concept.htm)
- **ACID Properties**: [Wikipedia](https://en.wikipedia.org/wiki/ACID)

---

## Next Steps

In preparation for **Chapter 4**, consider exploring:

- **Materialized Views in Streaming Databases**: How they maintain real-time data for queries.
- **Push vs. Pull-Based Query Models**: Their impact on system design and user experience.
- **Advanced Indexing Strategies**: Techniques like star-tree indexes in Pinot.

---

By understanding these concepts, you'll be well-equipped to design and implement systems that efficiently serve real-time data to consumers, meeting the high expectations of modern applications.