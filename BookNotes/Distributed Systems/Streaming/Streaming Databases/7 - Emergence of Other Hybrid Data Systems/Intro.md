## Overview

In this chapter, we broaden our focus to include the landscape of **hybrid systems** that have emerged in response to the growing demands of modern real-time, event-driven applications. While these systems are not streaming databases as we've defined them earlier, they share qualities that bridge between **relational**, **analytical**, and **streaming workloads**. We will explore the motivations behind their development, the innovative techniques they employ, and the specific use cases that make them relevant. More importantly, we will discuss the niches these other hybrid databases cover. This understanding will allow us to uncover the trends that databases are following to provide real-time analytics.

It's important to acknowledge that a streaming database is also an example of a hybrid system. Hybrid systems take at least two perspectives, and in the streaming database case, the two perspectives are **stream processing** and the **database**.

Appreciating the perspectives of hybrid systems reveals the problems they try to solve and how. In this book, we define streaming databases from the stream processing perspective as:

> A **streaming database** is a stream processor that exposes its state store for clients to issue pull queries.

An alternative definition from the database perspective is:

> A **streaming database** is a database that can consume and emit streams as well as execute materialized views asynchronously.

By defining the hybrid system from both perspectives, we expand the hybrid system's accessibility to other engineers and use cases. **Consistency in stream processing** is an example of this. Streaming database engineers were forced to see the database perspective, through which the lack of consistency in some established stream processors was then identified.

**State stores** can be implemented in many ways: key-value, row-based, and column-based. The implementation of the state store determines the supportable use cases that can range from high consistency requirements to low-latency analytical queries.

Interestingly, streaming databases are just one example of emerging hybrid and converging systems, reducing infrastructure complexity and increasing developer accessibility.

---

## Data Planes

To better understand these emerging systems, let's create a **Venn diagram** of where real-time systems live today. This will help in discerning the different use cases and deployment models in real-time analytical scenarios.

![Venn Diagram of Data Planes](data_planes.png)

*Figure 7-1: The streaming plane and its relation to operational and analytical planes.*

This diagram helps us to see both the streaming and the database perspectives. Respecting all perspectives provides hints as to what the next-generation databases might look like.

### The Three Data Planes

1. **Operational Plane (OLTP)**
   - **Characteristics**: Consistent, row-based storage.
   - **Components**: OLTP databases, applications that use the OLTP database.

2. **Analytical Plane (OLAP)**
   - **Characteristics**: Eventually consistent storage, columnar-based.
   - **Components**: OLAP databases, optimized to serve analytical queries.

3. **Streaming Plane**
   - **Characteristics**: Data mostly in motion.
   - **Components**: Source connectors, sink connectors, streaming platforms like Kafka, stateful stream processors, and streaming databases.

The **streaming data plane** connects the operational and analytical aspects of data processing. It captures and processes real-time data, allowing it to flow seamlessly into the analytical plane for storage, analysis, and insights.

---

## The Streaming Plane in Detail

Let's take a closer look at the streaming plane.

![Detailed Streaming Plane](detailed_streaming_plane.png)

*Figure 7-2: Detailed view of the streaming plane.*

In this diagram:

- **Horizontal Axis**: Consistency spectrum of stream processors, from strictly consistent (left) to eventually consistent (right).
- **Vertical Axis**: Storage types, from row-based at the top to column-based at the bottom.
- **Data Flow**: Streaming data travels from left to right as it moves from the operational plane to the analytical plane.

Connectors and streaming platforms like Kafka also reside in the streaming plane.

---

## Overlaps Between Data Planes

The interesting parts are where the circles overlap. Let's explore these overlaps.

### Hybrid Transactional/Analytical Processing (HTAP) Databases

![HTAP Databases](htap_databases.png)

*Figure 7-3: HTAP databases exist at the intersection of operational and analytical planes.*

- **Definition**: Databases that can handle both OLTP and OLAP workloads.
- **Origin**: Concept introduced by Gartner in 2014.
- **Goal**: Break the wall between transaction processing and analytics to enable more informed, real-time decision-making.

#### HTAP Database Architecture

![HTAP Internal Architecture](htap_architecture.png)

*Figure 7-4: HTAP internal architecture defined by Gartner.*

- **In-Memory Storage**: Used to execute analytical queries on "in-flight" transactions.
- **Persistent Storage**: Used for OLTP workloads, ensuring ACID compliance.

#### Examples of HTAP Databases

| Name          | Vendor      | Storage Implementation                                                                                                                            |
|---------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Unistore**  | Snowflake   | Data is stored in both row store and columnar store. Changes are synchronously updated in the row store and asynchronously flushed to the columnar store. |
| **SingleStoreDB** | SingleStore | Supports on-disk column-based (Columnstore) and in-memory row-based tables. Columnstore is the default table type for SingleStoreDB Cloud.       |
| **TiDB**      | PingCAP     | Supports transactional key-value store (TiKV) and columnar store (TiFlash). TiKV provides transactional APIs with ACID compliance. TiFlash powers analytical queries via columnar storage and an MPP query engine. |
| **HydraDB**   | Hydra       | Supports transactional row-based store (heap tables) and column-based storage layouts (default).                                                    |

#### Limitations of HTAP Databases

- **Monolithic Solutions**: May not effectively hold historical data (terabytes or petabytes).
- **Historical Data Handling**: Not designed to store extensive historical data like pure OLAP systems.
- **Data Divide**: May still require both OLAP and HTAP databases, reintroducing complexity.

### The Triad of Hybrid Databases

![Triad of Hybrid Databases](hybrid_databases_flower.png)

*Figure 7-5: The triad of hybrid databases.*

The overlaps form a flower with three petals:

1. **HTAP Databases**: Overlap between operational and analytical planes.
2. **Streaming OLTP Databases**: Overlap between operational and streaming planes.
   - **Examples**: Row-based, consistent streaming databases like RisingWave and Materialize.
3. **Streaming OLAP Databases**: Overlap between streaming and analytical planes.
   - **Examples**: Column-based, eventually consistent streaming databases like Proton.

The center of the flower (the pistil) is still undefined at this point but represents the potential for databases that encompass all three planes.

---

## Other Hybrid Databases

During our research, we encountered databases that combine unique features to solve problems typically addressed by systems in the streaming plane.

### Examples of Other Hybrid Databases

#### 1. PeerDB

- **Hybrid System**: **Postgres OLTP Database + Stream Processor**
- **Description**:
  - A Postgres-first data-movement platform.
  - Simplifies moving data in and out of Postgres.
  - Enables syncing, transforming, and loading data into an OLAP system.
  - Materialized views would need to be created in the OLTP database.
  - Does not meet our definition of a streaming database but falls within the streaming OLTP database category.

#### 2. Epsio

- **Hybrid System**: **Postgres OLTP Database + External Asynchronous Materialized View**
- **Description**:
  - Plugs into existing databases.
  - Continuously updates results for defined queries whenever underlying data changes.
  - Provides instant and up-to-date results for complex queries without recalculating the entire dataset.
  - Significantly reduces costs.

#### 3. Turso

- **Hybrid System**: **SQLite OLTP Database + Streaming Platform**
- **Description**:
  - Allows development locally and replication globally to multiple locations.
  - Exposes synchronous access to data instead of asynchronous access like Kafka.
  - Focuses on edge computing and low-latency data access.

#### 4. Redpanda

- **Hybrid System**: **Streaming Platform + Apache Iceberg (Database)**
- **Description**:
  - Developers can bring their own query engines to query data in Redpanda's tiered storage.
  - Avoids moving data across different systems, reducing infrastructure footprint on analytics.
  - Provides a unified platform for streaming and batch analytics.

---

## Mathematical Concepts

### Consistency and Storage Types

- **Consistency Models**:
  - **Strict Consistency**: Guarantees that all clients see the same data at the same time.
  - **Eventual Consistency**: Guarantees that, in the absence of updates, all replicas will eventually converge to the same value.

- **Storage Implementations**:
  - **Row-Based Storage**: Stores data row by row; optimized for transactional workloads (OLTP).
  - **Column-Based Storage**: Stores data column by column; optimized for analytical workloads (OLAP).

### Implications of Storage on Use Cases

- **Row-Based Storage**:
  - Suitable for frequent read/write operations on individual records.
  - Provides strong consistency and ACID compliance.

- **Column-Based Storage**:
  - Suitable for read-heavy workloads involving large datasets and complex queries.
  - Can be eventually consistent, favoring scalability and performance over immediate consistency.

---

## Code Examples

While specific code examples are not provided in the text, we can illustrate some concepts with pseudo-code and SQL.

### Example: Creating a Materialized View in a Streaming OLTP Database

Assuming we're using a streaming OLTP database that supports materialized views, such as Materialize.

```sql
-- Create a source from a Kafka topic
CREATE SOURCE transactions
FROM KAFKA BROKER 'localhost:9092' TOPIC 'transactions'
FORMAT AVRO USING SCHEMA 'transaction_schema';

-- Create a materialized view to compute real-time account balances
CREATE MATERIALIZED VIEW account_balances AS
SELECT
  account_id,
  SUM(amount) AS balance
FROM
  transactions
GROUP BY
  account_id;
```

- **Explanation**:
  - The database consumes streaming data from the `transactions` topic.
  - The `account_balances` materialized view is updated in real-time as new transactions arrive.
  - Clients can query `account_balances` to get up-to-date account balances.

### Example: Edge Replication with Turso

```sql
-- Create a table in SQLite
CREATE TABLE user_data (
  user_id INTEGER PRIMARY KEY,
  data TEXT
);

-- Assume Turso provides commands or configurations for replication
-- Configure Turso to replicate 'user_data' to multiple edge locations
```

- **Explanation**:
  - Turso allows you to define tables in SQLite.
  - Data is replicated globally to edge locations.
  - Applications access data synchronously from the nearest replica.

---

## Use Cases and Deployment Models

### Real-Time Analytics

- **Challenge**: Providing real-time insights requires processing and analyzing data as it arrives.
- **Hybrid Systems Solution**:
  - Combine stream processing with databases to enable low-latency analytics.
  - Use streaming databases or HTAP databases to reduce infrastructure complexity.

### Data Movement and Integration

- **Challenge**: Moving data between systems (e.g., from OLTP to OLAP) introduces latency and complexity.
- **Hybrid Systems Solution**:
  - Platforms like PeerDB simplify data movement from OLTP databases to OLAP systems.
  - Redpanda integrates storage and processing to minimize data movement.

### Edge Computing

- **Challenge**: Providing low-latency access to data in geographically distributed applications.
- **Hybrid Systems Solution**:
  - Turso replicates data to edge locations, allowing for fast, synchronous data access.
  - Supports applications that require real-time data at the edge.

---

## Trends in Hybrid Data Systems

- **Convergence of Systems**: Combining features of different data systems to reduce complexity and improve performance.
- **Real-Time Processing**: Increasing demand for systems that can handle real-time data processing and analytics.
- **Developer Accessibility**: Emphasis on making these systems accessible to a broader range of engineers, not just specialists.
- **Infrastructure Simplification**: Reducing the number of systems and data movements required to support modern applications.

---

## Conclusion

The emergence of hybrid data systems reflects the evolving needs of modern applications for real-time data processing and analytics. By combining features from operational, analytical, and streaming systems, these hybrids aim to reduce infrastructure complexity and improve developer accessibility. Understanding these systems and their use cases helps engineers design solutions that leverage the strengths of each data plane while addressing the limitations.

---

## References

- **Gartner's HTAP Definition**: [Hybrid Transactional/Analytical Processing](https://www.gartner.com/en/documents/2658415)
- **Snowflake Unistore**: [Snowflake Documentation](https://docs.snowflake.com/en/user-guide/unistore-overview)
- **SingleStoreDB**: [SingleStore Documentation](https://docs.singlestore.com/)
- **TiDB**: [TiDB Documentation](https://docs.pingcap.com/tidb/stable)
- **HydraDB**: [HydraDB GitHub](https://github.com/hydradatabase/hydra)
- **PeerDB**: [PeerDB GitHub](https://github.com/peerdb-io/peerdb)
- **Epsio**: [Epsio Official Website](https://www.epsio.io/)
- **Turso**: [Turso Official Website](https://turso.tech/)
- **Redpanda**: [Redpanda Documentation](https://docs.redpanda.com/)

---

## Tags

- #HybridDataSystems
- #StreamingDatabases
- #HTAP
- #RealTimeAnalytics
- #DataPlanes
- #Streaming
- #OLTP
- #OLAP
- #DataEngineering
- #StaffPlusNotes

---

## Footnotes

1. **Hybrid Systems**: Systems that combine different data processing paradigms to provide enhanced capabilities.
2. **State Stores**: Internal storage used by stream processors to maintain state between processing events.
3. **Consistency Models**: Defines how consistent the data view is across different nodes in a distributed system.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.