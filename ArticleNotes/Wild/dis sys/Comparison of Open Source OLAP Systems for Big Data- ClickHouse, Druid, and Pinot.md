https://leventov.medium.com/comparison-of-the-open-source-olap-systems-for-big-data-clickhouse-druid-and-pinot-8e042a5ed1c7

These three systems—**ClickHouse**, **Druid**, and **Pinot**—are popular open-source OLAP (Online Analytical Processing) systems designed for interactive, high-performance analytics over large volumes of data. They share fundamental similarities but also exhibit distinct architectural differences. This detailed overview explores the **architectures**, **data management approaches**, **query execution paths**, and other **core database internals** of each system.

---
### **Fundamental Similarities**
1. **Coupled Data and Compute**: 
   - Unlike decoupled systems like Google’s BigQuery, all three systems couple data storage and compute processing on the same nodes.
   - Each node handles **both data storage and query processing**, which allows for **faster query execution** but can limit scalability.
  
2. **Indexes and Static Data Distribution**:
   - All three systems utilize **indexes** and **static data distribution** to accelerate query processing. 
   - Unlike SQL-on-Hadoop systems like Hive, Impala, and Presto, which are data format agnostic, ClickHouse, Druid, and Pinot have custom formats that integrate tightly with their engines.
   - **Limitations**: These systems lack support for large data movements between nodes (e.g., joins between large tables).

3. **No Point Updates and Deletes**:
   - These systems primarily focus on **read-heavy workloads** and do not support point updates or deletes. This enables more aggressive compression and indexing, enhancing resource efficiency and query performance.
   - This contrasts with columnar systems like Kudu, InfluxDB, and Vertica, which support point updates but may sacrifice some performance.

4. **Big Data Style Ingestion**:
   - All three systems can ingest data from **streaming sources** like Kafka.
   - **Druid** and **Pinot** support both **streaming and batch ingestion**. 
   - **ClickHouse** supports batch inserts directly, eliminating the need for separate ingestion services.

---

### **Architectural Comparisons**

#### **1. Data Management**
##### **Druid and Pinot** 
- **Data Partitioning**: Data is divided into segments, which are self-contained entities stored in "deep storage" (e.g., HDFS, S3).
- **Segment Handling**: Segments include metadata, compressed data, and indexes, and are distributed across nodes.
- **Coordinator/Controller Nodes**: Assign segments to nodes, handling replication and movement across the cluster.
- **Metadata Storage**:
  - Druid uses **ZooKeeper** for segment management and an SQL database for metadata persistence.
  - Pinot uses **Helix**, an Apache project, along with ZooKeeper for metadata management.

##### **ClickHouse**
- **Flat Data Partitioning**: No segmentation; data is distributed across nodes, with each node responsible for both storage and processing.
- **No Deep Storage**: All data resides on local storage, making nodes responsible for persistence.
- **Manual Load Balancing**: Administrators must manage data distribution manually, unlike Druid and Pinot, which handle it automatically.

##### **Data Distribution Trade-offs**
- In ClickHouse, **queries are amplified** across all nodes holding table partitions, leading to higher processing costs. 
- In Druid and Pinot, queries can hit fewer nodes depending on segment boundaries, making them more efficient for large clusters.

##### **ASCII Diagram of Data Distribution**
```ascii
[ClickHouse Cluster]        [Druid/Pinot Cluster]
+---+---+---+               +---+---+---+
| N | N | N |  vs.          | S | S | S |
| 1 | 2 | 3 |               | 1 | 2 | 3 |
+---+---+---+               +---+---+---+
 Query touches all nodes    Query touches specific segments
```

#### **2. Query Execution**
##### **Druid and Pinot**
- **Broker Nodes**: Handle query distribution and merging results from processing nodes.
  - The broker keeps segment-to-node mapping information, ensuring efficient query routing.
- **Partial Query Handling**: 
  - Pinot can merge results from partial subqueries even if some fail, improving fault tolerance.
  - Druid currently fails the entire query if any subquery fails.

##### **ClickHouse**
- **Distributed Tables**: Each node can act as a broker, issuing subqueries and merging results.
- **Partial Results**: Can merge results from partial subqueries, similar to Pinot.

##### **Query Execution Flow Example (Pseudocode)**
```python
# Druid/Pinot Broker
def handle_query(query):
    segments = segment_mapping(query)
    results = [node.process(segment) for segment in segments]
    return merge(results)

# ClickHouse Distributed Table
def handle_query(query):
    subqueries = generate_subqueries(query)
    results = [node.execute(subquery) for subquery in subqueries]
    return merge(results)
```

#### **3. Data Replication**
##### **Druid and Pinot**
- **Segment-level Replication**:
  - Each segment is replicated for durability.
  - **Deep Storage Replication**: Segments are replicated in storage systems like HDFS or S3.
##### **ClickHouse**
- **Partition-level Replication**:
  - Entire table partitions are replicated between nodes, ensuring both durability and availability.
  - Uses **ZooKeeper** for managing replication states but not required in single-node deployments.

##### **ASCII Diagram of Replication**
```ascii
[Druid/Pinot]              [ClickHouse]
Segment-Level Replication   Table Partition Replication
+------+ +------+           +--------+ +--------+
| Seg1 | | Seg2 |           | Table1 | | Table2 |
| Rep1 | | Rep2 |           | Rep1   | | Rep2   |
+------+ +------+           +--------+ +--------+
```

#### **4. Tiering of Query Nodes (Druid only)**
- **Tiered Query Nodes**: Druid allows old data to be moved to cheaper nodes with higher storage but lower CPU and memory.
- **Cost Efficiency**: Reduces operational costs by allocating less expensive resources to less frequently accessed data.
- **Pinot and ClickHouse**: Do not support this feature natively, although it can be simulated in ClickHouse with subclusters.
---

### **Performance and Cost Considerations**

1. **Compression and Storage Efficiency**
   - ClickHouse, Druid, and Pinot all support columnar compression, but the level of optimization varies:
     - **ClickHouse**: Focuses on compressing entire partitions.
     - **Pinot**: Records min/max values per segment, supporting optimized compression.
     - **Druid**: Uses bit-level compression but requires additional work for sorting.

2. **Cost of Operations**
   - Druid and Pinot incur higher operational costs due to additional components (e.g., brokers, deep storage).
   - ClickHouse's simpler architecture can be more cost-efficient but requires manual data management.

3. **Query Latency**
   - ClickHouse generally achieves lower latency on smaller scales due to direct data access.
   - Druid and Pinot provide better performance on larger scales due to segmented data handling and more effective load balancing.

##### **Latency Equation**
For approximate latency calculation:
```equation
Latency_{total} = Latency_{network} + Latency_{data_access} + Latency_{aggregation}
```

---

### **Use Cases and Recommendations**

1. **Small-Scale Analytics**
   - **Best Fit**: ClickHouse
   - **Why**: Simpler deployment, fewer moving parts, lower latency, and lower maintenance overhead.

2. **High Scalability and Cloud Deployments**
   - **Best Fit**: Druid or Pinot
   - **Why**: Better segmentation, tiered storage options, and fault tolerance for large-scale data handling.

3. **Real-Time Analytics**
   - **Best Fit**: Druid or Pinot
   - **Why**: Efficient handling of streaming data from Kafka, with segment-based optimizations.

---

### **Conclusion**

ClickHouse, Druid, and Pinot all excel at different scales and use cases, with fundamental architectural similarities but different approaches to **data management, query execution, and scalability**. While ClickHouse shines at smaller scales, Druid and Pinot are more suitable for large, distributed environments. The choice of system should be based on **use case complexity**, **expected cluster size**, and **internal technical capabilities** for modifying or extending the chosen system.

### **9. Differences Between Druid and Pinot**

Druid and Pinot share a lot of architectural similarities, but they have distinct differences in their segment management, fault tolerance, query optimization, and other technical details.

#### **9.1 Segment Management**

##### **Segment Management in Druid:**
- **Druid’s Master Node**:
  - The **master node** (Coordinator) in Druid manages the segment metadata and mapping between segments and processing nodes.
  - Metadata is stored in both **ZooKeeper** and a **SQL database**. ZooKeeper holds the minimum required information, while extended metadata is stored in the SQL database.
  - When segments become too old, they are offloaded from the query nodes and removed from ZooKeeper but retained in the SQL database and deep storage.
  - Plans exist for making ZooKeeper optional in Druid by using HTTP-based announcements and commands, with the SQL database as the backup.

##### **Segment Management in Pinot:**
- **Helix Framework Integration**:
  - Pinot uses **Helix**, an external library, for segment and cluster management.
  - Segment metadata is fully managed in **ZooKeeper** through Helix, which tightly integrates with Pinot's operations.
  - Unlike Druid, Pinot’s segment management depends on Helix, making it more dependent on ZooKeeper.

```ascii
Segment Management:
|       Druid        |       Pinot        |
|--------------------|--------------------|
| - Custom logic     | - Helix framework  |
| - SQL + ZooKeeper  | - ZooKeeper only   |
```

#### **9.2 Query Optimization and Execution**

##### **Predicate Pushdown in Pinot:**
- When data is partitioned in **Kafka** by certain keys, Pinot can push down predicates during the query phase, filtering unnecessary segments early.
- Pinot can pre-filter segments based on predicates in queries, reducing the number of segments read, and enhancing performance.
- **Druid** supports key-based partitioning only in **batch segments** created by Hadoop, and does not yet implement predicate pushdown for real-time ingestion.

##### **Pluggability vs. Opinionated Design:**
- **Druid** offers several interchangeable options for core services:
  - Deep storage options: **HDFS**, **S3**, **Google Cloud Storage**.
  - Real-time ingestion sources: **Kafka**, **RabbitMQ**, **Samza**, etc.
  - Telemetry sinks: **Graphite**, **Ambari**, **StatsD**, etc.
- **Pinot** currently has fewer pluggable options, reflecting its development primarily at **LinkedIn**. It supports:
  - **HDFS** or **S3** for deep storage.
  - Only **Kafka** for real-time ingestion, though other options can be added as required.

#### **9.3 Segment Format Optimizations in Pinot**
- **Advanced Compression**: 
  - Pinot compresses data with bit-level granularity, making it more space-efficient compared to byte-level compression in Druid.
  - The use of **inverted indexes** is optional in Pinot, whereas Druid mandates it for all columns.
- **Min-Max Indexing**: 
  - Pinot records min and max values for numeric columns, enabling better pruning of segments.
- **Data Sorting**: 
  - Pinot supports sorting data during ingestion, which boosts compression and query performance.

#### **9.4 Fault Tolerance**
- **Partial Query Success in Pinot**:
  - Pinot’s brokers can merge partial results from nodes, even if some nodes fail, and return a partial result to the user.
  - **Druid** currently fails the entire query if any segment or subquery fails, making it less fault-tolerant during execution.

```ascii
Query Fault Tolerance:
|       Druid        |       Pinot        |
|--------------------|--------------------|
| - Fails entire query  | - Partial results allowed  |
```

#### **9.5 Tiering of Query Processing Nodes**
- **Druid** supports the concept of **query node tiers**, where older data can be stored on nodes with higher disk capacity but lower CPU/memory, reducing infrastructure costs.
- **Pinot** doesn’t have a similar built-in tiering mechanism, but it can be developed given its similar architecture.

#### **9.6 Performance and Scalability**
- **Compression Efficiency**: 
  - Pinot's format enables better compression than Druid, which has led to improved performance in **GROUP BY** queries as observed in Uber’s tests.
- **Query Balancing**:
  - **Druid** has a more sophisticated algorithm for segment assignment, considering dimensions and time intervals. Pinot uses a simpler algorithm, which prioritizes nodes with fewer loaded segments.
- **Scalability in the Cloud**:
  - **Pinot** benefits from its Helix integration, making it cloud-friendly, but it also makes it reliant on ZooKeeper for coordination.
  - **Druid** aims to reduce ZooKeeper dependency and can utilize SQL databases for metadata storage, potentially simplifying cloud deployments.

---

### **10. Summary: Strengths and Weaknesses**

#### **10.1 ClickHouse**
- **Strengths**:
  - Simpler architecture with fewer moving parts.
  - Better suited for **small-scale** deployments (< 100 nodes).
  - More traditional RDBMS-like behavior, making it suitable for users familiar with PostgreSQL.
- **Weaknesses**:
  - **Higher manual management**: Requires significant manual configuration for large clusters.
  - Limited built-in scaling mechanisms, making it challenging for cloud-native applications.

#### **10.2 Druid**
- **Strengths**:
  - **Pluggable architecture**: Supports multiple ingestion, storage, and telemetry sources.
  - **Tiering support** for older data, reducing infrastructure costs.
  - Advanced query routing and balancing.
- **Weaknesses**:
  - Dependence on multiple services, like ZooKeeper and SQL databases, increases operational complexity.
  - Performance can vary greatly based on segment balancing and query distribution.

#### **10.3 Pinot**
- **Strengths**:
  - Superior **fault tolerance** and partial query results.
  - Optimized segment format and better compression.
  - Supports **predicate pushdown** for better performance with specific queries.
- **Weaknesses**:
  - Fewer options for pluggability and real-time ingestion sources.
  - Simpler segment balancing, leading to potential bottlenecks.

---

### **11. Choosing the Right OLAP System**

#### **11.1 Key Considerations**
- **Data Size**:
  - For smaller datasets (< 1 TB), **ClickHouse** may offer a better fit.
  - For larger datasets or multi-terabyte workloads, **Druid** or **Pinot** are better suited due to their handling of segments and tiering.
- **Cloud Deployment**:
  - **Pinot** offers better fault tolerance and integration with cloud services.
  - **Druid** is more flexible for large-scale, cloud-based analytics workloads.
- **Performance Tuning**:
  - If fine-tuning query performance is essential, **Pinot**’s optimizations for sorting, indexing, and compression provide an edge.

#### **11.2 Table: Feature Comparison**

| **Feature**            | **ClickHouse**   | **Druid**          | **Pinot**          |
|------------------------|------------------|--------------------|--------------------|
| Fault Tolerance        | Medium           | Low                | High               |
| Data Compression       | Good             | Medium             | Best               |
| Cloud Scalability      | Moderate         | High               | High               |
| Tiered Query Nodes     | No               | Yes                | No                 |
| Query Balancing        | Basic            | Advanced           | Basic              |
| Segmentation           | No               | Yes                | Yes                |

---

### **12. Final Recommendations**

- **For Small Clusters (up to 100 nodes)**:
  - **ClickHouse** is the best choice for its simplicity and low operational overhead.
- **For Large Clusters (> 500 nodes)**:
  - **Druid** or **Pinot** are better suited, given their scalability and automated data management capabilities.
- **For Cloud Deployments**:
  - **Pinot** offers better native integration, while **Druid**’s planned migration away from ZooKeeper may make it a better choice in the future.

Ultimately, your choice should be guided by your organization's expertise with the system’s language (C++ for ClickHouse, Java for Druid and Pinot) and your ability to modify the source code to better fit your use case.

--- 

These detailed notes provide a comprehensive view of the internals, architecture, and trade-offs among ClickHouse, Druid, and Pinot. Each system has its strengths and weaknesses, making them suitable for specific scenarios based on your data scale, cloud strategy, and performance requirements.