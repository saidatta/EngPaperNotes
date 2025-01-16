https://hemantkgupta.medium.com/insights-from-paper-pinot-realtime-olap-for-530-million-users-b82a8a5478d6
#### **Introduction**

Apache Pinot is a **real-time distributed OLAP** (Online Analytical Processing) system developed by LinkedIn, designed to support high-throughput analytical queries with minimal latency. It excels at handling real-time data ingestion and provides a robust infrastructure for analytical queries at a massive scale. Pinot’s architecture and core design principles ensure scalability, low latency, and cost efficiency, making it an integral component for interactive analytics in modern web-scale applications.

---
#### **Core Design Requirements for a Scalable Real-time OLAP System**
1. **Interactive-Level Performance**:
   - Designed to deliver **sub-second latency** on analytical queries, even during peak loads.
   - Uses **columnar storage**, which is optimized for read-heavy workloads typical in OLAP systems.
   - Implements **inverted indexes** and other index types to accelerate read performance and optimize query execution.

2. **Scalability**:
   - Achieves **near-linear scaling** for both data ingestion and query processing by horizontally scaling its components.
   - Supports partitioning of segments and distribution across servers, allowing the system to handle increasing data volumes and query loads.
   - Uses **Apache Helix** for cluster management, segment balancing, and dynamic scaling.

3. **Cost-Effectiveness**:
   - Efficiently leverages **commodity hardware** by decoupling compute from storage.
   - Local ephemeral storage is used for caching, while persistent object stores (e.g., S3, HDFS) manage durable storage.
   - Minimizes hardware requirements and operational costs by offloading storage management to cloud storage solutions.

4. **Low Data Ingestion Latency**:
   - Uses a **lambda architecture** to manage real-time ingestion (via Kafka) alongside batch ingestion (via Hadoop).
   - Supports append-only immutable data, enabling ingestion latencies on the order of a few seconds.
   - Utilizes **real-time segments** for streaming data, supporting concurrent ingestion and query processing without compromising performance.

5. **Flexibility**:
   - Offers ad-hoc query support with PQL (Pinot Query Language), allowing for dynamic querying across a variety of dimensions and metrics.
   - Can integrate with various data sources, providing seamless support for both batch and streaming data pipelines.
   - Supports complex data transformations, custom indexing strategies, and multi-dimensional aggregation (e.g., star-cubing).

6. **Fault Tolerance & Continuous Operation**:
   - Utilizes **Zookeeper** for metadata storage, ensuring high availability and synchronization across cluster nodes.
   - Implements built-in **segment replication** for data redundancy, ensuring data availability and recovery.
   - Supports transparent recovery mechanisms in case of node failures, maintaining uninterrupted service.

7. **Cloud-Friendly Architecture**:
   - Native support for cloud environments, using cloud-based object stores for segment storage.
   - Stateless nodes (controllers, brokers, servers, minions) allow for rapid scaling and replacement without data loss.
   - HTTP-based communication between components enables easy integration with cloud-native tools and services (e.g., load balancers, monitoring).

---

#### **Architecture Overview**

Pinot’s architecture is built around the concept of **segments** and leverages various components to manage ingestion, query routing, and execution. 
![[Screenshot 2024-10-21 at 2.36.31 PM.png]]
##### **Key Components**
1. **Controllers**:
   - Serve as the **control plane** of Pinot.
   - Manage the lifecycle of segments, including segment assignment, addition, deletion, and replication.
   - Handle cluster metadata management using **Apache Helix**.
   - Use a **leader-election protocol** to ensure that one controller acts as the primary node for task coordination.

2. **Brokers**:
   - Act as query routers, distributing queries to the appropriate server nodes based on routing tables.
   - Collect partial results from servers, perform result aggregation, and return the final output to the client.
   - Support **query result merging**, error handling, and optimization of query routing to minimize latency.

3. **Servers**:
   - Store segments and manage segment-level query processing.
   - Execute both **logical and physical query plans** for segments assigned to them.
   - Support custom indexing techniques (e.g., inverted indexes, range indexes) to optimize query execution.

4. **Minions**:
   - Perform background maintenance tasks, such as segment purging, reindexing, and compaction.
   - Communicate with controllers to receive job assignments and update metadata post-completion.

5. **Zookeeper**:
   - Stores cluster metadata, including segment assignments, server states, and routing information.
   - Facilitates communication between Pinot components, ensuring **synchronization** and **coordination** across nodes.

6. **Persistent Object Store**:
   - Segments are stored in durable storage (e.g., HDFS, S3) for persistence.
   - Ephemeral local storage acts as a cache, enabling faster read access during query execution.

##### **Data and Query Model**
- Data in Pinot is represented as **tables**, which consist of **segments**.
  - **Segments**: Immutable units of storage, typically containing a few million records. Each segment has its metadata, index files, and record data.
  - #### **Segment Lifecycle**

1. **Segment Load**:
    
    - Managed by Helix’s state machines.
    - Segments transition from **OFFLINE** to **ONLINE** (batch data) or **CONSUMING** (real-time data).
    - Real-time segments use Kafka consumers to read from a given offset.
2. **Segment States**:
    
    - **OFFLINE**: Initial state, before being processed by servers.
    - **ONLINE**: Segment is available for querying.
    - **CONSUMING**: Real-time segment actively consuming data from Kafka.
3. **Realtime Segment Completion Protocol**:
    
    - Ensures consensus among independent replicas on segment contents.
    - Protocol actions include **HOLD**, **DISCARD**, **CATCHUP**, **KEEP**, **COMMIT**, and **NOTLEADER**.
    - Segments are committed after a sufficient number of replicas reach consensus.
- ![[Screenshot 2024-10-21 at 2.36.54 PM.png]]
  - Supports **columnar data storage** for faster read performance.
  - Uses **dictionary encoding** and **bit-packing** for efficient storage and retrieval.
  - Segments are logically partitioned into **real-time** (from Kafka) and **offline** (from batch sources), supporting hybrid tables that merge data streams.

- **Query Model**:
  - Queries are written in **PQL (Pinot Query Language)**, which supports selection, projection, aggregation, and top-n queries.
  - Supports dynamic routing of queries based on segment availability and routing tables.
  - Performs real-time merging of hybrid tables to deliver results that span both real-time and offline data.
![[Screenshot 2024-10-21 at 2.38.11 PM.png]]
##### **Data Upload**
- Segments are uploaded to controllers via HTTP POST.
- Controllers verify segment integrity, ensure size quotas, write metadata to Zookeeper, and update cluster states.
![[Screenshot 2024-10-21 at 2.38.51 PM.png]]
##### **Query Processing Workflow**
1. **Query arrives at the Broker**:
   - Broker parses the query and performs query optimization.
   - Selects a routing table to determine the segments to process.
   
2. **Servers execute the query**:
   - Generate logical and physical query plans for segments.
   - Apply segment-specific optimizations based on available indexes and physical record layouts.
   - Execute the query plans, aggregating results at the segment level.

3. **Brokers merge results**:
   - Collect partial results from servers, aggregate them, and return the final output to the client.
   - Handle error scenarios (e.g., partial results, timeouts) and mark query results accordingly.![[Screenshot 2024-10-21 at 2.38.36 PM.png]]
![[Screenshot 2024-10-21 at 2.38.21 PM.png]]
##### **Indexing Strategies & Optimization Techniques**
- Supports multiple index types, such as **bitmap indexes**, **inverted indexes**, and **range indexes**.
- Optimizes queries using columnar data orientation, enabling **efficient scans** for specific columns.
- Uses **star-cubing** to optimize iceberg queries by creating star trees, which allow efficient traversal for complex aggregations.
###### **Example of Iceberg Cubing for Aggregation Queries**
```sql
SELECT SUM(Impressions) 
FROM Table 
WHERE Browser = 'firefox' 
GROUP BY Country;
```
- **Star trees** pre-aggregate data based on specific dimensions, allowing fast retrieval of aggregated results.
---
#### **Advanced Topics: Routing & Partitioning**
1. **Routing Table Optimization**:
   - Pre-generates routing tables based on segment availability and server load.
   - Uses a **balanced routing strategy** by default, ensuring that segments are distributed evenly across servers.
   - Implements a **random greedy strategy** for NP-hard routing problems, approximating a minimal subset of segments while balancing server loads.
2. **Partitioning and Load Balancing**:
   - Pinot supports various partitioning strategies (e.g., range-based, hash-based) to distribute data evenly across servers.
   - Dynamic load balancing adapts to changing query patterns and segment availability, ensuring consistent performance.
3. **Multitenancy Management**:
   - Uses a **token bucket algorithm** to allocate query resources per tenant, preventing resource starvation.
   - Implements **resource isolation** to ensure fair resource distribution among multiple tenants on shared infrastructure.
---
#### **Pinot in the Cloud**
- Pinot is built to be **cloud-native**, leveraging cloud-based object stores (e.g., S3, GCS) for segment persistence.
- Uses **ephemeral local storage** as a cache, enabling nodes to be rapidly replaced or scaled without impacting data consistency.
- All user-facing operations are HTTP-based, facilitating seamless integration with cloud-native services (e.g., Kubernetes, cloud load balancers).
##### **Scaling Pinot in Cloud Environments**
- Supports horizontal scaling by adding more brokers, servers, or minions.
- Uses cloud infrastructure's elasticity to add/remove compute resources dynamically based on load.
- Optimizes costs by separating compute and storage, using object storage for durable segment storage and local storage for cache.

---
#### **Conclusion & Takeaways for Staff+ Engineers**

Apache Pinot is a highly optimized, real-time OLAP system designed for **large-scale, low-latency analytical workloads**. It achieves scalability, performance, and fault tolerance through its **segment-based architecture**, hybrid ingestion model, and flexible query processing.

**Key Takeaways**:
- **Decoupled Compute & Storage**: Separates storage management from compute, improving scalability and resource utilization.
- **Hybrid Tables**: Supports seamless merging of real-time and offline data, enabling interactive analytics on the latest data.
- **Advanced Indexing & Query Optimizations**: Uses a variety of indexes, cubing techniques, and routing strategies to accelerate queries and minimize latency.
- **Cloud-Native Design**: Built for cloud environments, leveraging ephemeral local storage, persistent object stores, and cloud orchestration tools for efficient scaling.

Pinot exemplifies modern OLAP architecture by handling complex analytical workloads while maintaining sub-second query response times and continuous availability, making it a robust choice for real-time analytics at scale.

-----
#### **Distributed Systems Aspects of Pinot**
Pinot is designed to handle distributed systems' inherent challenges while offering high availability, fault tolerance, and low latency. This section explores these characteristics in detail:
##### **1. Consensus & Coordination with Apache Helix**
   - Uses **Apache Helix** for cluster management, ensuring consensus across distributed nodes.
   - Helix manages:
     - **Leader election** for controllers, brokers, and minions.
     - **Segment assignment** and replication across servers.
     - **Failure detection and recovery**, monitoring nodes' health and reassigning segments or roles in case of node failure.
   - This coordination ensures that the cluster dynamically adapts to changes in node availability, maintaining consistency and performance.
##### **2. Segment Replication & Fault Tolerance**
   - Pinot supports **replication of segments** to ensure data availability.
   - Replicated segments are stored across multiple servers, preventing data loss in case of server failures.
   - During query execution, brokers aggregate results from replicas to maintain availability, even if a segment becomes temporarily inaccessible.
   - **Replication strategies** can be configured to balance availability and cost, depending on the specific deployment (e.g., 2x or 3x replication for critical tables).
##### **3. Distributed Ingestion & Data Consistency**
   - Pinot supports a **distributed ingestion model**, integrating batch (HDFS, S3) and real-time (Kafka) data ingestion.
   - Real-time segments are consumed independently by replicas from Kafka, leading to possible data divergence.
   - To address this, Pinot implements a **Segment Completion Protocol**, ensuring that independent replicas achieve consensus on final segment contents.
     - When a segment reaches completion, the controller coordinates replicas, guiding whether to **COMMIT**, **HOLD**, or **DISCARD** based on offsets and the segment state.
     - This process ensures eventual consistency across replicas, which is critical for accurate analytics.
##### **4. Data Partitioning & Sharding**
   - Pinot uses **data partitioning** to improve parallelism and scalability.
   - Partitioning strategies include:
     - **Range-based partitioning**: Data is split based on specific range values of the partitioning key.
     - **Hash-based partitioning**: Distributes data based on a hash of the partitioning key, ensuring even distribution across nodes.
   - Partitioning is critical to minimizing query latency by reducing the number of servers queried and enabling parallel execution across multiple segments.
##### **5. Leaderless Architecture for Query Processing**
   - Unlike many traditional databases, Pinot uses a **leaderless architecture** for query execution:
     - Any broker can route queries to any server responsible for a segment.
     - No single server is the designated leader for a segment, reducing potential bottlenecks and single points of failure.
   - This model supports **load balancing** and improves availability, as any server can respond to segment queries.
   - **Routing tables** are dynamically adjusted based on segment availability, load, and cluster state, ensuring optimal routing and minimizing latency.
##### **6. Multitenancy & Resource Isolation**
   - Pinot supports **multitenancy**, allowing multiple tenants to share the same physical cluster while maintaining isolation.
   - Uses a **token bucket algorithm** to allocate resources to tenants:
     - Ensures fair distribution of query resources based on configured limits.
     - Prevents any tenant from monopolizing resources, ensuring stable performance across all tenants.
   - Multitenancy allows efficient use of infrastructure and supports cost-sharing across different teams or applications.
##### **7. Real-time Query Execution Model**
   - Pinot’s real-time query execution model merges results from batch and streaming data sources.
   - Uses **hybrid tables**, where:
     - **Offline data** comes from batch ingestion sources like HDFS.
     - **Real-time data** comes from Kafka streams.
   - When querying hybrid tables, brokers split the query into two sub-queries: one for real-time data and one for offline data, merging results before returning them to the client.
   - This ensures that users receive **up-to-date results** without waiting for batch data processing.
##### **8. Cloud-Native Design Considerations**
   - Pinot's architecture inherently aligns with cloud environments, focusing on **decoupled compute and storage**.
   - Uses **ephemeral local storage** for caching and **persistent object storage** for durability:
     - Nodes can be easily replaced or scaled without affecting data consistency.
   - Offers seamless integration with **Kubernetes** and cloud-based orchestration tools, allowing:
     - Automated scaling of servers, brokers, and controllers.
     - Efficient use of spot instances and cost-effective cloud resources.
   - By using HTTP-based APIs, Pinot ensures easy integration with cloud-native load balancers, monitoring, and logging tools.
##### **9. Performance Optimizations & Indexing Strategies**
   - Pinot's performance hinges on a combination of **indexing techniques** and query optimizations:
     - **Bitmap indexes**, **inverted indexes**, and **range indexes** support fast query execution.
     - **Star-cubing** is used to speed up iceberg queries by pre-aggregating data in **star trees**, which provide rapid access to aggregated results.
     - Pinot dynamically decides the best index to use based on query patterns and available data statistics, ensuring optimal performance.
   - **Data reordering** based on primary and secondary keys helps to optimize physical data layout, improving scan efficiency and reducing I/O operations.
##### **10. Handling Complex Aggregations with Star Trees**
   - **Star-cubing** is an advanced technique in Pinot that builds on traditional OLAP cubes but uses **star trees** for efficiency.
   - Star trees enable faster responses to iceberg queries, where only aggregates that satisfy specific criteria are returned.
   - **Hierarchical traversal** of star trees allows Pinot to retrieve aggregated results quickly, without scanning the entire dataset.
   - This technique is particularly useful for aggregations like:
     ```sql
     SELECT SUM(Impressions) 
     FROM Table 
     WHERE Browser = 'firefox' 
     GROUP BY Country;
     ```
     - By pre-aggregating data based on dimensions like 'Country', Pinot can return results with minimal compute overhead.

##### **11. Routing & Load Balancing Enhancements**
   - Pinot’s routing layer employs advanced strategies to minimize the number of servers involved in query execution:
     - **Balanced routing strategy** distributes segments evenly across servers to avoid hotspots.
     - For large clusters, Pinot implements a **random greedy strategy** to approximate an optimal segment subset for processing.
   - Brokers maintain **routing tables** that map segments to available servers, dynamically adjusting based on server load and segment availability.

##### **12. Segment States & Dynamic Segment Loading**
   - Segment lifecycle is managed by **Helix state machines**, ensuring seamless transitions between states:
     - **OFFLINE → ONLINE** for batch ingestion.
     - **OFFLINE → CONSUMING** for real-time ingestion.
   - **Dynamic segment loading** allows Pinot to fetch only the required segments from object storage, minimizing memory and I/O usage.
   - Real-time segments enter a **CONSUMING state**, where Kafka consumers continuously ingest data until the segment reaches a completion threshold.

##### **13. Efficient Data Uploads & Real-time Segment Completion**
   - Data segments are uploaded to controllers via **HTTP POST**, ensuring segment integrity and compliance with quotas.
   - Real-time segments undergo a **segment completion protocol** to ensure consistency across replicas before committing to storage.

##### **14. Consistency vs. Availability Trade-offs**
   - Pinot’s consistency model emphasizes **eventual consistency** across segment replicas.
   - When real-time data ingestion leads to slight differences in segment content across replicas, Pinot uses consensus-based protocols to reconcile these differences.
   - This trade-off between strict consistency and availability allows Pinot to deliver high throughput and low latency, essential for real-time analytics.

##### **15. Pinot’s Physical Query Plan Generation**
   - The physical query plan is built based on the available indexes, physical record layout, and user query requirements.
   - Logical operators (e.g., filters, aggregations) are translated into physical operators that leverage specific data encodings and indexing structures.
   - Pinot supports the addition of **custom physical operators**, enabling continuous optimization for new data types and query patterns.

##### **16. Pinot’s Role in a Modern Data Lakehouse Architecture**
   - Pinot integrates seamlessly with modern data lakehouse architectures:
     - Acts as a real-time analytics layer, providing low-latency querying for fresh data.
     - Complements batch processing tools (e.g., Spark, Flink) by enabling instant insights over streaming data.
   - The **cloud-native** and distributed design aligns with the principles of a data lakehouse, offering:
     - Unified access to both structured and semi-structured data.
     - Real-time analytics capabilities, coupled with cost-effective storage and scaling.

---

#### **Conclusion & Strategic Insights for Staff+ Engineers**

Apache Pinot stands as a **scalable, real-time OLAP system** that bridges the gap between interactive analytics and distributed systems. It showcases a mix of advanced indexing, query optimization, and distributed data handling techniques, making it a critical component for web-scale analytics.

**Key Strategic Insights**:
- **Separation of Concerns**: Pinot’s architecture separates compute and storage, optimizing for cost-effectiveness and scalability.
- **Real-time Data Handling**: Its real-time segment completion protocol ensures data consistency across replicas, essential for accurate analytics.
- **Scalability**: Built to scale linearly, supporting dynamic growth in data volume and query complexity.
- **Cloud-Native Integration**: Pinot's compatibility with cloud infrastructure and microservices architecture makes it an excellent fit for modern cloud deployments.
- **Future Directions**: Potential integration with AI/ML workloads, extending its capabilities beyond OLAP to real-time feature generation for models.

Pinot embodies the principles of distributed systems, leveraging fault tolerance, consensus protocols, and efficient query processing to deliver analytics at scale

. Its robust, cloud-friendly architecture positions it as a strong candidate for organizations seeking **low-latency, real-time insights** over massive datasets.