https://docs.google.com/document/d/1Z4FLg3ezHpqvc6zhy0jR6Wi2OL8wLO_lRC6aLkskFgs/edit?tab=t.0
These notes explore the internals of **Apache Pinot**—a distributed OLAP system optimized for real-time analytics—and its implementation of **tiered storage**, covering the architecture, configurations, and performance considerations relevant to Staff+ engineers.

---

### **Apache Pinot Overview**
Apache Pinot is a **real-time distributed OLAP datastore** designed for low-latency analytics on high-throughput event streams. Initially developed at LinkedIn, Pinot serves as the backbone for real-time user-facing analytics. It supports:
- **High-throughput** querying of simple aggregates.
- **Low-latency ingestion** of streaming data.
- **Hybrid queries** that combine batch and streaming data sources.

### **Core Features of Pinot**
1. **Real-time Data Ingestion & Querying:**
   - Supports near-real-time data ingestion from Kafka and batch ingestion from HDFS.
   - Queries are processed on **column-oriented segments**, ensuring high performance.
2. **Hybrid Data Architecture:**
   - Merges batch and real-time data sources using a **lambda architecture**.
   - Implements **hybrid tables** for seamless querying across data tiers.
### **Pinot’s Distributed System Design**
Pinot's architecture is designed for horizontal scaling, fault tolerance, and real-time responsiveness. It achieves this through various components and strategies:
#### **Key Components**
- **Controllers:** Manage segment assignments, rebalancing, and segment metadata.
- **Brokers:** Route incoming queries to servers, aggregate results, and return responses.
- **Servers:** Host segments and execute queries.
- **Minions:** Run background tasks like compaction and purging.
- **Zookeeper & Apache Helix:** Manage metadata, consensus, and cluster state.
#### **Query Execution Flow**
1. A broker receives a client query and **parses** it.
2. It selects a **routing table** to determine the set of servers to contact.
3. Servers generate logical and physical query plans, execute them, and return results to the broker.
4. The broker **aggregates results** and sends them back to the client.

#### **Segment Lifecycle Management**
Pinot's segments transition between states like **OFFLINE**, **CONSUMING**, and **COMPLETED**, managed by Apache Helix. This ensures consistent data ingestion, availability, and query responsiveness.

---
### **Tiered Storage in Pinot**
Tiered storage is designed to manage **data of varying ages across different storage tiers** with varying cost and performance characteristics. It enables cost-effective scaling while maintaining optimal performance.
![[Screenshot 2024-10-21 at 3.01.18 PM.png]]
#### **Motivation for Tiered Storage**
1. **Efficient Resource Utilization:**
   - Recent data (e.g., < 1 month old) requires **low-latency access**, making SSDs an ideal choice.
   - Older data (> 6 months) is accessed less frequently, making it suitable for **lower-cost HDDs**.
2. **Cost Savings:**
   - By shifting older data to cheaper storage, organizations reduce costs while maintaining performance for recent, frequently accessed data.
3. **Scalable Retention Management:**
   - Supports different retention strategies for data across time ranges, helping to maintain system performance as data grows.
#### **Tiered Storage Configuration**
Pinot’s tiered storage is configured using the `tieredStorageConfig` in the table configuration file:
```json
{
  "tableName": "myTable",
  "tableType": "OFFLINE",
  "tenants": {
    "broker": "myTenant",
    "server": "myTenant"
  },
  "tieredStorageConfig": {
    "TIER_1": {
      "segmentSelector": "timeBased",
      "segmentAge": "30d",
      "storageSelector": "pinotServer",
      "tag": "myTier1_OFFLINE",
      "instanceAssignmentConfig": {
        "tagPoolConfig": {...},
        "replicaGroupPartitionConfig": {...}
      }
    },
    "TIER_2": {
      "segmentSelector": "timeBased",
      "segmentAge": "100d",
      "storageSelector": "pinotServer",
      "tag": "myTier2_OFFLINE",
      "instanceAssignmentConfig": null
    }
  }
}
```
#### **Tiered Storage Components**
1. **Segment Selector:**
   - Defines the criteria for segment selection, such as **time-based** selection (e.g., segments older than 30 days).
   - Future extensions could include selection by segment size or segment name.
2. **Segment Age:**
   - Specifies the age threshold for moving segments to the specified tier (e.g., 30 days, 100 days).
   - Ensures that older segments are moved to the appropriate tier.
3. **Storage Selector:**
   - Determines where segments should be stored (e.g., `pinotServer`, `deepStore`).
   - Defines the destination storage for segments based on tier.
4. **Tagging for Tiered Assignment:**
   - Uses tags to differentiate between nodes of different tiers (e.g., `tier1_OFFLINE`, `tier2_OFFLINE`).
   - This allows the system to identify and assign segments to appropriate nodes based on their tier.
#### **Instance Partitions for Tiered Storage**
- The **INSTANCE_PARTITIONS** structure represents how segments are distributed across different tiers.
- The layout includes:
  ```
  INSTANCE_PARTITIONS/
    myTable_OFFLINE
    myTable_TIER_1
    myTable_TIER_2
  ```
- This allows controllers to maintain and update the segment-to-node mapping based on tiers, facilitating seamless tier transitions.

#### **Rebalancing with Tiered Storage**
- Rebalancing operations must account for different segment storage locations and move segments to the appropriate tier.
- Rebalance tasks handle **segment movement**, ensuring segments are correctly assigned and the cluster state is updated.

#### **Periodic Task for Tiered Storage**
- A periodic task scans for segments that meet the criteria for tier movement.
- Moves one replica of a segment at a time to minimize disruptions.
- This task ensures that segments transition smoothly across tiers based on defined policies.
#### **Extensions & Enhancements**
1. **Smart Assignment Based on Data Temperature:**
   - Dynamic segment movement based on data access patterns.
   - Segments accessed frequently could be moved to higher-performance storage.
2. **Tiered Broker Tags:**
   - Similar to Druid, this allows brokers to route queries to segments based on their tier, optimizing query latency.
3. **TierBackend Properties:**
   - Introduced to support storage backends beyond local Pinot servers, such as cloud-based deep storage.
---
### **System Design Considerations for Tiered Storage**
#### **1. Load Distribution Across Tiers**
- Segments are balanced across nodes of different tiers, ensuring no single node is overwhelmed.
- Brokers use **routing tables** that are updated dynamically to account for tier transitions, ensuring consistent query performance.
#### **2. Data Movement Strategies**
- **Time-based movement**: Segments move based on age, typically from fast SSDs to slower HDDs.
- **Adaptive tiering**: Future iterations may involve data temperature analysis, where frequently accessed segments are promoted to faster tiers.
#### **3. Cost vs. Performance Trade-offs**
- The design supports **cost-efficient scaling** by optimizing storage usage based on data access patterns.
- Users can configure specific data retention policies to minimize storage costs while maintaining required performance for recent data.
#### **4. Query Performance Impact**
- Tiered storage can introduce **slight latency increases** for older data, as they reside on slower storage.
- **Iceberg queries** and **star-cubing techniques** can help optimize aggregations and filters on older segments, mitigating the latency impact.
#### **5. Fault Tolerance Across Tiers**
- Replication strategies ensure that segment movement does not affect data availability.
- If a node in a lower tier fails, Pinot can restore segments from persistent storage or higher tiers.
### **Advanced Use Cases for Pinot’s Tiered Storage**
#### **1. Long-term Data Retention**
- With tiered storage, Pinot can maintain data retention for years without incurring prohibitive costs.
- Older segments are stored in cheaper storage tiers, while recent segments leverage SSDs for rapid querying.
#### **2. Historical Data Analysis**
- Analysts can run slower, complex queries on older data while ensuring that interactive queries on recent data remain fast.
- Pinot’s **query planning** dynamically adapts based on segment location, ensuring optimal performance.
### **Conclusion**
Apache Pinot's tiered storage architecture provides a **scalable, cost-efficient solution** for managing large-scale OLAP workloads in distributed systems. By leveraging tiered storage, Pinot ensures **consistent performance** for recent data while minimizing costs for older data. The design choices in tiered storage align with the core principles of **distributed systems**, focusing on **scalability**, **availability**, and **cost optimization**.