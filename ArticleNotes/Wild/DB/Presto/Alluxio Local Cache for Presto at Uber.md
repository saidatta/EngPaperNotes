https://www.uber.com/blog/speed-up-presto-with-alluxio-local-cache/
### **Overview of Presto at Uber**
Presto is Uber’s core engine for **distributed SQL querying** and **interactive analytics**, processing:
- **500K queries per day**
- Over **50 PB of data** across **20 clusters** and **7,000 nodes**
- Utilized by multiple teams including **Operations**, **Compliance**, **Marketing**, and **Uber Eats**
#### **Architecture Overview**
The Presto architecture at Uber consists of the following layers:
1. **UI/Client Layer**:
   - Comprises tools like **Google Data Studio**, **Tableau**, internal dashboards, and backend services using **JDBC** or query parsing.
2. **Routing Layer**:
   - Routes queries to different clusters based on cluster stats (e.g., query count, CPU, memory).
   - Functions as a **load balancer** and **query gatekeeper**.
3. **Presto Clusters**:
   - Execute distributed queries by interacting with underlying storage systems like **Hive**, **HDFS**, and **Pinot**.
   - Supports **joins** across different connectors and datasets.
#### **Workload Categories**
- **Interactive**: Used by data scientists/engineers for ad-hoc queries.
- **Scheduled**: Includes batch queries for dashboards, ETL, etc.
---
### **Alluxio Local Cache Integration**

#### **Introduction to Alluxio Local Cache**
Alluxio is integrated as a local caching layer to **improve query performance** by storing frequently accessed data on **NVMe disks**. This local cache is implemented inside the Presto workers, reducing latency by avoiding repeated remote reads from HDFS.
#### **Alluxio's Functionality**
1. **Cache Layer Integration**:
   - Alluxio sits on top of Presto’s HDFS client and checks for data in the local NVMe cache.
   - For cache hits, data is read from the local SSD; otherwise, it's fetched from remote HDFS and then cached for future use.
   - ![[Screenshot 2024-10-21 at 5.01.48 PM.png]]
#### **Key Benefits**
- **Reduced latency**: Local SSDs are significantly faster than HDFS.
- **Increased cache hit rate**: Selective caching increases hit rate from **~65% to >90%**.
- **Reduced load on backend storage**: By serving repeated reads locally.
### **Challenges & Solutions in Implementing Alluxio Cache**
#### **Challenge 1: Real-Time Partition Updates**
- **Problem**: Frequent updates in Hudi tables cause cached partitions to be outdated.
- **Solution**: 
  - Modify the caching key to include the **latest modification time**.
  - The new caching key format ensures the cache reflects the latest data state:
    ```
    Previous key: hdfs://<path>
    New key: hdfs://<path><mod time>
    ```
  - **Drawbacks**:
    - Potential race conditions where the cache key is updated after a new file is modified.
    - Cache space may be wasted by outdated partitions until eviction.
#### **Challenge 2: Cluster Membership Change**
- **Problem**: Soft affinity scheduling is disrupted by node addition/removal, causing cache keys to remap incorrectly.
- **Solution**:
  - Replace mod-based hashing with **consistent hashing**.
  - Consistent hashing maintains relative node ordering, ensuring robust cache locality despite changes.
![[Screenshot 2024-10-21 at 5.04.23 PM.png]]
```ascii
+-----------------+                  +-------------------+
| Mod-Based Hash  |       vs.       | Consistent Hashing|
+-----------------+                  +-------------------+
| Changes all keys |      Change     | Maintains key     |
| on node failure  |       =>        | ordering          |
+-----------------+                  +-------------------+
```
![[Screenshot 2024-10-21 at 5.04.57 PM.png]]
#### **Challenge 3: Cache Size Restriction**
- **Problem**: Presto scans up to **50 PB** daily, while each node only has **500 GB** of local storage, leading to heavy eviction.
- **Solution**: 
  - Implement **cache filters** to selectively cache only high-priority tables and partitions.
  - Cache filters use a static configuration based on traffic patterns, prioritizing:
    - Frequently accessed tables.
    - Frequently accessed partitions.
    - Tables with low update frequency.

#### **Cache Filter Configuration**
Example configuration:
![[Screenshot 2024-10-21 at 5.05.24 PM.png]]
- **Performance Impact**: 
  - Increased cache hit rate from **65% to >90%**.
  - Reduced data load on the local cache, ensuring effective utilization.
The following are the areas to pay attention to when it comes to cache filter:

- Manual, static configuration
- Should be based on traffic pattern, e.g.:
    - Most frequently accessed tables
    - Most common # of partitions being accessed
    - Tables that do not change too frequently
- Ideally, should be based on shadow caching numbers and table-level metrics
### **Metadata Optimization in Alluxio Cache**
#### **Motivation**
- Prevent **stale caching** due to frequent data updates.
- Manage limited cache space effectively using **scoped quota management**.
- Enable **metadata recovery** after server restarts.
#### **File-Level Metadata**
1. **Persistent metadata storage**:
   - Stores metadata on disk instead of in memory to persist across server restarts.
2. **Metadata Structure**:
   - Metadata contains file modification time, scope, and quota for each cached file.
3. **Versioning**:
   - New timestamps are generated with each data update.
   - Outdated timestamps are removed to free space.

```ascii
+-------------------------+
| File-Level Metadata     |
+-------------------------+
| Last Modified Time      |
| Scope                   |
| Quota                   |
| Cached File Path (MD5)  |
+-------------------------+
```
#### **Cache Context and Metrics Aggregation**
1. **Cache Context**: 
   - Contains metadata about the database, schema, table, partition, and cache ID.
   - Passed from **Presto** to **Alluxio** during file operations.
   - ![[Screenshot 2024-10-21 at 5.23.39 PM.png]]
1. **Per-Query Metrics Aggregation**:
   - Metrics include cache hit rates, bytes read from cache, and bytes read from HDFS.
   - Aggregated metrics provide insights into cache performance, helping further optimizations.
   - ![[Screenshot 2024-10-21 at 5.23.51 PM.png]]
### **Future Work & Enhancements**
#### **1. Semantic Cache (SC) Implementation**
- **SC** leverages file-level metadata to cache specific data structures (e.g., Parquet footers, ORC indexes).
- **Benefits**: Improved data locality, faster query response, and reduced memory usage.
#### **2. Efficient Deserialization**
- Plan to use **FlatBuffers** instead of **Protobuf** for metadata storage, as Protobuf contributes to high CPU usage.
- **Impact**: Estimated reduction in deserialization overhead by **20-30%**.
#### **3. Load Balancing Enhancements**
- Implement advanced load balancing for better distribution of cache-related workloads.
- Integrate with Presto's **affinity scheduling** to ensure balanced cache utilization.
### **Conclusion**
The integration of **Alluxio Local Cache** with Presto at Uber significantly improves query performance and reduces latency. By leveraging local NVMe storage, implementing advanced caching strategies, and optimizing metadata management, Uber achieves efficient **compute-storage separation** while maintaining interactive analytics performance. As compute-storage separation continues to grow in big data systems, solutions like **Alluxio** play a pivotal role in bridging compute and storage.

These detailed notes capture the core design, implementation strategies, and optimizations of **Alluxio’s Local Cache** for **Presto** at Uber, making it suitable for **Staff+ engineers** seeking deep insights into **distributed caching systems** and **real-time query performance improvements**.