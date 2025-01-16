https://www.youtube.com/watch?v=AqY13RjWwJg
#### Introduction
- **Host**: Kap (Geek N Podcast)
- **Guest**: Felipe Mendes, Solution Architect at ScyllaDB
- **Topics**: ScyllaDB features, architecture, comparison with Apache Cassandra, performance optimizations.
#### Background of ScyllaDB
- **Why ScyllaDB?**: Created to optimize Apache Cassandra's performance.
- **Founders**: Creators of the KVM hypervisor.
- **Unique Selling Points**: Monstrously fast, scalable, high throughput, low latency.
- **API Compatibility**: Fully compatible with Apache Cassandra, DynamoDB, and experimental REST API.
- **Book Recommendation**: "Database Performance at Scale" - covers NoSQL databases, performance aspects, free and open access.
#### Architecture and Optimizations
- **Framework**: Built on the SeaStar C++ framework.
- **Shard Per Core Architecture**: Each CPU core is an independent unit, maximizing hardware utilization.
- **Custom Cache Implementation**: Own cache implementation, improving performance over Linux page cache.
- **No Garbage Collection**: Eliminates garbage collection pauses common in Java or Go.
- **Scheduling Groups and IO Priority Classes**: Prioritize user workloads, manage background operations effectively.
- **IO Scheduler**: Custom IO scheduler to balance various database operations.
#### Write Path
- **Data Storage**: Data stored in two places - commit log and main table.
- **Commit Log**: Append-only, used for durability.
- **Main Table**: Resides in memory, flushed to SS tables.
#### Read Path and Optimizations
- **Caching**: Heavy use of custom caching for read operations.
- **Index Caching**: Caches SS table indices to reduce disk IO.
- **Bypass Cache Option**: Allows bypassing cache for specific queries.
#### Data Model
- **Similar to Cassandra**: Uses partition and clustering keys.
- **Token Ring Architecture**: Consistent hashing for data distribution.
- **Enhancements**: Additional CQL extensions for timeouts, cache control.
#### Deployment and Scalability
- **Shard vs. Partition**: Shard for processing, partition for data storage.
- **Homogeneous Clusters Recommended**: Similar machine specs for balanced performance.
- **Vertical vs. Horizontal Scaling**: Both are supported; depends on the specific latency or throughput needs.
#### Observability
- **Granular Monitoring**: Cluster, data center, instance, and shard-level metrics.
- **Key Metrics**: IO priority classes, foreground/background writes and reads.
#### Case Study: Traction
- **Background**: Moved from PostgreSQL due to scalability needs.
- **Improvements**: Significant performance gains, detailed in their blog post.
#### Final Thoughts
- **Latency vs. Throughput Optimization**: Requires different approaches; avoiding contention is key for latency.
- **ScyllaDB's Flexibility**: Suitable for a wide range of applications, not just high-scale use cases.

---

#### Code Examples:
1. **CQL Extensions Usage**
   ```cql
   SELECT * FROM my_table USING TIMEOUT 2s;
   SELECT * FROM my_table WHERE id = 1234 BYPASS CACHE;
   ```

2. **Monitoring Query (Pseudo Code)**
   ```python
   # Pseudo code for querying ScyllaDB metrics
   def get_metrics(node, metric_type):
       # metric_type can be 'io_priority', 'foreground_writes', etc.
       return query_scylla_monitoring_api(node, metric_type)
   ```

#### Equations/Metrics:
1. **Throughput Calculation**:
   \[ Throughput = \frac{Total Operations}{Unit Time} \]

2. **Latency Measurement**:
   \[ Latency = \text{Time taken for a specific operation} \]

3. **Cache Hit Ratio**:
   \[ Cache Hit Ratio = \frac{Number of Cache Hits}{Total Cache Accesses} \]

---

#### Additional Notes:
- **For Developers**: Understanding shard per core architecture is crucial for optimal application design.
- **For Data Architects**: Data model considerations in ScyllaDB are similar to Cassandra but with enhanced features.
- **Best Practices**: Regular monitoring, understanding shard-level behavior, and right-sizing the cluster for workload needs.
---
#### Advanced Architectural Details
1. **Shard per Core in-depth**:
    - **Isolation**: Each core (shard) handles its own operations, avoiding costly cross-core communication.
    - **Reactor Model**: Each shard operates in a continuous loop (reactor pattern), handling tasks asynchronously.
    - **Handling Interrupts**: Network interrupts are handled by dedicated cores, reducing interference with database operations.
    - **Performance Impact**: The architecture avoids common performance issues like cache sideling and context switching.
    - **Scalability**: Linear scalability is achieved by equally distributing resources across shards.

2. **SeaStar Framework Specifics**:
    - **Asynchronous Operations**: Everything in SeaStar is non-blocking and asynchronous, ensuring high efficiency.
    - **Memory Allocation**: Custom memory allocation strategies to maximize performance and minimize latency.
    - **Zero Copy Networking**: Reduces overhead in network operations, enhancing throughput.

#### Write Path Enhancements
1. **Commit Log Optimizations**:
    - **Durability**: Commit log ensures data durability before acknowledgment to the client.
    - **Sequential Writes**: Optimized for sequential writes, reducing disk IO overhead.

2. **LSM Tree Optimizations**:
    - **Compaction Strategies**: Customizable compaction strategies to balance between read and write efficiencies.
    - **Write Amplification**: Efforts to reduce write amplification through intelligent compaction.

#### Read Path Enhancements
1. **Bloom Filters**:
    - **Efficiency**: Bloom filters quickly determine if data is present in an SS table, reducing unnecessary disk reads.
    - **Tuning**: Customizable to balance between memory usage and false positive rates.

2. **Partition Indexing**:
    - **Fast Lookup**: Enhances the speed of locating data within a partition.
    - **Reducing Disk Seeks**: Minimizes the number of disk seeks required to locate and read data.

#### Performance Considerations
1. **Garbage Collection Avoidance**:
    - **Language Choice**: C++ usage avoids garbage collection delays typical in JVM-based systems like Cassandra.
    - **Predictable Performance**: More predictable performance profile, especially under high load.

2. **IO and CPU Scheduling**:
    - **Prioritization**: Ability to prioritize certain types of IO or CPU-bound tasks.
    - **Adaptive Scheduling**: Dynamically adapts resource allocation based on current workload demands.

#### Data Distribution and Replication
1. **Consistent Hashing**:
    - **Token Ring**: Utilizes a token ring for distributing data across nodes, ensuring even data distribution.
    - **Replication Strategy**: Configurable replication strategies for data durability and availability.

2. **Multi-Datacenter Support**:
    - **Geographical Distribution**: Supports geographical data distribution for global scale applications.
    - **Replication Latency**: Handles the challenges of replication across geographically distant data centers.

#### Developer and Administrator Insights
1. **Monitoring and Observability**:
    - **Granular Metrics**: Provides detailed metrics at the shard, node, and cluster level.
    - **Performance Bottlenecks**: Tools to identify performance bottlenecks, like uneven shard load.

2. **Tuning and Configuration**:
    - **Cache Tuning**: Techniques to optimize cache size and eviction policies.
    - **Hardware Considerations**: Recommendations for hardware configurations based on workload types.

#### Best Practices
1. **Cluster Configuration**:
    - **Uniform Hardware**: Advises uniform hardware in clusters to prevent imbalances.
    - **Capacity Planning**: Importance of capacity planning for scalability and performance.

2. **Application Design**:
    - **Data Modeling**: Best practices in data modeling to leverage ScyllaDB's performance.
    - **Concurrency Management**: Managing concurrency at the application level to avoid overloading shards.

This deeper dive provides a more nuanced understanding of ScyllaDB’s internals, crucial for software engineers and architects looking to leverage its full potential.