#### **Decoupled Writes and Reads**

##### **Write Path**
1. **Batching at Agents**: 
   - Agents aggregate incoming messages from producers across multiple clients and partitions.
   - Agents maintain a buffer of incoming data, batching writes to the object storage every **250 ms** or after reaching a certain buffer size (e.g., **4-8 MB**).
2. **Metadata Commit**:
   - After writing data to the object store, agents send metadata to the **centralized metadata store**.
   - Offsets are only assigned after this commit, ensuring **strong consistency**.
3. **Acknowledgment**:
   - The data is acknowledged back to the producer only after both the object storage write and metadata commit succeed.
   - This mechanism ensures that data isn’t acknowledged prematurely, maintaining **“at least once” delivery semantics**.

##### **Read Path**
- The read path is significantly different due to the stateless nature of agents:
  - **No partition ownership**: Any agent can handle read requests for any partition.
  - Agents rely on **consistent hashing** to direct read requests to appropriate agents that share object storage reads.
  - **In-memory caching**: Frequent data is cached in memory, reducing the number of GET requests to the object store.

#### **Data Ordering and Sequencing**

1. **Global Sequencing via Metadata Store**:
   - Unlike Kafka, where brokers sequence messages at the partition level, Warpstream sequences data globally at the **metadata store**.
   - This design scales well because the metadata store handles offsets and file locations without processing raw data, enabling **millions of operations per second**.

2. **Concurrent Writes and Final Ordering**:
   - Multiple agents can write data concurrently to object storage.
   - The metadata store decides the final ordering of batches, assigning offsets during commit.

##### **Ordering Algorithm Pseudocode**
```python
def commit_batch(metadata_store, batch):
    # Store file metadata to metadata store
    file_metadata = write_to_object_store(batch)
    # Assign offsets after metadata commit
    offset = metadata_store.assign_offset(file_metadata)
    return offset
```

### **Compaction Strategy**

#### **Live Reads vs. Historical Reads**
1. **Live Reads**:
   - Live reads are optimized using in-memory caching.
   - Agents read from the latest files in the object store, often benefiting from a **high cache hit ratio**.
   - Consistent hashing ensures that adjacent data ranges are coalesced, reducing redundant GET requests.

2. **Historical Reads**:
   - For older data, reads may result in small file retrievals from the object store, increasing cost and latency.
   - Warpstream employs **background compaction** to merge small files into larger, contiguous files.
   - This compaction process ensures that historical reads are cost-effective, as fewer GET requests are needed to retrieve large blocks of data.

##### **Compaction Process Visualization**
```ascii
+-------+   +-------+   +-------+       =>       +-------------+
|File1  |   |File2  |   |File3  |               | Compacted   |
|64 KB  |   |32 KB  |   |128 KB |               | File (1MB)  |
+-------+   +-------+   +-------+               +-------------+
```

#### **LSM-Tree-like Compaction**
- The compaction strategy in Warpstream mimics **LSM-trees**, a popular approach in databases where multiple smaller files are periodically merged into larger files.
- This approach improves read performance by reducing the number of small files, ensuring that each GET request retrieves more relevant data.

##### **Compaction Algorithm Pseudocode**
```python
def compact_files(agent):
    small_files = agent.get_small_files()
    if len(small_files) > COMPACT_THRESHOLD:
        merged_file = merge(small_files)
        write_to_object_store(merged_file)
```

### **Performance Insights**

#### **Cost Model for Object Storage Operations**
Warpstream is designed to be highly **cost-effective** in cloud environments by optimizing storage operations:
1. **Reducing PUT/GET Requests**:
   - Batching writes and compacting reads minimizes the number of PUT/GET operations, which can become expensive at scale.
2. **Optimized for Throughput**:
   - The system scales with **throughput**, not with the number of partitions, allowing efficient handling of **tens of thousands of partitions**.

##### **Cost Optimization Equation**
The cost per batch operation is determined by:
```equation
Cost_per_operation = (num_puts + num_gets) * IO_cost + storage_cost
```
Where:
- **num_puts**: Number of write operations.
- **num_gets**: Number of read operations.
- **IO_cost**: Cost per object storage IO operation.
- **storage_cost**: Cost of storing data in the object store.

### **Tiered Storage in Warpstream**

#### **Design and Implementation**
Warpstream reintroduces the concept of **tiered storage**, but instead of local disks, it uses different classes of **cloud object storage**:
1. **Low-Latency Tier**:
   - For recently written data, Warpstream uses **low-latency object storage** (e.g., S3 Express One Zone).
   - This tier allows faster write acknowledgment and reduced producer latency.
   - Data is initially stored here to maintain low-latency guarantees.
   
2. **High-Latency Tier**:
   - Older data is compacted and moved to **standard object storage** (e.g., S3 Standard).
   - This reduces storage costs for long-term retention while maintaining scalability.
   - The compaction process ensures that less frequently accessed data does not consume high-cost, low-latency storage resources.

#### **Tiered Storage Flow**
1. **Data lands in Low-Latency Tier**: Producers send data to the low-latency tier for immediate availability.
2. **Compaction Process**: Background processes periodically read data from the low-latency tier and merge it into the high-latency tier.
3. **Final Storage**: Data in the high-latency tier is used for historical reads, ensuring cost-efficient storage.

##### **Example Tiered Storage Configuration**
```json
{
  "tieredStorageConfig": {
    "lowLatency": {
      "type": "S3_Express",
      "bucket": "low-latency-bucket"
    },
    "highLatency": {
      "type": "S3_Standard",
      "bucket": "high-latency-bucket"
    }
  }
}
```

### **Latency Trade-offs and Optimizations**

#### **Producer to Consumer Latency**
1. **Real-time Guarantees**:
   - Warpstream achieves **P99 producer-to-consumer latency** of around **1 second** under normal load, which can be reduced to **~150 ms** using optimized object storage configurations.
   - Producers experience a typical latency of **400-500 ms** for acknowledgment, depending on batch size and network conditions.

2. **Low-Latency Object Stores**:
   - Integration with **low-latency object stores** (e.g., S3 Express) further reduces write latency.
   - The trade-off involves higher costs for using low-latency storage, but compactions mitigate long-term costs by moving data to standard storage.

### **Advanced Replication Strategies**

#### **Global Consistency with Stateless Agents**
1. **No Topic Partition Leaders**:
   - Unlike Kafka, where specific brokers handle partition leadership, Warpstream has no dedicated partition leaders.
   - This enables any agent to handle writes or reads, improving load distribution and reducing bottlenecks.
2. **Quorum-Based Writes**:
   - Warpstream can write data to a quorum of **low-latency object stores** to ensure durability.
   - This provides **high availability** and **resiliency** while keeping producer acknowledgment times low.

### **Use Cases and Applications**

#### **1. Real-time Analytics**
- Warpstream is well-suited for real-time analytics where low latency is critical, but cost efficiency is also a priority.
- Examples include financial data processing, IoT telemetry, and log aggregation, where massive volumes of data need to be ingested and processed quickly.

#### **2. High-Throughput Event Streaming**
- Supports high-throughput workloads with thousands of partitions, making it ideal for use cases like social media streams, ad tech platforms, and gaming telemetry.
- Efficient batching, stateless agents, and object storage integration allow it to handle spikes in traffic without performance degradation.

#### **3. Long-term Data Retention**
- By using tiered storage, Warpstream can efficiently retain data for longer durations at lower costs.
- It can serve as a durable message store for compliance-related use cases, where historical data needs to be retained and accessed intermittently.

### **Conclusion**
Warpstream's design represents a shift towards **stateless, cloud-native messaging systems** optimized for cost, performance, and scalability. By rethinking Kafka’s traditional architecture, Warpstream achieves significant reductions in operational complexity and costs while maintaining strong consistency and durability. The system's use of **tiered storage**, **consistent hashing**, and **global sequencing** enables efficient handling of high-throughput, real-time workloads, making it a compelling alternative to self-hosted Kafka in the cloud.

### **Future Enhancements for Warpstream**
1. **Semantic Caching**:
   - Implementing **semantic caching** for better cache management at the agent level.
2. **Improved Metadata Management**:
   - Introducing **

faster metadata synchronization** methods to further reduce read latencies.
3. **Dynamic Tiering**:
   - Automating the migration of data between tiers based on access patterns, optimizing costs, and performance even further.