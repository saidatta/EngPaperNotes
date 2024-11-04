#### **Overview**
Warpstream, developed by Warpstream Labs and later acquired by Confluent, is a **distributed, Kafka-compatible messaging system** designed for **cloud-native environments**. It presents a **stateless architecture** with significant optimizations over traditional Kafka. The system eliminates the need for local disks entirely, offering improved cost efficiency and simplified operations in the cloud.

This note will cover **Warpstream's design**, its approach to **storage**, and how it integrates **tiered storage** with concepts like **consistent hashing** and **object storage** to provide **low-latency message streaming**. Additionally, it will dive into the technical aspects of **stateless agents**, **metadata management**, and **cost optimizations**.

### **Warpstream Architecture**

#### **Deployment Model**
Warpstream uses a unique architecture called **BUOC (Bring Your Own Cloud)**, consisting of:
1. **Stateless Data Plane**: Deployed in the customer’s cloud, managed by stateless agents that handle the messaging load.
2. **Control Plane**: Managed remotely and responsible for sequencing, metadata management, and maintaining offsets.

This split model reduces operational complexity and network costs while retaining Kafka's compatibility.

#### **Why Warpstream?**

##### **Motivation for Stateless Architecture**
1. **Cost Efficiency**: 
   - Local disks (EBS/SSD) are costly in the cloud, especially for large datasets.
   - Object storage (e.g., S3) offers significantly lower costs, around **2 cents/GB** (with replication included), compared to **50 cents/GB** for SSDs/EBS after accounting for replication and spare capacity.
2. **Simplified Operations**:
   - Traditional Kafka requires managing **stateful brokers**, partition balancing, and replication management.
   - Warpstream uses stateless agents that directly interact with object storage, eliminating these complexities.
3. **Reduced Networking Costs**:
   - In multi-AZ Kafka deployments, data is often transmitted across zones, incurring high network transfer costs. Warpstream’s stateless architecture ensures that producers and consumers interact with agents in the same zone, minimizing inter-AZ costs.

### **Technical Components and Concepts**

#### **1. Stateless Agents**
- **Agents** replace Kafka brokers and are completely **stateless**.
- Agents handle both writes and reads for **any topic partition** and interact directly with the object storage.
- Agents batch produce requests from multiple clients and write them as a single file to object storage every **250 ms**.
##### **Write Path Details**
- Producers send batches to agents.
- Agents buffer and batch these writes and commit them to object storage.
- Data acknowledgment to the producer only happens after:
  1. Data is flushed to object storage.
  2. Metadata is committed in the metadata store.
- Ensures **“at least once”** delivery semantics and **durability guarantees** equivalent to `acks=all` in Kafka.

#### **2. Metadata Management**
- The **metadata store** serves as the **sequencer for the entire cluster**, as opposed to Kafka’s per-partition sequencing model.
- The metadata store only sequences offsets, not the actual data, making it **scalable to millions of operations per second**.
- Ensures global consistency and ordering across partitions by assigning offsets only after data is committed.

##### **Code Example for Metadata Store**
Here’s a simplified code sketch showing how metadata assignment works:

```python
class MetadataStore:
    def __init__(self):
        self.offsets = {}

    def commit_file(self, topic_partition, file_info):
        # Assigns the offset for the committed file
        last_offset = self.offsets.get(topic_partition, 0)
        new_offset = last_offset + file_info.record_count
        self.offsets[topic_partition] = new_offset
        return new_offset
```

This ensures that **offsets** are assigned only when data is durably stored, providing the same consistency guarantees as Kafka.

#### **3. Consistent Hashing for Reads**
- Warpstream uses **consistent hashing** for efficiently distributing read requests among agents.
- The hashing ensures that reads are directed to the correct agent, minimizing redundant GET requests to object storage.

##### **Consistent Hashing Ring Visualization**
```ascii
+------+     +------+     +------+
|Agent1|---> |Agent2|---> |Agent3|
+------+     +------+     +------+
```
- Read requests are hashed to identify the responsible agent.
- Adjacent read requests for similar data are coalesced, reducing object storage requests.

##### **Example Code for Consistent Hashing**
```python
import hashlib

class ConsistentHashing:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, key):
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return self.nodes[hash_val % len(self.nodes)]

# Usage
nodes = ["Agent1", "Agent2", "Agent3"]
ch = ConsistentHashing(nodes)
print(ch.get_node("partition_42"))  # Assigns partition to an agent
```

#### **4. Distributed Map for Caching**
- Warpstream reintroduces a concept akin to Kafka’s **OS page cache**, but across stateless agents.
- Data read requests are routed to a central agent that manages a distributed in-memory cache, minimizing object storage GET requests.

##### **In-Memory Caching Strategy**
- Agents implement a **shared cache** that holds recently accessed data.
- Agents collaborate to fulfill read requests from the local cache whenever possible, reducing the need to interact with object storage.

##### **Cache Hit Process Flow**
```ascii
+---------+       +---------+
|Consumer1|       |Consumer2|
+----+----+       +----+----+
     |                  |
     v                  v
  +-------+          +-------+
  |Agent1 |--------->|Agent2 |
  +-------+          +-------+
      ^                  |
      |                  v
+-------------------------+
| In-Memory Cache         |
+-------------------------+
```

### **Cost Optimization Techniques**

#### **1. Object Storage Cost Efficiency**
- Object storage costs scale linearly with throughput, not partitions, making Warpstream’s design **cost-effective**.
- Reducing the number of PUT/GET operations is crucial to maintaining low costs.

##### **Cost Equation**
The cost equation for Warpstream with object storage:

```equation
Total Cost = (Puts + Gets) * Cost_per_request + Storage * Cost_per_GB
```
- **Puts and Gets**: Minimized by batching and coalescing.
- **Storage Cost**: Reduced by using compacted files for historical reads.

#### **2. Compaction for Historical Reads**
- Compaction reduces the number of small files, allowing historical reads to retrieve larger, contiguous data blocks.

##### **Compaction Process Visualization**
```ascii
+-------+   +-------+   +-------+
|File1  |   |File2  |   |File3  |  =>  Compacted into => +------------+
|32KB   |   |64KB   |   |128KB  |                        | 1MB File   |
+-------+   +-------+   +-------+                        +------------+
```
- Background processes continuously merge small files into larger ones, ensuring cost-effective historical data access.

### **Integration with Tiered Storage**

#### **Low-Latency Tiering**
- Warpstream uses **tiered storage** within object storage to manage latency trade-offs:
  - **Low-latency storage** (e.g., S3 Express) is used for live data.
  - **High-latency storage** (e.g., S3 Standard) is used for older data.
##### **Tiered Storage Configuration**
```json
"tieredStorageConfig": {
  "lowLatency": {
    "storageType": "S3_Express",
    "bucket": "low-latency-bucket"
  },
  "highLatency": {
    "storageType": "S3_Standard",
    "bucket": "high-latency-bucket"
  }
}
```

- Data is initially written to the low-latency tier and later compacted into the high-latency tier, minimizing costs while maintaining performance.
### **Performance Insights and Trade-offs**
- **P99 Producer to Consumer Latency**: ~1 second, with optimizations achieving **<150 ms** using low-latency object storage.
- **Scalability**: Scales linearly with throughput, supporting millions of operations per second.
- **Data Consistency**: Consistency is maintained through the centralized metadata store, which handles sequencing and offset assignment.
### **Conclusion**
Warpstream offers a **cost-effective, cloud-native alternative to Kafka**, leveraging stateless architecture, object storage, consistent hashing, and intelligent tiered storage to optimize for cost and performance. The design not only reduces hardware and networking costs but also achieves **durability and consistency** through innovative use of distributed systems principles.