https://medium.com/@vutrinh274/i-spent-8-hours-researching-warpstream-65ac1dd1f027

WarpStream is a cloud-native distributed messaging system that provides an alternative to Apache Kafka, designed to address some of the operational complexities, cost inefficiencies, and scaling challenges associated with managing Kafka clusters. Unlike Kafka, WarpStream is stateless, storing message data in object storage (e.g., Amazon S3 or Google Cloud Storage) while using a control plane to manage metadata. 

This note provides a detailed comparison between **WarpStream** and **Kafka**, focusing on the internals, operational challenges, cost efficiencies, and how WarpStream manages to improve upon Kafka.

---
## **1. Motivation for WarpStream**

Managing Kafka at scale often requires a dedicated team to handle the operational complexities such as:
- **Adding Nodes**: Adding more nodes to Kafka involves rebalancing topic partitions, replicating data, and managing the high load on an already utilized cluster.
- **Networking Costs**: When Kafka is deployed across multiple availability zones (AZs) in the cloud, network costs can be high due to data transfer across zones.
- **Storage Costs**: Kafka uses durable storage, such as AWS EBS, which becomes expensive with increasing replication factors. Scaling Kafka storage requires adding more machines (with CPU and memory), leading to resource wastage.

### WarpStream Motivation:
WarpStream was designed to simplify these processes by offloading much of the operational burden to a stateless architecture where data is stored in object storage and metadata is managed separately.

---

## **2. Overview of WarpStream Architecture**

WarpStream's architecture differs significantly from Kafka by using a stateless model and object storage for data persistence, which decouples storage from compute resources.

### **Key Components:**

1. **WarpStream Agents**:
   - Stateless components that manage message processing, caching, and fetching.
   - Deployed in the customer’s VPC, ensuring data privacy.
   - The local disk is used as a cache, while long-term data is stored in object storage (S3, GCS, etc.).

2. **WarpStream Cloud**:
   - Manages metadata and the control plane.
   - Decides which agents are responsible for tasks such as compacting data files and managing the cache.
   - Offloads the consensus and coordination tasks from the agents.

3. **Metadata Store**:
   - Replaces Zookeeper or KRaft with a combination of a strongly consistent database and object storage (e.g., DynamoDB and S3 in AWS).
   - Manages topic partition leaders, message ordering, and ensures message consistency.

### **Data Flow in WarpStream**:
- **Data Plane**: WarpStream agents interact with object storage to manage data.
- **Control Plane**: WarpStream Cloud manages metadata and coordination tasks.

---

## **3. Service Discovery in WarpStream**

### Kafka’s Approach:
- In Kafka, clients connect to a list of bootstrap brokers via URLs or IP addresses and issue metadata requests to identify brokers and topic partition leaders.

### WarpStream’s Approach:
- **Stateless Agents**: In WarpStream, agents are stateless and handle any request.
- **WarpStream Bootstrap URL**: Clients use a bootstrap URL to resolve to available WarpStream agents via DNS.
- **Agent Discovery**: The WarpStream agent proxies the metadata request to the discovery service, which returns the appropriate agent for that partition, ensuring AZ locality.
- **Round-Robin Load Balancing**: WarpStream uses round-robin load balancing across agents instead of partition rebalancing, minimizing connections and simplifying scaling.

---

## **4. Write Path in WarpStream**

In Kafka, producers send data to the leader broker of a topic partition. In WarpStream, the process is optimized for object storage and latency.

### **Steps in the Write Path**:
1. **Producer Sends Data**: Producers send batches of messages to agents.
2. **Buffering**: Agents buffer the data for 250ms or until 8 MiB of data is accumulated.
3. **Write to Object Storage**: The buffered data is written to object storage (S3, GCS, etc.).
4. **Metadata Commit**: After the data is stored, agents commit the metadata to the WarpStream metadata store.
5. **Acknowledgement**: Once the metadata is committed, the agent acknowledges the producer's requests.

### **Advantages**:
- WarpStream decouples the storage layer from compute resources, providing cost savings by using object storage.
- Buffering can be adjusted to optimize cost vs. latency.

```ascii
+---------+      +---------+        +------------+          +-------------+
|Producer | ---> | WarpStream | ---> | Object Storage | ---> | Metadata Store|
+---------+      +---------+        +------------+          +-------------+
```

---

## **5. Read Path in WarpStream**

The read path is optimized for caching and fetching data from object storage when necessary.

### **Steps in the Read Path**:
1. **Consumer Fetches Data**: The consumer sends a Fetch() request to an agent.
2. **Routing**: The agent routes the request to the correct agent responsible for the partition.
3. **Cache Check**: The responsible agent checks if the requested data is cached in memory.
4. **Object Storage Access**: If not cached, the agent loads the file chunk from object storage into memory.
5. **Return to Consumer**: Once the data is fetched, it is returned to the consumer, and subsequent requests are served from the cache.

```ascii
+---------+      +---------+        +---------+         +------------+ 
| Consumer| ---> | Agent A  | ---> | Agent B  | ---> | Object Store |
+---------+      +---------+        +---------+         +------------+ 
```

---

## **6. Message Ordering in WarpStream**

Unlike Kafka, where leaders ensure strict message ordering within partitions, WarpStream’s stateless agents write data to object storage in an unordered fashion.

### **How Ordering is Maintained**:
- **Metadata Store**: The order of messages is defined when agents commit metadata to the store after writing to object storage.
- **Ordered Batches**: The metadata store maintains an ordered list of files and batches for each partition, ensuring correct ordering for consumers.

```ascii
+---------+       +-------------+       +-------------+
| Producer| ---> | Metadata Store| ---> | Object Storage|
+---------+       +-------------+       +-------------+
```

---

## **7. WarpStream’s Caching Strategy**

To improve read performance, WarpStream uses a distributed cache in memory.

### **Cache Mechanism**:
- **Consistent Hashing**: WarpStream uses consistent hashing to distribute data chunks across agents.
- **AZ-Aware Cache**: Each availability zone has its own cache to avoid cross-AZ latency and data transfer costs.
- **Data Size Optimization**: Data chunks are approximately 16 MiB, optimized for efficient caching and retrieval.

---

## **8. Compaction in WarpStream**

Compaction in WarpStream consolidates small files into larger files to improve read performance for historical data.

### **Compaction Goals**:
- **File Size Uniformity**: Standardize file sizes to optimize object storage reads.
- **Efficient Sequential Reads**: Ensure that data from the same partition is physically close together in object storage, improving throughput.

```ascii
+---------+      +---------+        +---------+
|  File A | ---> | Compact | ---> | Large File |
+---------+      +---------+        +---------+
```

---

## **9. Advantages of WarpStream Over Kafka**

### **1. Simplified Scalability**:
- WarpStream's stateless design allows it to scale easily by adding more agents, without worrying about partition rebalancing or data replication.

### **2. Cost Efficiency**:
- **Object Storage**: Storing data in object storage (S3, GCS) instead of durable block storage (EBS) significantly reduces costs.
- **Reduced Networking Costs**: AZ-aware caching minimizes cross-zone data transfer costs, making WarpStream more cost-effective in cloud environments.

### **3. Flexibility**:
- **Agent Groups**: WarpStream allows users to isolate workloads by using different groups of agents for specific tasks (e.g., writing, consuming, compaction).
- **Separation of Control and Data Planes**: This architecture provides operational flexibility and efficiency.

---

## **10. Trade-offs: Latency in WarpStream**

While WarpStream is highly cost-effective, it introduces slightly higher latency compared to Kafka. The key contributors to latency include:
- **Buffering Data**: Agents buffer data before writing to object storage, which introduces additional time.
- **Commit to Metadata Store**: Metadata commits to the cloud store add overhead.

### **Mitigating Latency**:
- Users can reduce the buffer time to decrease latency but at the cost of higher object storage writes.

---

### **ASCII Diagram: WarpStream Architecture Overview**

```ascii
+-------------------------------------------+
|                WarpStream Cloud           |
|  +-----------------+    +---------------+ |
|  | Control Plane    |    | Metadata Store| |
|  +-----------------+    +---------------+ |
+-------------------------------------------+
          |                     ^
          |                     |
+--------------------+     +------------+
|  WarpStream Agent  | --> | Object Store|
+--------------------+     +------------+
       |    ^                  |
       |    |                  |
+----------------+   +-----------------+
| Producer       |   |  Consumer        |
+----------------+   +-----------------+
```

---

### **Conclusion**

WarpStream is a compelling alternative to Kafka, particularly in cloud environments, where networking and storage costs can be significant. Its stateless architecture and use of object storage offer clear cost advantages. While there is a trade-off in terms of slightly increased latency, WarpStream provides a robust solution for data streaming

 with reduced operational overhead, making it highly suitable for cloud-based workloads. 

For organizations prioritizing cost savings, simplified scaling, and operational flexibility, WarpStream represents a valuable evolution in distributed streaming systems.