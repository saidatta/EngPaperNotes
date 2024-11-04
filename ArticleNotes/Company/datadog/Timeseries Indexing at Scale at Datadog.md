https://artem.krylysov.com/blog/2024/06/28/timeseries-indexing-at-scale/

These detailed notes cover the evolution of Datadog’s timeseries indexing architecture. The content focuses on distributed systems, time series, and database internals, detailing the old and new indexing approaches, their design decisions, and performance implications.

---
### **Overview**
Datadog collects billions of events per minute, creating immense challenges in indexing timeseries data efficiently. The architecture had to evolve to support:
1. **Increased data volume** (30x growth between 2017-2022).
2. **More sophisticated queries** driven by users’ growing monitoring needs.
3. **Higher cardinality** metrics due to complex tech stacks.

### **Architecture Overview**
The Datadog metrics platform consists of three core components:
1. **Intake**: Ingests metric data points.
2. **Storage**: Stores timeseries data.
3. **Query**: Executes user queries.

#### **ASCII Diagram: Metrics Platform Architecture**
```
+-----------------+
| Datadog Agent   |
+-----------------+
        |
        v
+----------------+
| Load Balancer  |
+----------------+
        |
        v
+-----------------+
| Metrics Intake  | ---> Kafka (Message Broker)
+-----------------+
        |
        v
+--------------------+
| Timeseries Storage |
| (Short-term + Index)|
+--------------------+
        |
        v
+----------------+
| Query Engine   |
+----------------+
```

- **Intake**: Processes data with metric names, tags, timestamps, and values. Example metric:
  ```
  containerd.cpu.total{env:prod, service:event-consumer, host:I-ABC} @timestamp, value
  ```
- **Storage**:
  - Short-term metrics storage is split into:
    - **Timeseries Database (DB)**: Stores `<timeseries_id, timestamp, float64>`.
    - **Timeseries Index**: Stores `<timeseries_id, tags>`, built on top of RocksDB.
- **Query**:
  - Connects to timeseries index nodes, fetches intermediate results, and combines them.

---

### **Original Indexing Strategy**
The initial indexing system, implemented in 2016, faced challenges in scaling as data and query complexity increased.

#### **Design Details**
- Built using Go, with embedded databases: **SQLite** (for metadata) and **RocksDB** (for indexing).
- **Key data stores**:
  - **Tagsets**: Maps timeseries IDs to tags.
  - **Metrics**: Lists timeseries IDs per metric.
  - **Indexes**: Optimizes queries by indexing common tag combinations.

##### **Data Stores Example:**
- **Tagsets Database**:
  ```
  Key          | Value
  1            | env:prod,service:web,host:i-187
  2            | env:prod,service:web,host:i-223
  7            | env:prod,service:db,host:i-409
  ```
- **Metrics Database**:
  ```
  Key          | Value
  cpu.total    | 1,2,7,8,9
  ```
- **Indexes Database**:
  ```
  Key                                  | Value
  cpu.total;service:web,host:i-187     | 1
  cpu.total;service:db                 | 7,8,9
  ```

#### **Performance Challenges**
- Queries often required **full scans** of the Tagsets database, similar to full table scans in SQL.
- Index creation was based on **query logs**, which worked well for periodic queries but not for unpredictable user queries.
- **Operational challenges**: Required manual intervention for index maintenance.

---

### **Next-Gen Indexing Strategy**
The next-generation system adopted an **inverted index** inspired by search engines, improving efficiency, predictability, and scalability.

#### **Inverted Index Design**
An inverted index maps tags to timeseries IDs, akin to search engines mapping words to document IDs.
- **Example Inverted Index**:
  ```
  Key                       | Value
  cpu.total;env:prod        | 1,2,7,8
  cpu.total;service:web     | 1,2,3
  cpu.total;host:i-187      | 1
  ```
- **Query Execution**:
  - To query `cpu.total{service:web AND host:i-187}`:
    1. Fetch `cpu.total;service:web` → `1,2,3`.
    2. Fetch `cpu.total;host:i-187` → `1`.
    3. Compute set intersection: `1`.

#### **Advantages**
- **No more full scans**: Every tag has an index, ensuring consistent query performance.
- **Simpler ingestion**: No need to match timeseries against existing indexes.
- **No backfilling**: Eliminates CPU-intensive background jobs for index maintenance.

#### **Challenges**
- **Write & space amplification**: Each timeseries ID must be stored multiple times (once for each tag).
- **Higher disk usage**: Early tests showed increased write and storage demands, but overall CPU utilization improved.

---

### **Intranode Sharding**
To better utilize CPU cores, the new architecture introduced **intranode sharding**:
- RocksDB shards within a single node allow parallelized reads/writes.
- Hash-based sharding distributes timeseries evenly across shards.

##### **ASCII Diagram: Intranode Sharding**
```
+-----------------------+
|   Timeseries Node     |
|-----------------------|
| RocksDB Shard 1       | <-- Query executes in parallel
| RocksDB Shard 2       |
| ...                   |
| RocksDB Shard N       |
+-----------------------+
```
- **Performance Boost**: Experiments showed **8x speedup** with eight shards on 32-core nodes.

---

### **Switching to Rust**
The move from Go to Rust aimed at reducing CPU costs and improving performance.
- **Go challenges**: High garbage collection (GC) overhead (~30% CPU).
- **Rust advantages**: No GC, better memory control, and faster execution for CPU-intensive operations.

#### **Rust vs. Go Comparison**
- **Group Key Extraction**: Rust is **3x faster** than Go in grouping operations.
  - **Rust code snippet**: 
    ```rust
    fn group_key(tags: &[&str], groups: &[&str]) -> String { ... }
    ```
  - **Go equivalent**: 
    ```go
    func groupKey(tags []string, groups []string) string { ... }
    ```
- **ID Merging**: Rust's k-way merge is **3x faster** than Go’s equivalent.
  - **Rust implementation**: Uses efficient `BinaryHeap`.
  - **Go implementation**: Uses `container/heap` with less optimization.

#### **Benefits of Rust Transition**
- Up to **6x faster** performance for CPU-bound tasks.
- Reduced resource usage, enabling Datadog to handle **20x higher cardinality metrics**.

---

### **Results & Improvements**
The next-gen indexing architecture delivered:
1. **Consistent query performance**: No more full scans, reduced tail latencies.
2. **Reduced query timeouts**: 99% reduction.
3. **50% cost reduction**: Lower hardware usage due to intranode sharding and Rust’s efficiency.

##### **ASCII Graph: Tail Query Latency**
```
|  Latency  |    _______      
|  (ms)     |   /       \     __
|           |  /         \   /
|           |_/           \_/
|          Original       Next-Gen
|         Indexing       Indexing
|----------------------------------
```

---

### **Conclusion**
The redesign of Datadog’s timeseries indexing improved scalability, predictability, and operational efficiency. Key takeaways:
1. **Inverted index** design: Consistent query speed by always having partial indexes available.
2. **Intranode sharding**: Better CPU utilization and scalability.
3. **Rust adoption**: Faster execution, lower CPU costs, and reduced latency.

These optimizations enabled Datadog to maintain high performance in the face of rapidly growing data and query demands, setting a foundation for handling **future scaling challenges** in distributed systems, time series, and database internals.