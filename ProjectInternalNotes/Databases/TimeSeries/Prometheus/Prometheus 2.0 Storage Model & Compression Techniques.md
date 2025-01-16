Here’s an expanded version of the **Prometheus 2.0 Storage Model & Compression Techniques for 16-Byte Time Series at Scale** Obsidian notes with more details from the transcript. This version includes additional context on the **technical decisions**, **design challenges**, and **specific implementation details**.

---
## Introduction
Prometheus 2.0 tackles the challenge of efficiently storing **time series data** at scale, with billions of samples to handle. This note will dive into the core principles behind Prometheus 2.0's storage design, specifically focusing on **compression techniques**, **chunking**, and **block-based storage**, which make it possible to store and query **16-byte time series data** efficiently.

The key aspects of the new storage model include:
- **Delta encoding** and **XOR compression**.
- **Chunking** and **block-based storage** for efficient data ingestion and query.
- Handling **data churn** in dynamic environments like **Kubernetes**.
- Optimizations in **CPU**, **memory**, and **disk I/O**.
---
## Prometheus Data Model
Prometheus's data model is designed to handle high-dimensional time series data:
- **Metric Names**: Provide the semantic meaning of a time series (e.g., `http_requests_total`).
- **Labels**: Allow partitioning of the data (e.g., `method="GET"`, `status="200"`).
Each unique combination of metric names and labels creates a **distinct time series**.
### ASCII Visualization of Time Series Model

```plaintext
Metric Name: http_requests_total

Labels:
| method = "GET" | code = "200" | instance = "server1" |
---------------------------------------------------------
| time   | value |
|--------|-------|
| t1     | v1    |
| t2     | v2    |
| ...    | ...   |
```

Data is organized on a **two-dimensional plane**:
- **Series**: The vertical axis, representing different time series (metric + labels).
- **Time**: The horizontal axis, representing time as the series evolves.

This design allows querying data by time ranges and specific label filters, such as selecting all time series for `status="200"` within the last hour.

---

## Storage Challenges

### Scale of Data

In dynamic environments with **high churn** (like **Kubernetes**), Prometheus must handle millions of **short-lived** time series. The sheer volume of data stored over time can escalate dramatically:

- Example workload:
  - **2,000 to 15,000 microservices**
  - **5-minute scrape interval**
  - **Every 30 seconds**
  - **1 month of retention**

**Calculation**:
- 2,000 services × 30-second scrape interval = **6,000 time series per second**.
- **432 billion samples** stored per month.
- Each sample occupies **16 bytes** (8 bytes for timestamp, 8 bytes for value).
- **7 terabytes** of raw data per month.

The challenge is not only in **storing** this data but also **querying** it efficiently, particularly in **dynamic environments** where instances come and go frequently.

### Key Issues:
- **Disk I/O Bottlenecks**: Writing millions of samples per second causes excessive disk operations. On **SSD** storage, this can lead to premature wear-out, while on **spinning disks**, seek times become a bottleneck.
- **Churn**: Dynamic systems like Kubernetes often create and delete containers, resulting in short-lived time series but with a very large total number.

### ASCII Visualization of Churn

```plaintext
Time Series Churn in Kubernetes:
|-------|-------|-------|-------|-------|--------|
| Pod 1 | Pod 2 | Pod 3 | Pod 4 | Pod 5 | Pod 6  |
|-------|-------|-------|-------|-------|--------|
| Created, scraped data for a short period, then deleted |
```

---

## Time Series Compression

Prometheus 2.0 uses **delta encoding** for timestamps and **XOR compression** for values to significantly reduce storage size.

### Delta Encoding for Timestamps

In most cases, the **time between samples** is consistent (e.g., scraping every 30 seconds). **Delta encoding** stores the **difference** between consecutive timestamps, and further compresses this by storing the **delta of deltas**.

```plaintext
Timestamps (in seconds): [1609459200, 1609459230, 1609459260, ...]
Delta(T) = [30, 30, 30, ...]
```

Instead of storing each timestamp, Prometheus stores the **initial timestamp** and a **series of deltas**. This results in significantly fewer bytes needed for timestamps.

### XOR Encoding for Values
For **float64 values**, Prometheus uses **XOR encoding**. If two consecutive values are similar, the XOR result will have many **leading zeros**, which can be compressed further.

```plaintext
Values:
v1 = 10.0  (binary: 0100000000100100000000000000000000000000000000000000000000000000)
v2 = 10.1  (binary: 0100000000100100001100110011001100110011001100110011001100110011)

XOR(v1, v2) = 0000000000000000001100110011001100110011001100110011001100110011
```

Prometheus only stores the **XOR result** and compresses the leading and trailing zeros, saving a significant amount of space.
### Compression Results
- Raw data: **16 bytes per sample** (8 bytes for timestamp, 8 bytes for value).
- After compression: **1.37 bytes per sample**.
  - This is a **12x reduction** in storage size.
By using both delta and XOR compression, Prometheus can handle **7 terabytes** of data per month with a **final size of 0.8 terabytes**.
### ASCII Visualization of Delta and XOR Compression

```plaintext
| Time       | Value    |
|------------|----------|
| 1609459200 | 10.0     |
| 1609459230 | 10.1     |
| 1609459260 | 10.1     |

Delta(T):
| 30  | 30  | 30  |

XOR(Value):
| 0x0 | 0x0012 | 0x0 |
```

---
## Chunking & Block-based Storage
### Chunking
Prometheus stores data in **fixed-size chunks** (typically **1 kilobyte** per chunk). Each chunk represents a **series of samples** for a specific time range. Chunks allow Prometheus to:

- **Efficiently store recent data in memory**.
- **Write completed chunks to disk**, reducing random disk I/O.

Each **chunk** contains between 100-1,000 samples, depending on the time series.
### Block-based Storage
Chunks are grouped into **blocks** that represent a fixed time range (e.g., 2 hours). These blocks enable efficient **querying** by time range. When querying data, only the blocks that intersect with the query time range are accessed.
```plaintext
|---------------------- Time ----------------------|
| Block 1 | Block 2 | Block 3 | Block 4 | Block 5   |

Each Block:
[ Index File ]
[ Chunk 1 | Chunk 2 | Chunk 3 | ... ]
```
Blocks also make **data retention** easier: as blocks become older than the retention period, they are simply **deleted**, which is computationally cheap.

---
## Handling Data Churn
### The Problem of Churn
**Kubernetes** and other dynamic environments introduce **churn**, where instances (and therefore their corresponding time series) are short-lived. This results in an explosion of **inactive time series** over time, making querying and storage more complex.
- In Prometheus 1.x, churn could lead to millions of **small files** being created, which overburdens disk I/O.
- In Prometheus 2.0, churn is addressed by **block-based storage** and efficient **chunk management**.
---
## Efficient Querying with Inverted Index
### Inverted Index for Label Filtering
Prometheus uses an **inverted index** (borrowed from search engine technology) to efficiently filter time series by label-value pairs. This index maps label pairs to the **IDs of time series** that contain those labels.
- **Inverted Index**: A mapping of **label-value pairs** to **series IDs**.
This allows Prometheus to quickly identify the time series that match a query, rather than scanning the entire dataset.
### Set Operations
Set operations like **intersection** and **union** are used to combine results when multiple label filters are applied. For example, querying `status="200"` and `method="GET"` requires finding the intersection of two sets of series.
### ASCII Visualization of Inverted Index
```plaintext
Label Pair: "status=200" -> Time Series IDs: [1, 3, 5, 7]
Label Pair: "method=GET" -> Time Series IDs: [2, 3, 4, 7]

Intersection:
Resulting Time Series IDs: [3, 7]
```
---
## Benchmarks & Performance Improvements
### Memory and CPU Usage
Prometheus 2.0 offers dramatic reductions in both **memory** and **CPU usage**:
- **Memory**: Uses **2-5x less memory** than Prometheus 1.x. Memory usage remains stable even under high churn
- **CPU**: Optimized to handle high sample ingestion rates with minimal CPU overhead.
### Disk I/O and Writes
- **Disk writes**: Prometheus 2.0 reduces **disk I/O** by batching writes and using efficient chunking. Disk write operations are reduced by **98-99%**.
### Latency Improvements
Prometheus 2.0 consistently shows **low query latencies**. While Prometheus 1.x experiences performance degradation over time due to churn, Prometheus 2.0 maintains **steady performance** even with millions of active time series.
### Key Metrics from Benchmarks
- **Memory consumption**: 18GB for **4 million series**.
- **CPU usage**: Around **half a CPU core** for handling **100,000 samples per second**.
---
## Conclusion
Prometheus 2.0's design addresses the core challenges of **storing and querying time series data at scale**. By leveraging advanced compression techniques like **delta and XOR encoding**, using **chunking** for efficient ingestion, and employing an **inverted index** for fast querying, Prometheus is able to handle the demands of modern, dynamic environments like Kubernetes.

### Key Takeaways

- **Delta & XOR Compression**: Reduce storage size by 12x.
- **Block-based Storage**: Efficiently handles data churn and retention.
- **Inverted Index**: Enables fast and scalable queries on billions of time series.

Prometheus 2.0 sets the foundation for handling time series data efficiently, even as infrastructures become increasingly dynamic and complex.

### Additional Resources
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Facebook Gorilla Paper on Time Series Compression](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf)