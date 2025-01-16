https://www.youtube.com/watch?v=2SUBRE6wGiA
---
### **Introduction to Time Series Databases**
Time series databases (TSDBs) are specialized databases designed to handle **time-stamped data** efficiently. Common use cases include stock trading, metrics, logs, and sensor data. The goal of these databases is to store large volumes of time-stamped data, perform analytics, and manage the lifecycle of the data (e.g., downsampling and aging out old data).

---
### **What is InfluxDB?**
- **InfluxDB** is an open-source, **MIT-licensed**, time series database written in **Go**. It is designed to handle **high write throughput** and **efficient querying** of time-series data.
- InfluxDB supports **single-server** deployments in its open-source version, while a **commercial version** supports features such as **high availability** and **horizontal scalability**.
### **Types of Time Series in InfluxDB**
- **Regular Time Series**: Fixed interval data points (e.g., server metrics sampled every 10 seconds).
- **Irregular Time Series**: Event-driven time series, such as API request times or stock trades. You can derive a regular time series from an irregular one by aggregating over time windows.
---
### **Why a Specialized Database for Time Series?**
#### **Challenges with General Databases:**
1. **Scale**: Handling billions of data points (e.g., monitoring 2,000 servers at 1,000 measurements per server every 10 seconds = ~17.2 billion data points/day).
2. **Efficient Aging of Data**: High precision data is often kept for short periods, with lower precision summaries stored for longer.
3. **Deletes & Updates**: General databases like relational databases struggle with high-frequency deletes, often handled by creating separate tables for each time block.
4. **Query Performance**: High write throughput and the ability to quickly perform **range queries** over time are essential.

---
### **InfluxDB Architecture**

InfluxDB is organized into two primary components:
1. **Time-Series Database**: Handles the raw time series data.
2. **Inverted Index**: Maps metadata (tags) to the time series for efficient querying.
---
### **Data Model in InfluxDB**
- **Measurement**: Analogous to a table in a relational database.
- **Tags**: Key-value pairs used to index time series. Values are strings, allowing filtering by metadata.
- **Fields**: Key-value pairs where values can be integers, floats, booleans, or strings. These are **not indexed**.
- **Timestamp**: The time at which a value is recorded, stored at **nanosecond precision**
#### **Example Data Ingestion in Line Protocol:**

```plaintext
cpu,host=server01,region=us_west usage_idle=87,usage_user=12 1609871230000000000
```

This represents:
- Measurement: `cpu`
- Tags: `host=server01`, `region=us_west`
- Fields: `usage_idle=87`, `usage_user=12`
- Timestamp: `1609871230000000000` (in nanoseconds)

---

### **Storage Engine: TSM Tree (Time-Structured Merge Tree)**

InfluxDB initially used various third-party storage engines (e.g., LevelDB, RocksDB, LMDB), but none met the performance requirements. In 2015, InfluxDB introduced the **TSM Tree**, a custom storage engine optimized for time series data.

#### **TSM Tree Components:**

1. **Write-Ahead Log (WAL)**: Appends new writes to ensure durability.
2. **In-Memory Cache**: Temporarily holds data before flushing to disk.
3. **TSM Files**: Immutable on-disk storage files that hold compressed time series data.

#### **Write Path**:
- New writes are appended to the WAL and stored in the in-memory cache.
- Periodically, data from the cache is flushed to TSM files.

#### **Compression**:
- **Timestamps**: Compressed using techniques like **Run-Length Encoding** (RLE) for regular intervals.
- **Values**: Compression varies by data type (e.g., **Double-Delta** for floating-point values).

```plaintext
TSM File Structure:
  Header: Magic Bytes + Version
  Data Blocks: Timestamped value blocks
  Index: Mappings of series to data block locations
  Footer: Offset to index
```

#### **Compaction**:
- Compactions merge multiple smaller TSM files into larger ones, improving read performance and compression.
- Inactive shards (e.g., previous day’s data) are fully compacted to optimize for reads.

---

### **InfluxDB Query Language**

InfluxDB has a **SQL-like query language**, but it’s moving toward a **functional query language** better suited for time series data.

#### **SQL-Like Query Example**:

```sql
SELECT percentile(usage_idle, 90) FROM cpu
WHERE region = 'us_west' AND time > now() - 12h
GROUP BY time(10m), host
```

- This query retrieves the 90th percentile of `usage_idle` for the last 12 hours, grouped by 10-minute windows for each host in the `us_west` region.

#### **Functional Query Language (Planned)**:

InfluxDB is developing a **functional query language** to express time series operations more naturally. This language will resemble **chained function calls**, similar to **D3** or **jQuery**.

Example:

```plaintext
from(bucket: "mydb")
  |> range(start: -30m)
  |> filter(fn: (r) => r._measurement == "cpu" and r._field == "usage_idle")
  |> group(columns: ["host"])
  |> window(every: 10m)
  |> percentile(column: "_value", percentile: 90)
```

---

### **Inverted Index: Efficient Querying**

InfluxDB uses an **inverted index** to map **tags** (metadata) to time series IDs for efficient querying.

#### **Index Example**:
```plaintext
Measurement: cpu
Tags: host=server01, region=us_west
Fields: usage_idle, usage_user
```

- **Inverted Index** maps tags to **series IDs**, which in turn map to specific TSM file locations.
- The first version of the index is in-memory, but future versions will be **disk-based** to handle higher cardinality efficiently.

#### **Index Compression Techniques**:
- **Robinhood Hashing**: Used for efficient indexing.
- **HyperLogLog++**: Used for **cardinality estimation** (e.g., counting the number of unique hosts in a region).

---

### **Advanced Features in InfluxDB**

#### **Handling Cardinality Explosion**:
- Cardinality refers to the number of unique series. High cardinality can affect query and write performance.
- InfluxDB is optimized for **high cardinality scenarios**, but performance can degrade when cardinality exceeds ~20 million series.

#### **Retention Policies & Downsampling**:
- Retention policies define how long data is kept at different levels of precision (e.g., 10-second intervals for 7 days, hourly intervals for 1 year).
- **Downsampling** allows users to store aggregated data (e.g., mean, max, min) for longer periods, reducing storage requirements.

#### **Handling Deletes**:
- Deletes are handled by writing **tombstones**, which are resolved during queries.
- **Compactions** eventually remove deleted data from disk.

---

### **Use Cases of InfluxDB**

1. **Stock Trades**: Monitoring millions of stock trades and generating aggregate statistics over various time windows.
2. **Server Monitoring**: Tracking metrics (e.g., CPU usage, memory usage) for thousands of servers sampled every few seconds.
3. **IoT and Sensor Data**: Collecting and analyzing data from sensors, such as temperature, humidity, or motion, deployed across large geographic regions.

---

### **Conclusion**

InfluxDB is a powerful and flexible time series database designed for high-write throughput, efficient compression, and fast query performance over large datasets. Its architecture, based on the TSM Tree, allows for optimized storage and querying of time-series data, making it ideal for monitoring, IoT, financial data, and any other use case involving high-velocity time-stamped data.

