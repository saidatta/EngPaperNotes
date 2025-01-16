Here are the detailed **Obsidian notes** for Staff+ engineers, based on the transcript of a video explaining the internals of **time-series databases (TSDBs)**, particularly focusing on the storage engine design of Prometheus TSDB. This deep dive includes **code**, **examples**, **equations**, and **ASCII visualizations** to ensure a comprehensive understanding.

---
### **1. Introduction to Time-Series Databases**
A **time-series database (TSDB)** is designed to handle data points that occur over time. Each record in a TSDB typically contains:
- A **timestamp**.
- A **value** (e.g., stock price, CPU usage).
- One or more **tags** to define metadata (e.g., method, path).

#### **Characteristics of TSDBs:**
- **Time as the primary index**: All querying is done on time, e.g., "What is the CPU usage from time X to time Y?".
- **Append-only**: Data is mostly appended and rarely updated or deleted, making it highly **write-heavy**.
  
The focus of this discussion is on **Prometheus TSDB**, though some references are made to **InfluxDB IOx**.

### **2. Time-Series Data Model**

```plaintext
Metric: http_requests_total
Tags:
  - Path: /status
  - Method: GET
  - Instance: instance_01
Timestamp: <time_in_unix_format>
Value: <number_of_requests>
```

In TSDB, a **metric name** defines the data's identity (e.g., `http_requests_total`), and tags provide more detailed context.

---

## **3. Writing Data Efficiently to Disk**

### **Batch Writing and Sequential Writes**
In a TSDB, batch writing is more efficient than writing small, scattered chunks. By buffering data in memory and flushing it to disk sequentially, TSDBs optimize I/O performance.

- **Buffering**: Data is buffered in memory, often called **"Head"**, and then flushed periodically to disk.
- **Flushing to Disk**: After a predefined time (e.g., 2 hours of data), the memory buffer is written as a **block** on disk.

#### **ASCII Visualization**:
```plaintext
+-----------------+          +----------------+          +----------------+
|      Head       |   ---->  |      Block      |   ---->  |      Disk       |
| (In-Memory Data)|          | (2 hours of data)|         | (Stored Block)  |
+-----------------+          +----------------+          +----------------+
```

---

### **4. Chunking Data & Memory Mapping**

**Chunking**: Data is written in smaller manageable pieces called **chunks**. For instance, every 100 data points might be written as a **chunk** in the memory.

- **Chunks in Memory**: Let's say the Head holds 2 hours of data, which could be divided into smaller chunks (e.g., 24-minute chunks).
- **Chunks on Disk**: These chunks are then written to **chunk files** on disk.
#### **Chunk File Details**:
- **Memory Mapping (Mmap)**: TSDB uses **memory-mapped files** to allow direct access to the chunk files from memory. The OS manages the transfer of these files to and from disk.
  
  **Mmap Advantage**:
  - If there is enough memory, the OS caches the data in memory, allowing faster reads.
  - The OS can handle writes to disk asynchronously, optimizing performance.
#### **ASCII Visualization of Chunk Writing**:
```plaintext
+---------+           +----------+      +----------+     +---------+
|  Head   |  ----->   | Chunk #1 | ---> | Chunk #2 | ---> |  Disk   |
| (2 hrs) |           |  (24 min) |     | (24 min)  |     | (Blocks)|
+---------+           +----------+      +----------+     +---------+
```
#### **Code Example for Mmap:**
In Go, memory mapping can be implemented using the `syscall` package:
```go
package main

import (
    "os"
    "syscall"
)

func mmapFile(filename string) ([]byte, error) {
    file, err := os.OpenFile(filename, os.O_RDWR, 0644)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    fileInfo, err := file.Stat()
    if err != nil {
        return nil, err
    }

    data, err := syscall.Mmap(int(file.Fd()), 0, int(fileInfo.Size()), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
    if err != nil {
        return nil, err
    }

    return data, nil
}
```
---
### **5. Write Ahead Log (WAL)**
To prevent data loss, a **Write Ahead Log (WAL)** is maintained. The WAL records every write operation before it is committed to memory or disk. In case of a crash, the WAL can be replayed to restore the database's state.
#### **How WAL Works:**
- Data is **appended** to the WAL before being written to memory.
- **Checkpoints** are used to periodically commit operations from the WAL, allowing old WAL files to be deleted, reducing storage usage.
#### **ASCII Visualization of WAL**:
```plaintext
+----------------------+    +-----------------------+
|     WAL (Append)     |    |     Checkpoint Files   |
+----------------------+    +-----------------------+
| Operation: Write X   |    | Operations up to time X|
| Operation: Delete Y  |    | are summarized here.   |
+----------------------+    +-----------------------+
```
#### **WAL Code Example**:
```go
package wal

import (
    "os"
)

func writeWAL(filename, data string) error {
    file, err := os.OpenFile(filename, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
    if err != nil {
        return err
    }
    defer file.Close()

    _, err = file.WriteString(data + "\n")
    return err
}
```
---
### **6. Tombstones and Deletions**
TSDB uses **tombstone files** to mark deletions. Since blocks are immutable once written, deletions are marked in a separate tombstone file rather than modifying the actual data.
#### **Tombstone Example**:
- If you want to delete data for a certain time range, a tombstone entry is added, and future queries will skip the marked data.
---
### **7. Block Metadata (meta.json)**
Each block on disk contains a **meta.json** file, which holds metadata about the block, such as:
- **Min timestamp**
- **Max timestamp**
- **Sources of the block (e.g., block compaction history)**
#### **meta.json Structure Example**:
```json
{
    "min_time": "1625083200000",
    "max_time": "1625090400000",
    "sources": ["block1", "block2"]
}
```
---
### **8. Compaction**

**Compaction** is the process of merging multiple blocks into one larger block, making it easier and faster to read. Compaction also removes data marked by tombstones, saving disk space.

#### **Compaction Process**:
- Merge multiple small blocks into a larger block.
- Remove tombstone-marked data.

#### **Compaction Visualization**:
```plaintext
+----------+   +----------+   +----------+       +-------------+
|  Block 1 |   |  Block 2 |   |  Block 3 |  ---> |  Compacted   |
+----------+   +----------+   +----------+       |    Block 4   |
                                                  +-------------+
```

---

### **9. Indexing and Inverted Index**

TSDB uses **inverted indexes**, similar to full-text search engines, to efficiently locate data points based on tags. An inverted index helps map tags (e.g., `method=GET`) to the relevant chunk locations on disk.

#### **How Inverted Index Works**:
- For each tag-value pair, the index stores pointers to the locations of the relevant data in chunk files.
- When a query comes in (e.g., `method=GET`), the index provides the chunk files and their offsets where this data is located.

#### **ASCII Visualization of Inverted Index**:
```plaintext
+-------------------+     +-------------------+
|   Tag: method=GET | --> | Chunks: [File1, File3]|
+-------------------+     +-------------------+
|  Tag: method=POST | --> | Chunks: [File2, File4]|
+-------------------+     +-------------------+
```

---

### **10. Separation of Compute and Storage**

Modern time-series databases, such as **InfluxDB IOx**, separate **compute** and **storage**. This allows storage to be scalable independently (e.g., using cloud storage like AWS S3) while keeping compute nodes smaller.

In contrast, **Prometheus TSDB** ties compute and storage together, optimized for metrics and observability.

---

## **Conclusion**

This detailed exploration covered various components of **TSDBs**, especially focusing on **Prometheus TSDB** internals. Key aspects discussed include:
- Efficient writing using **memory mapping** and **chunks**.
- Ensuring data durability via **WAL**.
- Efficient querying through **inverted indexes**.
- **Compaction** for managing multiple blocks and cleaning tombstone data.

#### **Additional Reading:**
- Prometheus TSDB [documentation](https://prometheus

.io/docs/prometheus/latest/storage/)
- InfluxDB IOx blog posts for understanding modern TSDB designs.

--- 

By understanding these TSDB internals, you can apply these principles when designing your own time-series storage systems or optimizing existing solutions.