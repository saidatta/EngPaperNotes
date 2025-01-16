### **Overview**
Prometheus TSDB (Time Series Database) is a fundamental component of **Prometheus 2.0**, designed to efficiently store and manage time series data. Understanding the inner workings of the **TSDB**, especially the **Head block** (the in-memory part), is crucial for optimizing its performance and contributing to its development. This note delves into the internal mechanics of the Head block, its interaction with the Write-Ahead Log (WAL), memory mapping, and its role in data lifecycle management.
#### **Table of Contents**
1. **Introduction to Prometheus TSDB**
2. **Head Block Lifecycle**
3. **Write-Ahead Log (WAL) Integration**
4. **Chunk Management in Head Block**
5. **Compaction and Persistent Blocks**
6. **Index and Memory Management**
7. **Handling Restarts**
8. **Code References and Internal Flow**
---
### **1. Introduction to Prometheus TSDB**

- **Release**: Prometheus 2.0 introduced its TSDB several years ago, significantly improving its data storage capabilities.
- **Documentation**: Resources explaining TSDB are limited, with most information found in **Fabian's blog post** or **PromCon 2017** talks.
- **Goal**: This note aims to explain the internal workings of the **Head block** of Prometheus TSDB, serving as a foundation for developers looking to contribute or optimize it.

#### **Components of Prometheus TSDB**
```plaintext
|----------------------------|------------------------------|
|       Head Block           |         Persistent Block     |
|----------------------------|------------------------------|
| In-memory (mutable data)   | On-disk (immutable data)    |
| Uses WAL for durability    | Created from Head Block     |
| Memory-mapped chunks       | After compaction            |
|----------------------------|------------------------------|
```

---

### **2. Head Block Lifecycle**

The **Head block** is the in-memory component of Prometheus TSDB, where all new data samples are initially stored before being persisted to disk.

#### **Lifecycle of a Sample in the Head Block**
1. **Sample Ingestion**:
   - Incoming data samples are stored in **compressed units** called **chunks**.
   - The **active chunk** is the only writable unit within the Head block.

2. **Write-Ahead Log (WAL)**:
   - Each sample is also written to the **WAL** to ensure data durability.
   - The WAL guarantees that data can be recovered even if there is a system crash.

```plaintext
   +------------------------+
   |        Sample          |
   +------------------------+
   | -> Written to Active   |
   | Chunk in Head Block    |
   | -> WAL (Durable Write) |
   +------------------------+
```

3. **Chunk Full Condition**:
   - A **chunk** is considered full when it either contains **120 samples** or spans a **chunkRange** of 2 hours (default).
   - At this point, a **new chunk** is created, and the full chunk is marked as "complete".

4. **Memory Mapping**:
   - Completed chunks are **flushed to disk** and memory-mapped to reduce memory usage.
   - Only a **reference** to these chunks is kept in memory, dynamically loading them when required.

5. **Compaction**:
   - Once the Head block data spans **chunkRange \* 3/2** (typically 3 hours), it is compacted into a **persistent block**.
   - The older data is removed from the Head block, and a **checkpoint** is created in the WAL.

```plaintext
  +-------------------------------------------------------+
  |         Head Block Lifecycle Flow (ASCII Diagram)     |
  +-------------------------------------------------------+
  |   Sample Ingestion -----> Active Chunk                |
  |                |                 |                   |
  |                v                 v                   |
  |         Write to WAL         Chunk Full Condition    |
  |                                 |                   |
  |                                 v                   |
  |                       Flush to Disk and Memory Map   |
  |                                 |                   |
  |                                 v                   |
  |                       Data Compaction to Persistent  |
  |                          Block and WAL Checkpoint    |
  +-------------------------------------------------------+
```

---

### **3. Write-Ahead Log (WAL) Integration**

- **Purpose**: The WAL records all in-memory data to ensure that no data is lost during crashes.
- **Checkpointing**: As chunks are flushed and compacted, the WAL is **truncated**, and checkpoints are created to clean up older data.
  
#### **Key Features of WAL in TSDB**
- **Durability**: Ensures all in-memory samples can be recovered.
- **Truncation**: Regular truncation helps manage WAL size and prevent it from growing indefinitely.

---

### **4. Chunk Management in Head Block**

#### **Structure of a Chunk**
- **Chunks** are the primary units of data storage in the Head block.
- Each chunk consists of **compressed samples** to optimize storage space.

```plaintext
+---------------------------------------+
|            Chunk Layout               |
+---------------------------------------+
|   Active Chunk (Writable)             |
|   Full Chunks (Read-only, memory-mapped)|
+---------------------------------------+
```

#### **Memory Mapping Strategy**
- **Newly created chunks** are immediately flushed to disk to minimize memory usage.
- Prometheus uses **Operating System memory mapping** to load chunks dynamically when they are accessed.

---

### **5. Compaction and Persistent Blocks**

- **Compaction** merges full chunks from the Head block into **persistent blocks** to create immutable, on-disk data.
- Persistent blocks are further optimized and stored for long-term access.
  
#### **Compaction Process (ASCII Diagram)**
```plaintext
+-----------------------------+
|      Head Block (3h Data)   |
|-----------------------------|
| Chunk 1 | Chunk 2 | Chunk 3 |
+-----------------------------+
   |       |       |
   v       v       v
+---------------------------+
|   Persistent Block        |
+---------------------------+
```

---

### **6. Index and Memory Management**

- **Inverted Index**: Prometheus stores the index of the time series in memory for fast lookups.
- During compaction, any outdated series entries in the index are **garbage collected** to free up space.

#### **Handling Indexes during Compaction**
- The in-memory index is trimmed when the data is compacted into a persistent block.
- Only relevant data that remains in the Head block is kept.

---

### **7. Handling Restarts**

#### **Restart Process**
- During a restart, Prometheus uses **memory-mapped chunks** and **WAL entries** to reconstruct the Head block.
- This ensures that even in the event of a crash, all data can be recovered to its latest state.

---

### **8. Code References and Internal Flow**

- **File Location**: The core logic governing the Prometheus TSDB is located in `tsdb/db.go`.
- **Head Block Code**: Specific logic for handling the Head block can be found in `tsdb/head.go`.
- The code integrates **WAL** and **memory mapping** directly into the chunk ingestion and storage process.

---

### **Conclusion**

This detailed explanation of the **Head block** in Prometheus TSDB lays the groundwork for understanding the entire data lifecycle. The next steps will dive deeper into **WAL checkpointing**, **persistent blocks**, and further optimizations to enhance the efficiency of Prometheus' storage model. Understanding these concepts is essential for developers and contributors looking to optimize or extend Prometheus TSDB's capabilities.