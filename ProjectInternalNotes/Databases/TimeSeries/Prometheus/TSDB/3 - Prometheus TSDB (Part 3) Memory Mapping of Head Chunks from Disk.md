https://ganeshvernekar.com/blog/prometheus-tsdb-persistent-block-and-its-index
---

### **Overview**

This note is a deep dive into the **memory mapping of Head chunks** in Prometheus TSDB. This approach optimizes both memory usage and performance in Prometheus by efficiently managing how full chunks are stored and accessed. Memory mapping is crucial for speeding up the **Write-Ahead Log (WAL)** replay and reducing the memory footprint of the **Head block**. 

### **Table of Contents**
1. **Introduction to Memory Mapping in Prometheus TSDB**
2. **Chunk Life Cycle in the Head Block**
3. **Chunk Format on Disk**
4. **Memory-Mapped Chunk Access**
5. **WAL Replay with Memory-Mapped Chunks**
6. **Performance Enhancements**
7. **Garbage Collection and File Management**
8. **Code References and Internal Flow**

---

### **1. Introduction to Memory Mapping in Prometheus TSDB**

- **Memory mapping** in Prometheus TSDB involves storing full chunks on disk and mapping them into memory as required.
- This mechanism reduces the memory footprint of the **Head block** and optimizes the WAL replay during Prometheus startup.
- The immutability of these chunks ensures they can be efficiently read from disk without modification.

```plaintext
+------------------------------------------------------+
|               Memory Mapping in TSDB                 |
+------------------------------------------------------+
| Reduces memory usage by offloading chunks to disk    |
| Enhances performance by reducing replay time         |
| Ensures immutability for efficient chunk access      |
+------------------------------------------------------+
```

---

### **2. Chunk Life Cycle in the Head Block**

- **Full Chunks**: When a chunk reaches the limit of **120 samples** or spans a **2-hour range**, it is marked as full and flushed to disk.
- **Immutability**: The chunk becomes read-only, ensuring efficient retrieval.
- **New Chunks**: A new chunk is created to continue ingesting incoming samples.

```plaintext
+------------------------------+
|         Chunk States         |
+------------------------------+
| Active Chunk (Writable)      |
| Full Chunk (Read-Only)       |
+------------------------------+
```

#### **Visual Representation**
```plaintext
┌────────────────────┬───────────────────┐
│     Active Chunk   │   Full Chunk      │
│     (Writable)     │  (Memory-mapped)  │
└────────────────────┴───────────────────┘
```

---

### **3. Chunk Format on Disk**

#### **File Structure**
- The directory structure for **chunks_head** is similar to the WAL but starts from **000001**.
- Each file has a maximum size of **128MiB** and begins with an 8-byte header.

#### **Chunk File Layout**
```plaintext
data/
├── chunks_head/
│   ├── 000001
│   └── 000002
└── wal/
    ├── checkpoint.000003/
    ├── 000004
    └── 000005
```

#### **File Header Details**
```plaintext
┌──────────────────────────────┐
│  magic(0x0130BC91) <4 byte>  │
├──────────────────────────────┤
│    version(1) <1 byte>       │
├──────────────────────────────┤
│    padding(0) <3 byte>       │
└──────────────────────────────┘
```
- **Magic Number**: Uniquely identifies the file.
- **Version**: Indicates the chunk format.
- **Padding**: Reserved for future header extensions.

#### **Chunk Data Structure**
```plaintext
┌─────────────────────┬───────────────────────┬───────────────────────┬───────────────────┬───────────────┬──────────────┬────────────────┐
| series ref <8 byte> | mint <8 byte, uint64> | maxt <8 byte, uint64> | encoding <1 byte> | len <uvarint> | data <bytes> │ CRC32 <4 byte> │
└─────────────────────┴───────────────────────┴───────────────────────┴───────────────────┴───────────────┴──────────────┴────────────────┘
```
- **Series Ref**: Reference to the series in memory.
- **mint** / **maxt**: Minimum and maximum timestamps of the chunk.
- **CRC32**: Checksum for verifying data integrity.

---

### **4. Memory-Mapped Chunk Access**

- **Reference Mechanism**: Uses a reference consisting of file number and byte offset to locate the chunk on disk.
- **Efficient Access**: Uses memory mapping to access only the required portion of the chunk without loading the entire file into memory.

#### **Reference Encoding Example**
- Suppose the chunk is in file **000093** at offset **1234**:
  - Reference: `(93 << 32) | 1234`
  - **First 4 bytes** indicate the file number, **last 4 bytes** indicate the byte offset.

```plaintext
┌──────────────┬───────────────┐
│  File Number │  Byte Offset  │
├──────────────┼───────────────┤
│     93       │     1234      │
└──────────────┴───────────────┘
```

---

### **5. WAL Replay with Memory-Mapped Chunks**

- **Startup Process**:
  - Iterates through all chunks in the `chunks_head` directory and builds a map of series references.
  - Uses memory-mapped chunks to skip samples already covered by disk-based chunks, speeding up replay.

#### **Modified WAL Replay Flow**
1. **Series Record**: Check and attach memory-mapped chunks to the series.
2. **Samples Record**: Skip samples if they fall within the range of memory-mapped chunks.
3. **Tombstones Record**: Apply tombstone deletions as needed.

```plaintext
+--------------------------------------+
|        Modified WAL Replay           |
+--------------------------------------+
| 1. Attach Memory-Mapped Chunks       |
| 2. Skip Covered Samples              |
| 3. Apply Tombstones Efficiently      |
+--------------------------------------+
```

---

### **6. Performance Enhancements**

#### **Memory Savings**
- **In-Memory Storage**: Reduces from 120-200 bytes to only **24 bytes** (8 bytes each for chunk reference, mint, and maxt).
- **Real-World Impact**: Achieves 15-50% reduction in memory usage depending on data churn.

#### **Faster Startup**
- **Reduced Replay Time**: Speeds up startup by 15-30% by avoiding chunk recreation from individual samples.
- **Efficient Data Access**: Memory-mapped chunks ensure faster iteration over historical data.

```plaintext
+---------------------------------------------+
|            Performance Enhancements         |
+---------------------------------------------+
| Memory Savings: 15-50%                      |
| Faster Startup: Reduced replay by 15-30%    |
+---------------------------------------------+
```

---

### **7. Garbage Collection and File Management**

#### **Garbage Collection Process**
- **Head Truncation**: Removes references to chunks older than truncation time **T**.
- **Disk Cleanup**: Deletes files with all chunks older than **T** while maintaining sequence.

#### **Rotating Live Files**
- Closes the live chunk file and opens a new one to aid in timely file deletions.

```plaintext
+-------------------------------------------------+
|   Garbage Collection and File Management        |
+-------------------------------------------------+
| Removes outdated chunks beyond truncation time  |
| Deletes old memory-mapped files systematically  |
+-------------------------------------------------+
```

---

### **8. Code References and Internal Flow**

- **Chunk Management**: `tsdb/chunks/head_chunks.go`
  - Implements disk writing, memory-mapping, file handling, and chunk iteration.
  
- **Head Integration**: `tsdb/head.go`
  - Coordinates memory-mapping of chunks and uses head_chunks.go as a black box.
  
- **WAL Handling**: Coordinates the WAL replay process using the mapped chunks to optimize memory and startup performance.

```plaintext
+------------------------------------------------+
|               Code References                  |
+------------------------------------------------+
| Chunk Management: tsdb/chunks/head_chunks.go   |
| Head Integration: tsdb/head.go                 |
| WAL Handling: Uses optimized replay strategies |
+------------------------------------------------+
```

---

### **Conclusion**

Memory mapping of chunks in Prometheus TSDB is a powerful technique to balance memory usage and performance. This method reduces the memory footprint of the Head block while speeding up WAL replay during restarts. By offloading chunks to disk and accessing them on-demand, Prometheus enhances both data processing efficiency and system stability. The next parts in this series will explore **persistent blocks, data retention**, and **compaction strategies**.