https://ganeshvernekar.com/blog/prometheus-tsdb-queries
---
### **Overview**
This note provides an in-depth look at **persistent blocks** in the Prometheus TSDB (Time-Series Database) and their structure, covering everything from metadata and chunk files to the internal workings of the index and tombstones. This is Part 4 in the Prometheus TSDB series, following the exploration of **Head blocks**, **WAL (Write-Ahead Log)**, and **memory-mapped chunks**.

---
### **Table of Contents**

1. **Introduction to Persistent Blocks**
2. **Creation and Properties of Persistent Blocks**
3. **Components of a Persistent Block**
4. **Detailed Examination of Block Components**
   - A. **Meta File (`meta.json`)**
   - B. **Chunks Directory**
   - C. **Index File**
   - D. **Tombstones**
5. **In-Depth Index Structure**
   - A. **TOC (Table of Contents)**
   - B. **Symbol Table**
   - C. **Series Information**
   - D. **Label Index and Label Offset Table**
   - E. **Postings and Postings Offset Table**
6. **Code References and Internal Flow**

---

### **1. Introduction to Persistent Blocks**

- **Persistent blocks** in Prometheus TSDB store immutable data on disk, providing durability and efficient access to historical time-series data.
- These blocks are created when the **Head block** exceeds a certain data range and are stored separately from in-memory data.
- Each block has a unique **ULID (Universally Unique Lexicographically Sortable Identifier)**.

```plaintext
+------------------------------------------------------+
|             Key Properties of Persistent Blocks      |
+------------------------------------------------------+
| Immutable data storage                               |
| Efficient querying via indexes                       |
| Unique ULID for identification                       |
| Compaction process merges older blocks               |
+------------------------------------------------------+
```

---

### **2. Creation and Properties of Persistent Blocks**

- **Creation Triggers**:
  - A new block is created when the **Head block** contains data spanning **chunkRange*3/2**.
  - The **first chunkRange** of data from the Head block is converted into a persistent block (default span: **2 hours**).

- **Block Compaction**:
  - Older blocks are periodically merged into larger blocks to optimize storage and query performance.
  - Compaction is essential to manage disk space and to reduce the number of small blocks.

```plaintext
+-------------------------+
|      Block Lifecycle    |
+-------------------------+
| Created from Head block |
| Immutable data storage  |
| Compacted over time     |
+-------------------------+
```

---

### **3. Components of a Persistent Block**

A persistent block consists of the following components:

1. **meta.json**: Contains metadata about the block.
2. **chunks (directory)**: Stores the raw data chunks.
3. **index (file)**: Maintains the index for efficient querying.
4. **tombstones (file)**: Holds deletion markers.

#### **Directory Structure Example**

```plaintext
data/
├── 01EM6Q6A1YPX4G9TEB20J22B2R/     <- Block ID (ULID)
│   ├── chunks/
│   │   ├── 000001
│   │   └── 000002
│   ├── index
│   ├── meta.json
│   └── tombstones
└── wal/
    ├── checkpoint.000003/
    ├── 000004
    └── 000005
```

---

### **4. Detailed Examination of Block Components**

#### **A. Meta File (`meta.json`)**

- **Purpose**: Holds metadata information about the block.
- **Key Fields**:
  - `ulid`: Unique identifier of the block.
  - `minTime` / `maxTime`: The time range covered by the block.
  - `stats`: Number of samples, series, and chunks in the block.
  - `compaction`: Information about the block's creation and compaction history.

```json
{
    "ulid": "01EM6Q6A1YPX4G9TEB20J22B2R",
    "minTime": 1602237600000,
    "maxTime": 1602244800000,
    "stats": {
        "numSamples": 553673232,
        "numSeries": 1346066,
        "numChunks": 4440437
    },
    "compaction": {
        "level": 1,
        "sources": [
            "01EM65SHSX4VARXBBHBF0M0FDS",
            "01EM6GAJSYWSQQRDY782EA5ZPN"
        ]
    },
    "version": 1
}
```

#### **B. Chunks Directory**

- **Structure**: Contains chunk files named sequentially (e.g., 000001, 000002).
- **File Format**:
  - Header with `magic number`, `version`, and `padding`.
  - List of chunks with each chunk containing:
    - `len`: Length of the chunk.
    - `encoding`: Compression type.
    - `data`: Compressed data.
    - `CRC32`: Checksum for integrity.

```plaintext
┌──────────────────────────────┐
│  magic(0x85BD40DD) <4 byte>  │
├──────────────────────────────┤
│    version(1) <1 byte>       │
├──────────────────────────────┤
│    padding(0) <3 byte>       │
├──────────────────────────────┤
│ ┌──────────────────────────┐ │
│ │         Chunk 1          │ │
│ ├──────────────────────────┤ │
│ │          ...             │ │
│ └──────────────────────────┘ │
└──────────────────────────────┘
```

#### **C. Index File**

- **Function**: The index is an inverted index that maps label names and values to series IDs, facilitating fast lookups.
- **Components**:
  - **Symbol Table**: Holds all unique label values.
  - **Series Information**: Lists all series in the block.
  - **Postings**: Maps label-value pairs to series IDs.
  - **TOC (Table of Contents)**: Guides access to various sections in the index.

#### **D. Tombstones**

- **Purpose**: Holds deletion markers for series data.
- **File Format**:
  - **series ref**: Reference to the series.
  - **mint** and **maxt**: Time range of data to be excluded during queries.

```plaintext
┌────────────────────────┬─────────────────┬─────────────────┐
│ series ref <uvarint64> │ mint <varint64> │ maxt <varint64> │
└────────────────────────┴─────────────────┴─────────────────┘
```

---

### **5. In-Depth Index Structure**

#### **A. TOC (Table of Contents)**

- **Purpose**: Provides byte offsets to the major sections of the index.
- **Structure**: Lists offsets for the **Symbol Table**, **Series Information**, **Label Offset Table**, and more.

```plaintext
┌─────────────────────────────────────────┐
│ ref(symbols) <8b>                       │ -> Symbol Table
├─────────────────────────────────────────┤
│ ref(series) <8b>                        │ -> Series
├─────────────────────────────────────────┤
│ ref(label indices start) <8b>           │ -> Label Index 1
├─────────────────────────────────────────┤
│ ref(postings start) <8b>                │ -> Postings 1
└─────────────────────────────────────────┘
```

#### **B. Symbol Table**

- **Function**: Stores deduplicated strings found in label pairs.
- **Format**:
  - List of symbols, each with its length and UTF-8 encoded string.
  - Reduces index size by referring to symbols instead of raw strings.

```plaintext
┌────────────────────┬─────────────────────┐
│ len <4b>           │ #symbols <4b>       │
├────────────────────┴─────────────────────┤
│ ┌──────────────────────┬───────────────┐ │
│ │ len(str_1) <uvarint> │ str_1 <bytes> │ │
└────────────────────────┴───────────────┘
```

#### **C. Series Information**

- **Purpose**: Contains metadata about series, including label sets and chunk references.
- **Chunk Metadata**: Includes timestamps and references for efficient access.

```plaintext
┌──────────────────────────────────────────────────────┐
│ len <uvarint>                                        │
├──────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────┐ │
│ │            labels count <uvarint64>              │ │
│ ├──────────────────────────────────────────────────┤ │
│ │  ┌────────────────────────────────────────────┐  │ │
│ │  │ ref(l_i.name) <uvarint32>                  │  │ │
└──────────────────────────────────────────────────┘
```

#### **D. Label Index and Label Offset Table**

- **Label Index**: Maps possible label values.
- **Label Offset Table**: Maps label names to values in the Label Index.

#### **E. Postings and Postings Offset Table**

- **Postings**: Lists series IDs for specific label-value pairs.
- **Postings Offset Table**: Links

 label-value pairs to their postings list.

```plaintext
┌─────────────────────┬──────────────────────┐
│ len <4b>            │ #entries <4b>        │
├─────────────────────┴──────────────────────┤
│ ┌────────────────────────────────────────┐ │
│ │ ref(series_1) <4b>                     │ │
│ ├────────────────────────────────────────┤ │
│ │ ref(series_n) <4b>                     │ │
└──────────────────────────────────────────┘
```

---

### **6. Code References and Internal Flow**

- **Meta File Handling**: `tsdb/block.go`
- **Chunk Management**: `tsdb/chunks/chunks.go`
- **Index File Operations**: `tsdb/index/index.go`
- **Tombstones Processing**: `tsdb/tombstones/tombstones.go`

These files contain implementations for creating, reading, and managing blocks within Prometheus TSDB.

```plaintext
+-----------------------------------------------------+
|              Code References and Flow               |
+-----------------------------------------------------+
| Meta File: tsdb/block.go                            |
| Chunk Files: tsdb/chunks/chunks.go                  |
| Index Management: tsdb/index/index.go               |
| Tombstones: tsdb/tombstones/tombstones.go           |
+-----------------------------------------------------+
```

---

### **Conclusion**

Persistent blocks in Prometheus TSDB play a crucial role in ensuring efficient storage, durability, and querying of historical data. The intricate design of the **index**, combined with the **chunk** and **tombstone** structures, enables Prometheus to maintain high performance even under large-scale data ingestion and querying conditions. Further exploration of **queries and compaction** will provide insights into the broader functionality of these blocks in Prometheus.