https://ganeshvernekar.com/blog/prometheus-tsdb-wal-and-checkpoint/
---
### **Overview**
This note is a deep dive into the **Write-Ahead Log (WAL)** and **Checkpointing** mechanism used in Prometheus TSDB. The WAL plays a crucial role in ensuring **durability** and **recovery** of time-series data stored in the **Head block** of Prometheus' time-series database (TSDB). Understanding WAL and checkpoints is vital for managing in-memory state and handling graceful or abrupt shutdowns of Prometheus.
### **Table of Contents**
1. **Introduction to WAL in Prometheus TSDB**
2. **WAL Structure and Functionality**
3. **WAL Record Types**
4. **Writing to the WAL**
5. **Checkpointing and WAL Truncation**
6. **Replaying WAL for Data Recovery**
7. **Low-Level Details of WAL Operations**
8. **Code References and Internal Flow**

---
### **1. Introduction to WAL in Prometheus TSDB**
- The **Write-Ahead Log (WAL)** is a sequential log that records all events in the TSDB.
- **Purpose**: Ensures durability and helps in recovery during crashes by replaying the recorded events.
- The WAL is used to restore the in-memory state of the **Head block** upon restarts.
```plaintext
+------------------------------------------------------+
|                    Write-Ahead Log (WAL)             |
+------------------------------------------------------+
| Ensures durability and data recovery                 |
| Records all incoming samples before database writes  |
| Replayable to restore in-memory state during startup |
+------------------------------------------------------+
```
---
### **2. WAL Structure and Functionality**
- **WAL Design**: Composed of sequential files known as **segments**.
- Each segment is capped at **128MiB** to facilitate garbage collection and reduce replay times.
- **Files** are numbered sequentially, ensuring no gaps in the order of events.
#### **WAL Directory Layout Example**
```plaintext
data/
└── wal/
    ├── 000000
    ├── 000001
    └── 000002
```

- **Segment Management**: The segments grow linearly, and older segments are truncated periodically.
---
### **3. WAL Record Types**
Prometheus records three types of entries in the WAL:
1. **Series Record**:
   - Contains label values that identify the time series.
   - Created once for each series to store a unique reference.
2. **Samples Record**:
   - Contains the reference of the time series and its samples.
   - Multiple samples can be recorded in a single write request.
3. **Tombstones Record**:
   - Represents deleted series with specific time ranges.
   - Helps manage data deletions without immediately removing data from memory.
```plaintext
+---------------------------------------+
|           WAL Record Types            |
+---------------------------------------+
| Series Record   -> Series Metadata    |
| Samples Record  -> Data Samples       |
| Tombstones Record -> Deletion Markers |
+---------------------------------------+
```
---
### **4. Writing to the WAL**
- **Order of Operations**:
   - **Series Record** is written to the WAL when a new time series is created.
   - **Samples Record** is appended after series creation to avoid issues during replay.
   - **Tombstones Record** is added before a delete operation to log which data will be removed.
#### **Writing Sequence Example**
1. **New Series**: Write Series Record -> Write Samples Record
2. **Existing Series**: Only write Samples Record
3. **Delete Request**: Write Tombstones Record
```plaintext
+--------------------+
| Write Request Flow |
+--------------------+
| Series Record      |
| Samples Record     |
| Tombstones Record  |
+--------------------+
```
### **5. Checkpointing and WAL Truncation**
#### **Checkpointing Process**
- **Purpose**: Checkpoints ensure that the necessary data is preserved when old WAL segments are truncated.
- **Checkpoint Creation**:
  - Drops series no longer present in the Head.
  - Filters out samples and tombstones that are out of the current time range.
  - Rewrites necessary data to avoid losing important records.
#### **WAL Truncation Example**
- Segments 000000 to 000003 are deleted after creating a checkpoint for segment 000003.

```plaintext
data/
└── wal/
    ├── checkpoint.000003/
    |   ├── 000000
    |   └── 000001
    ├── 000004
    └── 000005
```

- This process ensures data integrity by safely archiving WAL records that are still needed in the checkpoint.
---
### **6. Replaying WAL for Data Recovery**
#### **Replay Mechanism**
- Prometheus replays WAL from the last **checkpoint.X**, continuing with WAL segment **X+1**.
- **Replaying Order**:
  1. **Series Record**: Recreates series in the Head block.
  2. **Samples Record**: Adds data samples to the appropriate series.
  3. **Tombstones Record**: Applies deletions based on previously recorded tombstones.
#### **Replay Flow Diagram**
```plaintext
+--------------------------------------------+
|      WAL Replay Order for Data Recovery    |
+--------------------------------------------+
| 1. Read from last Checkpoint (e.g., 000003)|
| 2. Replay WAL segments from X+1 (000004)   |
| 3. Restore Series, Samples, Tombstones     |
+--------------------------------------------+
```

- **Non-atomic Operations**: Replay may involve reading the additional 2/3rd WAL segments due to non-atomic deletion and checkpoint operations.

---

### **7. Low-Level Details of WAL Operations**

#### **Disk Interaction and Record Handling**
- **Data Pages**: Data is written to disk in **32KiB pages** to minimize write amplification.
- **Record Chaining**:
  - Large records are split across multiple pages with headers to track record continuity.
- **Checksum Verification**:
  - Checksum appended to each record to detect corruption during reads.

#### **Compression in WAL**
- Prometheus uses **Snappy compression** for WAL records to reduce disk usage.
- Compression is optional and can coexist with uncompressed records.

```plaintext
+---------------------------------------+
|      Low-Level WAL Data Management    |
+---------------------------------------+
| 32KiB Page Writes                     |
| Record Chaining and Continuity        |
| Checksum for Integrity Check          |
| Snappy Compression (Optional)         |
+---------------------------------------+
```

---

### **8. Code References and Internal Flow**

- **WAL Logic and Disk Operations**: Located in `tsdb/wal/wal.go`.
  - Responsible for record writes, segmentation, and reading operations.
  
- **Record Encoding and Decoding**: Defined in `tsdb/record/record.go`.
  - Contains the logic for transforming records into byte streams and back.

- **Checkpoint Management**: Handled in `tsdb/wal/checkpoint.go`.
  - Manages the creation, storage, and replay of checkpoint data.

- **Head Block Integration**: Implemented in `tsdb/head.go`.
  - Coordinates between the Head block, WAL, and checkpointing processes.

---

### **Conclusion**

Understanding WAL and checkpointing in Prometheus TSDB is crucial for ensuring data durability and efficient recovery during restarts. The WAL mechanism in Prometheus offers a robust way to handle in-memory data in a fault-tolerant manner, and proper checkpointing ensures that old data is safely archived without impacting performance. In the next part of this series, we will focus on **persistent blocks, compaction strategies**, and **data retention** mechanisms.