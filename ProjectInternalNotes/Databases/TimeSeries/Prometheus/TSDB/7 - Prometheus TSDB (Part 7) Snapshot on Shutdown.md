https://ganeshvernekar.com/blog/prometheus-tsdb-snapshot-on-shutdown
---
### **Overview**
This detailed note covers the snapshot feature in **Prometheus TSDB**, introduced in **v2.30.0**, which optimizes restart times by avoiding the need to replay the **Write-Ahead-Log (WAL)**. We will explore how snapshots are created, their structure, and how they enhance performance during shutdowns and restarts.
### **Table of Contents**
1. **Introduction**
2. **Understanding Snapshots**
   - A. **Snapshot Structure**
   - B. **Snapshot File Format**
3. **Restoring In-Memory State from Snapshots**
4. **Faster Restarts**
5. **Important Considerations**
6. **Implementation Details and Code References**

---
### **1. Introduction**

- **Problem Statement**: WAL replay on restart can be time-consuming, especially at scale.
- **Solution**: Use **snapshots** to speed up restarts by skipping most of the WAL replay process.
- **Snapshot Activation**: Enabled via `--enable-feature=memory-snapshot-on-shutdown`.

```plaintext
+----------------------------------------------------+
| Prometheus Snapshot Overview                       |
+----------------------------------------------------+
| Solves slow restart issues by skipping WAL replay  |
| Contains in-memory data at shutdown time          |
| Enabled via a configuration flag                  |
+----------------------------------------------------+
```

---

### **2. Understanding Snapshots**

#### **A. Snapshot Structure**

- **Snapshot Definition**: A **read-only static view** of in-memory data in TSDB.
- **Components**:
  1. **Time series** and **in-memory chunks** (except the last chunk).
  2. **Tombstones** present in the Head block.
  3. **Exemplars** from the Head block.

- **Naming Convention**: `chunk_snapshot.X.Y`
  - `X`: Last WAL segment number.
  - `Y`: Byte offset within the segment.

```plaintext
+------------------------------------------+
| Snapshot Components                      |
+------------------------------------------+
| Time series with in-memory chunks        |
| Tombstones in the Head block             |
| Exemplars in the Head block              |
+------------------------------------------+
| Named as chunk_snapshot.X.Y              |
+------------------------------------------+
```

#### **B. Snapshot File Format**

- **Record Order** in snapshot files:
  1. **Series Records**: Contains metadata and in-memory chunk data.
  2. **Tombstones Record**: Represents deletion markers.
  3. **Exemplar Records**: Stored in order, based on their addition to the circular buffer.

```plaintext
data
├── chunk_snapshot.000005.12345
|   ├── 000001
|   └── 000002
└── wal
   ├── checkpoint.000003
   ├── 000004
   └── 000005
```

- **Snapshot Storage**: Stored in sequence with references to WAL segments for future replay if required.

---

### **3. Restoring In-Memory State from Snapshots**

#### **Procedure to Restore State:**

1. **Load Memory-Mapped Chunks**:
   - Use memory-mapped chunks to build the series map.

2. **Process Series Records from Snapshot**:
   - Reconstruct each series in memory using labels and in-memory chunk data from the snapshot.

3. **Restore Tombstones and Exemplars**:
   - Read tombstone and exemplar records to rebuild deletion markers and exemplars.

4. **WAL Replay (if required)**:
   - Replay only the WAL after the segment `X` and offset `Y` if the snapshot was incomplete.

```plaintext
+------------------------------------------------+
| Snapshot Restoration Workflow                  |
+------------------------------------------------+
| 1. Load memory-mapped chunks                   |
| 2. Process series records from snapshot        |
| 3. Restore tombstones and exemplars            |
| 4. Replay partial WAL (if necessary)           |
+------------------------------------------------+
```

- **Efficiency**: Minimizes the need to replay large WAL segments, focusing only on unreplayed portions.

### **4. Faster Restarts**

- **Restart Time Improvement**:
  - Significant reduction in restart time (50-80%) due to skipping WAL replay.
- **Shutdown Overhead**:
  - Snapshot creation can delay shutdown by several seconds but is worth the reduced startup time.

```plaintext
+--------------------------------------------------+
| Faster Restart Benefits                          |
+--------------------------------------------------+
| Skips lengthy WAL replay entirely in most cases  |
| Saves 50-80% of restart time on large datasets   |
| Snapshotting adds minor shutdown delay           |
+--------------------------------------------------+
```

---

### **5. Important Considerations**

#### **Why Snapshots on Shutdown Only?**

- **Write Amplification**: Frequent snapshots would increase the number of times data is written to disk, leading to inefficiency.
- **Best Use Case**: Majority of cases involve graceful shutdowns; hence, taking snapshots only at this time is optimal.

#### **Why WAL is Still Required?**

- **Crash Resilience**: WAL is essential for data recovery in case of crashes, as snapshots are only taken on graceful shutdowns.
- **Remote Write Dependencies**: Remote write relies on WAL for consistent data transfer.

```plaintext
+---------------------------------------------+
| Key Considerations                          |
+---------------------------------------------+
| Snapshots only on shutdown to avoid writes  |
| WAL needed for crash recovery and durability|
+---------------------------------------------+
```

### **6. Implementation Details and Code References**

#### **Core Components and Code Structure**

- **Snapshot and Restore Logic**: `tsdb/head_wal.go`
  - Manages snapshot creation during shutdown.
  - Handles restoration from snapshots during startup.

```plaintext
+------------------------------------------------+
|                Code References                 |
+------------------------------------------------+
| Snapshot Logic: tsdb/head_wal.go               |
| Restoration Workflow: tsdb/head_wal.go         |
+------------------------------------------------+
```

---

### **Conclusion**

The snapshot feature in **Prometheus TSDB** addresses the performance bottleneck of replaying the **WAL** during restarts, providing a significant improvement in restart times. By storing the in-memory state during shutdown and efficiently restoring it on startup, **Prometheus TSDB** becomes more resilient and performant, especially at scale. Understanding the internal mechanics of snapshots, their creation, restoration, and interaction with WAL helps in optimizing **Prometheus deployments** for reliability and speed.