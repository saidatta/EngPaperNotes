https://ganeshvernekar.com/blog/prometheus-tsdb-compaction-and-retention
---
### **Overview**
This detailed note explores the **compaction** and **retention** processes in **Prometheus TSDB**, focusing on how data blocks are merged, optimized, and maintained to ensure efficient disk usage and performant queries. This information is critical for understanding how **Prometheus TSDB** efficiently handles large datasets over time.

---

### **Table of Contents**

1. **Introduction to Compaction and Retention**
2. **Compaction Process**
   - A. **Why Compaction is Necessary**
   - B. **Steps in Compaction**
   - C. **Detailed Analysis of Compaction Planning**
   - D. **Head Compaction**
3. **Retention Policies**
   - A. **Time-based Retention**
   - B. **Size-based Retention**
4. **Implementation Details and Code References**

---

### **1. Introduction to Compaction and Retention**

- **Compaction** and **retention** are background processes in **Prometheus TSDB** that ensure the database uses disk space efficiently and queries remain performant.
- **Compaction** merges smaller blocks into larger blocks, optimizing data storage.
- **Retention** controls how long data is kept or how much disk space it consumes.

```plaintext
+---------------------------------------------+
| Key Concepts of Compaction and Retention    |
+---------------------------------------------+
| Compaction: Merging smaller data blocks     |
| Retention: Managing data lifespan or size   |
| Background processes ensure efficiency      |
+---------------------------------------------+
```

---

### **2. Compaction Process**

#### **A. Why Compaction is Necessary**

1. **Tombstone Management**: Removes data marked for deletion (tombstones), reducing disk usage.
2. **Index Deduplication**: Merges adjacent blocks to eliminate duplicate index entries, saving disk space.
3. **Query Optimization**: Reduces the overhead of querying multiple blocks by merging them into fewer blocks.
4. **Handling Overlapping Blocks**: Combines overlapping blocks to avoid expensive deduplication during queries.

```plaintext
+----------------------------------------------+
| Why Compaction Matters                      |
+----------------------------------------------+
| Removes tombstone data                      |
| Deduplicates index entries                  |
| Optimizes query performance                 |
| Combines overlapping blocks                 |
+----------------------------------------------+
```

#### **B. Steps in Compaction**

- **Step 1**: The "Plan" — Select blocks for compaction based on conditions.
- **Step 2**: The actual **compaction** — Merge the selected blocks into a new block.

#### **C. Detailed Analysis of Compaction Planning**

##### **Step 1: The Plan**

- **Condition 1: Overlapping Blocks**
  - Compaction prioritizes merging overlapping blocks to maintain non-overlapping state.
  - Example of planning overlapping blocks:
    ```plaintext
    |---1---|
            |---2---|
      |---3---|
                  |---4---|
    ```
    - The plan selects [1, 2, 3, 4] for compaction to address all overlaps in one pass.

- **Condition 2: Preset Time Ranges**
  - Blocks are merged to fit into predefined time ranges: [2h, 6h, 18h, 54h, 162h, 486h].
  - Ensures no bucket has more than one block for the same time range.

- **Condition 3: Tombstones Threshold**
  - Blocks with tombstones affecting more than 5% of their series are compacted to remove stale data.
  - Only one block is selected in this case.

```plaintext
+----------------------------------------------+
| Compaction Planning Conditions               |
+----------------------------------------------+
| 1. Overlapping blocks                        |
| 2. Preset time ranges (e.g., 2h, 6h, etc.)   |
| 3. Tombstones affecting >5% of series        |
+----------------------------------------------+
```

##### **Step 2: The Compaction Itself**

- The compaction process writes a **new block** from the source blocks.
- Merges series in **sorted order**, deduplicating data if necessary.
- Assigns a **compaction level** to the new block, indicating its generation.

```plaintext
┌──────────────────────┐
│ Compaction Process   │
├──────────────────────┤
│ Merges series data   │
│ Deduplicates samples │
│ Creates new block    │
└──────────────────────┘
```

- **Handling Overlapping Blocks**:
  - **Overlapping chunks** are uncompressed, deduped, and recompressed.
  - **Non-overlapping chunks** are simply concatenated.

#### **D. Head Compaction**

- The **Head block** is also compacted periodically to persist its data into new blocks.
- Implements the same compaction logic as persistent blocks but operates on live data.

```plaintext
┌──────────────────────────────┐
│ Head Compaction Process      │
├──────────────────────────────┤
│ Persists live data to blocks │
│ Uses same compaction logic   │
└──────────────────────────────┘
```

---

### **3. Retention Policies**

Retention policies manage how long data is kept in the TSDB or how much space it consumes.

#### **A. Time-based Retention**

- **Definition**: Deletes data older than a specified time threshold.
- **Mechanism**: Removes blocks completely beyond the retention period based on the latest block's time.
- Example: If retention is set to 15 days, blocks older than 15 days from the latest block are deleted.

```plaintext
┌─────────────────────┐
│ Time-based Retention│
├─────────────────────┤
│ Removes old blocks  │
│ Based on retention  │
└─────────────────────┘
```

#### **B. Size-based Retention**

- **Definition**: Deletes blocks when the total disk usage exceeds a specified size.
- **Mechanism**: Considers WAL, checkpoints, m-mapped chunks, and blocks in the size calculation.
- **Behavior**: Even if the TSDB exceeds the limit due to WAL or checkpoint, only blocks are deleted.

```plaintext
┌──────────────────────┐
│ Size-based Retention │
├──────────────────────┤
│ Manages disk usage   │
│ Deletes old blocks   │
└──────────────────────┘
```

### **4. Implementation Details and Code References**

#### **Compaction Code Structure**

- **Compaction Plan**: `tsdb/compact.go`
  - Handles the creation of compaction plans and the merging of blocks.
- **Chunk Merging Logic**: `storage/merge.go`
  - Code for concatenating and merging chunks from multiple blocks.
- **Compaction Cycle Management**: `tsdb/db.go`
  - Controls the periodic initiation of the compaction cycle.
  - Includes logic for the compaction of the Head block and block deletions.

#### **Retention Code Structure**

- **Retention Policies**: Managed in `tsdb/db.go`
  - Time-based and size-based retention logic for deleting blocks.

```plaintext
+------------------------------------------------------+
|                Code Reference Summary                |
+------------------------------------------------------+
| Compaction Plan Creation:     tsdb/compact.go        |
| Chunk Merging Logic:          storage/merge.go       |
| Compaction Cycle Management:  tsdb/db.go             |
| Retention Policy Handling:    tsdb/db.go             |
+------------------------------------------------------+
```

---

### **Conclusion**

Compaction and retention are critical processes that ensure **Prometheus TSDB** operates efficiently, optimizing disk usage and query performance. Understanding the mechanics of these processes, including their planning, execution, and retention strategies, provides deeper insight into how Prometheus handles growing datasets effectively. This knowledge is essential for maintaining scalable and performant monitoring infrastructures.