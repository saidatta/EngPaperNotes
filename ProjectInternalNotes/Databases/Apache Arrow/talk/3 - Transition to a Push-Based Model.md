https://www.youtube.com/watch?v=bZOvAKGkzpQ

Deep insights into **DuckDB’s push-based execution model, parallelism handling, storage architecture, and concurrency (MVCC)**. We incorporate:

Previously, DuckDB used a **pull-based (Vector Volcano)** model. The **catalyst** for switching to push-based was the difficulty of managing parallelism and “long chain” control flow in Volcano. In a push-based model, operators are **classified** as sources, intermediates, or sinks, each with distinct interfaces for parallelism.

#### 1.1 Sources, Sinks, and Intermediates

1. **Source Operators**  
   - Provide data (e.g., table scans, file scans).  
   - **Parallelism-Aware**: They can split work among multiple threads (e.g., partition row groups).

2. **Intermediate Operators**  
   - Transform incoming data (e.g., projections, joins in probe phase).  
   - Often **parallelism-unaware** if they’re purely read-only or stateless.  
   - Example: **Hash Probe** uses an existing hash table; doesn’t need to coordinate parallel writes.

3. **Sink Operators**  
   - Consume incoming data and perform some *finalizing* action (e.g., building a hash table, aggregating into a global structure).  
   - **Parallelism-Aware**: Must coordinate multiple threads writing into shared state (e.g., a shared hash table).  

**Example**:

```plaintext
[ Pipeline 1: Build phase ]         [ Pipeline 2: Probe + Aggregate ]
        (Source)                            (Source)  (Intermediate)    (Sink)
     Table Scan Sales  ->  HashTable Build  <--->  Table Scan Orders ->  Hash Probe -> GroupBy
```

- **Build**: Hash table creation acts as a **sink** in the first pipeline (parallel writes).  
- **Probe**: Hash join probe acts as an **intermediate** in the second pipeline (parallel reads).  
- **GroupBy**: Another **sink** that accumulates results (parallel writes).

---

### 2. Advantages of Push-Based Execution

1. **Centralized Control Flow**  
   - Instead of each operator recursively calling `GetChunk()`, a central scheduler “pushes” vectors to operators.  
   - The **call stack** is no longer the implicit state; operator states (global/local) are *explicitly* tracked.

2. **Parallelism is Cleanly Modeled**  
   - Each source/sink can define how to partition and merge data.  
   - Parallel operators know how to handle concurrency (e.g., building a shared hash table).

3. **Operator Simplicity**  
   - Operators only do local transformations. They don’t “pull” data or manage complex control flow.  
   - Example: A **projection** gets a vector from the scheduler, applies expressions, and pushes the result onward.

4. **Cache & Buffering Between Operators**  
   - It’s easier to insert **small caches** to accumulate data if a filter or other operator outputs only a few rows at a time.  
   - Prevents performance degradation from “tiny vector” propagation (where you’d effectively regress to a row-at-a-time model).

5. **Scan Sharing**  
   - Data from a single source can be pushed to multiple consumers.  
   - E.g., scanning the same table for multiple aggregates can push the scanned chunks to each aggregator concurrently.

6. **Pausing & Resuming**  
   - Because the execution state is tracked centrally (not on the call stack), we can **pause** a pipeline if buffers are full or if async I/O is needed, and **resume** later without losing context.  
   - This is crucial for **query cancellation** or for **async I/O** scenarios.

> [!info] **Query Cancellation**  
> In push-based DuckDB, whenever the scheduler hands off a batch of tuples (vectors) to an operator, it returns control to a central loop. If a cancel signal arrives, we can terminate the pipeline *before* the next batch is processed.

---

### 3. Parallel Execution and Synchronization

- **Global State vs. Local State**:  
  - A **global state** may store shared data structures (e.g., a global hash table).  
  - Each **thread**/**task** has its own **local state**, merging or finalizing into the global state when complete.
- **Synchronization**:  
  - If multiple threads build a shared hash table, concurrency is managed inside that sink operator.  
  - Buffer managers or other shared services (like the block manager) might also require careful synchronization.

> [!warning] **Potential Bottlenecks**  
> If a sink (e.g., building a massive shared hash table) forces many threads to synchronize on the same lock, performance may degrade. DuckDB’s design tries to limit contention by partitioning data or using lock-free structures where possible.

---

### 4. Storage Architecture

#### 4.1 Single-File Block Storage

DuckDB uses a **single file** for storing the entire database, plus a separate **write-ahead log (WAL)** file. Each file block is **fixed-size** (256 KB). This design:

- Prevents internal fragmentation by always dealing with uniform blocks.  
- Allows re-use of freed blocks when data is deleted or row groups are dropped (though the file might not shrink yet).

**Two Headers**:  
- The database file has **two headers** at the front. This allows DuckDB to write a new version in place, then “flip” which header is the active version atomically (for crash safety).

**Row Groups**:  
- Tables are divided into **row groups**, typically containing around ~120k rows each.  
- Row groups are the unit of parallel scan (e.g., each thread can scan a different row group) and also the unit of *incremental checkpointing*.

```plaintext
+----------------------------------------------------------+
|                     DuckDB File                          |
+----------------------------------------------------------+
| File Header 1 | File Header 2 | Block 1 | Block 2 | ...  |
+----------------------------------------------------------+
                  ^               ^         ^
                  |               |         |
                256KB          256KB     256KB
```

> [!tip] **Incremental Checkpoints**  
> Instead of rewriting the entire table on an update or delete, DuckDB can rewrite *only* the affected row groups. The headers are then flipped to point to the new version.

---

#### 4.2 Write Ahead Log

- **WAL** is stored in a separate file.  
- **Checkpointing** periodically merges WAL changes into the main database file to keep it consistent and reduce WAL size.  
- **Bulk Load Optimization**: For large inserts, DuckDB may skip the WAL entirely and directly compress/write to the main file. This prevents massive WAL overhead for bulk loading.

---

### 5. MVCC and In-Memory Versioning

- **MVCC (Multi-Version Concurrency Control)** in DuckDB is **in-memory** only.
  - Old row versions (for uncommitted or concurrent transactions) are not stored in the main file.  
  - Once the process restarts, uncommitted transactions vanish.
- **Benefits**:  
  - Faster transactional updates because version chains stay in memory.  
  - Simpler on-disk format since we don’t store old versions in the file.

---

### 6. Compression

#### 6.1 Rationale

- Compression can reduce storage size *and* improve I/O performance (smaller reads).  
- Columnar layout amplifies the benefits, as identical or similar data in each column compresses well.  
- DuckDB even **stores compressed data in memory** (for columns not actively used), further benefiting from lighter in-memory footprint.

#### 6.2 General-Purpose vs. Lightweight

1. **General-Purpose**: e.g., zstd, gzip, Snappy, LZ4  
   - Simple to apply (no data-type knowledge).  
   - Potentially higher decompression overhead.  
   - Typically bulk decompression (less random access).

2. **Lightweight, Data-Aware**: e.g., RLE, dictionary encoding, bit-packing, frame-of-reference, FSST (Fast String Search Transforms)  
   - Faster decompress or *even direct execution on compressed data*.  
   - Must detect specific patterns (e.g., repeated values, low cardinality).  
   - Per **column** or **row group** basis: analyze phase → compress phase.

**Implementation**:  
- For each column in a row group, DuckDB chooses from a suite of compression methods (e.g., RLE, dictionary, FSST, bit-packing).  
- **Analyze Phase**: Inspect data, estimate compression ratio.  
- **Compress Phase**: Write compressed blocks to disk.

```plaintext
+-----------------------+
|  RowGroup (120k rows) |
+-----------+-----------+
| Column A  | Column B  |
|  ...      |  ...      |
+-----------+-----------+
     ↓            ↓
  Compress     Compress
(choose best) (choose best)
     ↓            ↓
 Disk Blocks  Disk Blocks
```

---

### 7. Buffer Manager

- **Similar to LeanStore** design, focusing on minimal global locks and maximizing parallel access.
- Each block of 256 KB can be **pinned** in memory by threads. Freed blocks can be reused for new data if rows or row groups are dropped.
- **Not** strictly LRU in a centralized sense (that would cause heavy locking). Instead, uses local queues or approximate LRU.

> [!important]  
> As core counts grow, a naive centralized buffer manager would become a bottleneck. DuckDB’s design strives to partition or localize buffer state to reduce contention.

---

### 8. Query Cancellation & Pausing Pipelines

Since push-based scheduling is centralized:
- **After processing each vector**, the operator returns control to a main scheduler loop.  
- The system checks if a *cancellation flag* is set. If so, it **terminates** the query gracefully.  
- Similarly, if a sink’s output buffer is full, the pipeline can **pause** until space frees up, without losing the operator’s state.

---

### 9. Example Pseudocode: Push-Based Hash Build

A simplified illustration:

```cpp
class HashBuildSink : public SinkOperator {
public:
    void InitializeGlobalState() override {
        global_hash_table = make_unique<HashTable>();
    }

    void InitializeLocalState() override {
        local_hash_table = make_unique<HashTable>();
    }

    // The scheduler pushes chunks of data to this method
    void Sink(Chunk &input) override {
        local_hash_table->Build(input);
    }

    // When pipeline ends or local states finalize
    void CombineLocalState() override {
        global_hash_table->Merge(*local_hash_table);
    }

private:
    unique_ptr<HashTable> global_hash_table;
    unique_ptr<HashTable> local_hash_table;
};
```

- **Scheduler** repeatedly invokes `Sink()` with new data chunks from a table scan (the source).  
- Each thread has a **local hash table**, merged into the **global** one at the end.  
- No “pulling” from child operators; data simply arrives (“pushed”) from the scan pipeline.

---

### 10. Summary & Key Takeaways

1. **Push-Based Execution**:
   - Central scheduling, easier parallel orchestration, simpler operator code, and robust support for pause/resume/cancellation.
2. **Storage**:
   - Single-file design with block-based layout (256 KB blocks).  
   - Row groups as the checkpoint and parallel scan unit (~120k rows each).
3. **In-Memory MVCC**:
   - Concurrency control doesn’t bloat the disk file with old versions.  
   - Simpler on-disk format, but less suited for very long-running transactions that must survive a crash.
4. **Compression**:
   - Both data-type–specific (lightweight) and general-purpose approaches.  
   - Automatic selection of the best method per column per row group.  
   - Increases performance as well as decreasing storage footprint.
5. **Buffer Management**:
   - Minimal locking, partial LRU or similar policies.  
   - Aim for high parallel throughput.

---

### 11. Further Reading

- [DuckDB GitHub](https://github.com/duckdb/duckdb) (see `src/execution/pipeline` for push-based pipeline code).  
- [LeanStore Paper](https://db.in.tum.de/~leis/papers/leanstore.pdf) for buffer management inspiration.  
- [Umbra Paper (Neumann et al.)](http://cidrdb.org/cidr2020/papers/p18-neumann-cidr20.pdf) - covers string compression, morsel-driven parallelism.  
- [MonetDB/X100 Papers](https://www.monetdb.org/documentation/dev) for historical background on vectorized engines.  

---

**End of Notes**