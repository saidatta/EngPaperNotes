
This document provides a *comprehensive*, *PhD-level* walkthrough of **fault-tolerant execution** in Trino. It covers:

1. **Basic Concepts** behind Trino’s fault-tolerant mode  
2. **Configuration** and **Supported Connectors**  
3. **Retry Policies** (QUERY vs. TASK)  
4. **Exchange Manager** setup for spooling  
5. **Advanced Configuration** such as node allocation, task sizing, retry limits, and encryption  

Use these features to enhance Trino’s reliability, especially for *long-running or large-scale* queries.

---

## 1. Overview

By default, if a **Trino node** fails or runs out of resources while executing a query, that query **fails**—and the user must manually rerun it. **Fault-tolerant execution** aims to mitigate such failures by **automatically retrying** queries or tasks in the event of failure. This mechanism relies on *spooled* intermediate data, enabling another worker to pick up the work if a node goes down.

> **Note**: Fault tolerance does *not* fix user or SQL errors (e.g., syntax errors). Trino only retries queries or tasks that fail due to node or resource issues, *not* errors caused by invalid SQL.

### 1.1 Key Points

- **Intermediate Exchange Data** is *spooled* to storage (memory or external).  
- If a node fails, **another** node can reuse the spooled data.  
- Fault tolerance can be turned **on** or **off**, and can be set to *QUERY-level* or *TASK-level* retries.

---

## 2. Configuration

### 2.1 Enabling Fault-Tolerant Execution

In your `config.properties` (or equivalent):

```ini
# Choose a retry policy
retry-policy=QUERY
# or
retry-policy=TASK
```

**Supported Connectors**: Only certain connectors allow fault-tolerant retries (e.g., BigQuery, Hive, Iceberg, PostgreSQL, MySQL, Oracle, etc.). If a connector does *not* implement fault tolerance, queries fail with “This connector does not support query retries.”

### 2.2 Exchange Manager (Recommended for Larger Data)

When the result set or intermediate shuffle data is *larger than 32MB*, you should configure an **exchange manager** that stores spool data externally (e.g., in S3, HDFS) to handle fault tolerance effectively.

---

## 3. Retry Policy

### 3.1 `retry-policy=QUERY`

- **Entire query** is retried if a failure occurs on any node.  
- Typically used for **many small queries** or short interactive queries.  
- By default, queries that produce results **exceeding 32MB** in the final stage do not get fault tolerance. You can raise `exchange.deduplication-buffer-size` or configure an **external** exchange manager to handle larger results.

**Pros**: Simpler logic, good for short queries.  
**Cons**: If a query is *large*, re-running the entire query can be expensive.

### 3.2 `retry-policy=TASK`

- Only **individual tasks** (parts of a query) are retried upon failure, rather than the entire query.  
- **Requires** an exchange manager for spooling.  
- Best for **large batch queries**, as it saves time by re-running only the failed piece.  
- **Warning**: It can add latency overhead to short queries. A dedicated fault-tolerant cluster for large jobs is recommended if you also have a high volume of short queries.

---

## 4. Encryption of Spooled Data

Trino **encrypts** all spooled intermediate data:

- A random **encryption key** is generated per query.  
- The key is discarded when the query completes.  
- Controlled by `fault-tolerant-execution.exchange-encryption-enabled=true`.

---

## 5. Advanced Configuration

### 5.1 Retry Limits

Configure how many times Trino retries queries or tasks before giving up:

| Property                  | Default | Policy  | Description                                                                         |
|---------------------------|---------|---------|-------------------------------------------------------------------------------------|
| `query-retry-attempts`   | `4`     | QUERY   | Max times Trino retries the entire query.                                           |
| `task-retry-attempts-per-task` | `4`     | TASK    | Max times a single task can be retried.                                             |
| `retry-initial-delay`     | `10s`   | Both    | Minimum wait time before first retry.                                               |
| `retry-max-delay`         | `1m`    | Both    | Maximum wait time for subsequent retries.                                           |
| `retry-delay-scale-factor`| `2.0`   | Both    | Exponential backoff scale factor for each retry.                                    |

### 5.2 Task Sizing (TASK Policy Only)

When `retry-policy=TASK`, tasks might be **too small** (leading to overhead) or **too big** (risk running out of memory). Trino supports limited auto-scaling, but you can manually tweak:

| Property                                          | Default  | Description                                                                            |
|---------------------------------------------------|----------|----------------------------------------------------------------------------------------|
| `fault-tolerant-execution-standard-split-size`    | `64MB`   | The “standard” size for each input split.                                              |
| `fault-tolerant-execution-max-task-split-count`   | `2048`   | Limits how many splits a single task can process.                                      |
| `fault-tolerant-execution-arbitrary-distribution-compute-task-target-size-min` | `512MB` | Minimum task size for non-writer stages in an arbitrary distribution plan.            |
| `fault-tolerant-execution-arbitrary-distribution-write-task-target-size-min`   | `4GB`   | Minimum task size for writer stages (e.g., insert).                                    |
| ... (various “growth factor” and “target size” props)     | *various*| Control how quickly tasks scale up in size after every `N` tasks are launched.         |

**How it Works**: If tasks keep finishing very fast, the engine might **adapt** to create larger tasks. If tasks fail for memory reasons, Trino can shrink them or reallocate them to bigger nodes.

### 5.3 Node Allocation (TASK Policy Only)

The property `fault-tolerant-execution-task-memory` sets the **initial** memory estimate per task. The system uses bin-packing logic to allocate tasks to nodes based on memory. If a task fails for OOM reasons, it can retry with a larger memory reservation or on a node with more available memory.

### 5.4 Additional Tuning

| Property                                                   | Default        | Policy | Description                                                                                                                     |
|------------------------------------------------------------|----------------|--------|---------------------------------------------------------------------------------------------------------------------------------|
| `fault-tolerant-execution-task-descriptor-storage-max-memory` | 15% of heap   | TASK   | The max memory on coordinator for storing metadata about tasks.                                                                 |
| `fault-tolerant-execution-max-partition-count`            | `50`           | TASK   | Upper bound on the number of partitions for distributed joins/aggregations in fault-tolerant mode.                              |
| `fault-tolerant-execution-min-partition-count`            | `4`            | TASK   | Lower bound on partitions for distributed joins/aggregations.                                                                   |
| `fault-tolerant-execution-min-partition-count-for-write`  | `50`           | TASK   | Minimum partitions in write queries (CTAS/INSERT) in fault-tolerant mode.                                                       |
| `max-tasks-waiting-for-node-per-query`                    | `50`           | TASK   | Limits how many tasks can be in the queue for node allocation at once.                                                          |

---

## 6. Exchange Manager

**Spooling** intermediate data to external storage is crucial for large queries in fault-tolerant mode:

1. **Create** a file like `etc/exchange-manager.properties` on all nodes:
   ```ini
   exchange-manager.name=filesystem
   exchange.base-directories=s3://my-spool-bucket
   # Additional S3 credentials or settings
   ```
2. The manager can store spool data on:
   - AWS S3 or S3-compatible systems (MinIO, GCS with S3 API, etc.)  
   - Azure Blob Storage  
   - Google Cloud Storage (via S3 compatibility or direct GCS approach)  
   - HDFS  
   - Local file system (not recommended for production).
3. The manager can compress and encrypt data. By default, LZ4 compression is enabled to reduce I/O.

**Example** (S3-based spool):

```ini
exchange-manager.name=filesystem
exchange.base-directories=s3://my-spooling-bucket
exchange.s3.region=us-west-1
exchange.s3.aws-access-key=EXAMPLEKEY
exchange.s3.aws-secret-key=EXAMPLESECRET
```

**Multiple Buckets**:
```ini
exchange.base-directories=s3://bucketA,s3://bucketB
```
Distributes spool data across multiple buckets to avoid I/O throttling on a single bucket.

---

## 7. Visualization of Fault-Tolerant Execution

```mermaid
flowchart LR
    A[Coordinator] -- Create Query --> B1[Stage 1]
    B1 -- Schedule Tasks --> W1[Worker 1]
    B1 -- Schedule Tasks --> W2[Worker 2]
    B1 --> B2[Stage 2]
    B2 -- Additional Tasks --> W3[Worker 3]
    B2 --> A

    W1 -- spool data --> S3/External Storage
    W2 -- spool data --> S3/External Storage
    W3 -- spool data --> S3/External Storage

    style B1 fill:#e7f1fe,stroke:#333,stroke-width:1px
    style B2 fill:#e7f1fe,stroke:#333,stroke-width:1px
    style W1 fill:#fce5cd,stroke:#333,stroke-width:1px
    style W2 fill:#fce5cd,stroke:#333,stroke-width:1px
    style W3 fill:#fce5cd,stroke:#333,stroke-width:1px
```

1. The **coordinator** breaks the query into stages.  
2. Each stage spools intermediate results to an external system (S3, HDFS, etc.).  
3. If *Worker 1* fails mid-query, its partial data is available in the spool. The coordinator can reassign those splits or tasks to *Worker 2* or *Worker 3*, reusing spooled data instead of reprocessing from scratch.

---

## 8. Best Practices & Recommendations

1. **Small Queries** → Use `retry-policy=QUERY` if your cluster primarily runs short, interactive queries.  
2. **Large Batch Queries** → Use `retry-policy=TASK` with a properly configured exchange manager for spool data.  
3. **Memory** → Ensure your coordinator has enough memory for storing spool metadata. Consider external spool storage to avoid running out of coordinator memory for big results.  
4. **Encryption** → Keep `fault-tolerant-execution.exchange-encryption-enabled=true` for secure spooling.  
5. **Multiple Clusters** → Possibly run separate clusters: one for short interactive queries with `QUERY` retries, another for large ETL/batch with `TASK` retries.

---

## 9. Conclusion

**Fault-tolerant execution** in Trino allows:

- **Automatic retries** of entire queries (`QUERY` policy) or individual tasks (`TASK` policy).  
- More resilient, reliable query execution.  
- Effective handling of node failures or transient issues.  
- Flexible *adaptive plan optimizations* for large-scale queries.

By carefully configuring an exchange manager, memory and retry limits, and choosing the right retry policy, you can significantly reduce failures and manual restarts for long-running queries in a distributed environment.