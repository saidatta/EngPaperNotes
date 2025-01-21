Below is a **PhD-level** explanation of the **`WalObjectStore`** code from InfluxDB 3.0’s WAL. We will explore **architecture**, **object store usage**, **flush concurrency**, **snapshot logic**, **replay** steps, and how all these components work together to provide durable, consistent writes in the face of restarts. Code snippets, **detailed diagrams**, and advanced design insights are provided.

---

# Table of Contents
1. [High-Level Overview](#high-level-overview)  
2. [Core Components](#core-components)  
    1. [`WalObjectStore`](#walobjectstore)  
    2. [`FlushBuffer` & `WalBuffer`](#flushbuffer--walbuffer)  
    3. [Snapshots & Concurrency](#snapshots--concurrency)  
3. [Lifecycle & Flows](#lifecycle--flows)  
    1. [Initialization & Replay](#initialization--replay)  
    2. [Buffering Writes & Flushing](#buffering-writes--flushing)  
    3. [Snapshot Handling & Cleanup](#snapshot-handling--cleanup)  
    4. [Shutdown Sequence](#shutdown-sequence)  
4. [Object Store Serialization & Retrying](#object-store-serialization--retrying)  
5. [Example Code Flow](#example-code-flow)  
6. [Diagram](#diagram)  
7. [Summary & Notes](#summary--notes)

---

## 1. High-Level Overview

The **`WalObjectStore`** type implements the **`Wal`** trait and writes WAL files to an **object store** (e.g., S3, GCS, or in-memory). It:

- **Buffers** `WalOp` writes in memory (e.g., timeseries data or catalog ops).  
- **Serializes** them periodically (or on demand) to a WAL file in the object store.  
- **Notifies** a consumer (`file_notifier`) so that newly persisted data can be loaded into in-memory query buffers.  
- **Triggers snapshots** at configured intervals, removing older WAL files after a successful snapshot.  
- **Replays** existing WAL files at startup to restore state.  

This design ensures **durability** in distributed storage, while also supporting asynchronous flush and snapshot operations.  

---

## 2. Core Components

### 2.1 `WalObjectStore`

```rust
#[derive(Debug)]
pub struct WalObjectStore {
    object_store: Arc<dyn ObjectStore>,
    host_identifier_prefix: String,
    file_notifier: Arc<dyn WalFileNotifier>,
    flush_buffer: Mutex<FlushBuffer>,
}
```

1. **`object_store`**: A generic handle for uploading/downloading WAL files.  
2. **`host_identifier_prefix`**: A string prefix for generating the path like `my_host/wal/00000000001.wal`.  
3. **`file_notifier`**: Callback that processes newly persisted WAL files (loading them into caches, etc.).  
4. **`flush_buffer`**: A `Mutex<FlushBuffer>`, holding in-memory state (unflushed writes, concurrency trackers, snapshot logic).

### 2.2 `FlushBuffer` & `WalBuffer`

**`FlushBuffer`** encapsulates:

- **`wal_buffer: WalBuffer`** – Stores unpersisted writes (`database_to_write_batch`, `catalog_batches`).  
- **`snapshot_tracker`** – Tracks how many WAL files have been persisted, when to trigger snapshots, etc.  
- **`snapshot_semaphore`** – A concurrency limit ensuring only one snapshot at a time.

**`WalBuffer`**:

```rust
struct WalBuffer {
    op_count: usize,
    op_limit: usize,
    database_to_write_batch: HashMap<Arc<str>, WriteBatch>,
    catalog_batches: Vec<CatalogBatch>,
    write_op_responses: Vec<oneshot::Sender<WriteResult>>,
    ...
}
```
- Buffers `WalOp::Write(...)` or `WalOp::Catalog(...)`.  
- Caps the total number of ops with `op_limit`; otherwise returns `BufferFull`.  
- Each write may optionally be **blocking**: a `write_ops(...)` call uses a oneshot channel to notify the client that the data is fully persisted.

### 2.3 Snapshots & Concurrency

**`SnapshotTracker`**:

- Tracks how many WAL files have been created.  
- Determines if a new flush should also trigger a snapshot.  
- Associates each WAL file with a time range (min/max timestamps) for potential pruning.

The WAL uses an internal **semaphore**:

```rust
snapshot_semaphore: Arc<Semaphore>
```
…where only one snapshot can proceed at a time, preventing race conditions when removing old WAL files.

---

## 3. Lifecycle & Flows

### 3.1 Initialization & Replay

```rust
pub async fn new(
    ...
    config: WalConfig,
    ...
) -> Result<Arc<Self>, crate::Error> {
    let wal = Self::new_without_replay(...);
    wal.replay().await?;
    // spawn background flush
    background_wal_flush(Arc::clone(&wal), flush_interval);
    Ok(Arc::new(wal))
}
```
On startup:
1. **`new_without_replay`** sets up initial WAL state, including the next WAL file number.  
2. **`replay()`** scans object store for existing WAL files, replays each in ascending sequence, calling `file_notifier.notify(...)` or `notify_and_snapshot(...).await` for any embedded snapshot.  
3. The `background_wal_flush` is spawned, ensuring flushes happen periodically (based on `WalConfig::flush_interval`).

### 3.2 Buffering Writes & Flushing

Two main methods:

1. **`buffer_op_unconfirmed(...)`**  
   - Appends to memory, returning immediately. Data isn’t persisted yet.  
   - If the in-memory buffer is full (`op_count >= op_limit`), returns `Error::BufferFull(...)`.

2. **`write_ops(...)`**  
   - Appends ops, but also sets up a **oneshot channel** that is signaled once the data is actually persisted.  
   - Allows a user to block until disk durability is guaranteed.

At some point, a flush occurs (either triggered by `background_wal_flush` or a direct call to `flush_buffer`). The code:

- **Grabs** the lock on `flush_buffer`.  
- **Swaps** out the current `WalBuffer` with a fresh one.  
- **Serializes** the old buffer into `WalContents` struct, storing ops + min/max timestamps.  
- **Writes** it to the object store, retrying if needed.  
- If a snapshot is due, includes a `Some(snapshot_details)` in `WalContents`.

### 3.3 Snapshot Handling & Cleanup

If the newly flushed `WalContents` has a `snapshot`:

```rust
let snapshot_done = file_notifier.notify_and_snapshot(wal_contents, snapshot_details).await;
...
(snapshot_done, snapshot_info, snapshot_permit)
```

- The `WalObjectStore` spawns a background task that eventually calls `snapshot_done.await`, guaranteeing the snapshot is finished.  
- Then `cleanup_snapshot(...)` is invoked to remove older WAL files. It calls `object_store.delete(...)` in a loop until success or certain errors are encountered, releasing the **`snapshot_permit`** in the end.

### 3.4 Shutdown Sequence

```rust
async fn shutdown(&self) {
    // 1) block further writes
    self.flush_buffer.lock().await.wal_buffer.is_shutdown = true;

    // 2) flush current buffer (if any)
    if let Some((snapshot_done, snapshot_info, snapshot_permit)) = self.flush_buffer().await {
        let snapshot_details = snapshot_done.await.expect("snapshot should complete");
        self.remove_snapshot_wal_files(snapshot_info, snapshot_permit).await;
    }
}
```
This ensures no more new writes are accepted, any unflushed data is persisted, and pending snapshots complete. This “clean” shutdown means minimal data loss risk.

---
## 4. Object Store Serialization & Retrying
During flush:
```rust
loop {
    match self
        .object_store
        .put(&wal_path, PutPayload::from_bytes(data.clone()))
        .await
    {
        Ok(_) => { break; }
        Err(e) => {
            error!(%e, "error writing wal file to object store");
            retry_count += 1;
            if retry_count > 100 {
                // eventually give up
                ...
                return None;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
```
If object store writes fail (e.g., network issues), the WAL tries again up to 100 times. Beyond that, it signals failures to all waiting `oneshot` channels, returning an error message. This handles **transient** errors gracefully.

---
## 5. Example Code Flow

```mermaid
sequenceDiagram
    participant Client
    participant WalObjectStore
    participant flush_buffer(Mutex<FlushBuffer>)
    participant WalBuffer

    Client->>WalObjectStore: write_ops([WalOp::Write(...)]).await
    WalObjectStore->>flush_buffer(Mutex): lock()
    flush_buffer(Mutex)->>WalBuffer: buffer_ops_with_response(ops, channel)
    WalBuffer->>WalBuffer: store op in memory
    flush_buffer(Mutex)->>WalObjectStore: unlock()

    note over WalObjectStore: background flush ticks

    WalObjectStore->>flush_buffer(Mutex): lock()
    flush_buffer(Mutex)->>WalBuffer: swap buffer, produce WalContents
    flush_buffer(Mutex)->>WalObjectStore: unlock()

    WalObjectStore->>ObjectStore: put() WAL file
    alt success
        WalObjectStore->>file_notifier: notify(...) or notify_and_snapshot(...)
        file_notifier->>WalObjectStore: snapshot_done (if snapshot)
        WalObjectStore->>oneshot channels: send Success(())
    else error
        WalObjectStore->>oneshot channels: send Error("object store fail")
    end
```

1. **Client** calls `write_ops(...)`.  
2. The WAL enqueues ops + a response channel in `WalBuffer`.  
3. Periodically, a flush empties `WalBuffer`, creating a `WalContents`.  
4. If object store write succeeds, the WAL notifies the `file_notifier`.  
5. If snapshot is requested, the WAL spawns snapshot tasks, eventually cleaning up older WAL files.

---

## 6. Diagram

Here’s a more detailed, static ASCII diagram capturing concurrency and states:

```
         ┌────────────────────┐
         │ WalObjectStore     │
         │   (implements Wal) │
         └─────────┬──────────┘
                   │
                   │ (1) new(...) -> replay()
                   v
      +----------------------------------+
      |  flush_buffer: Mutex<FlushBuffer>|
      |   - snapshot_tracker             |
      |   - wal_buffer (WalBuffer)       |
      |   - snapshot_semaphore           |
      +----------------------------------+
                   │
                   │ (2) buffer_op_unconfirmed / write_ops
                   v
      +----------------------------------+
      |  WalBuffer                       |
      |   - op_limit, op_count          |
      |   - map<db_name,WriteBatch>      |
      |   - catalog_batches             |
      |   - write_op_responses          |
      +----------------------------------+
                   │  store ops in memory
                   │
                   └─(3) background flush interval
                   │
                   v
            [flush_buffer().await]
                   │
                   │ (A) "swap out" old WalBuffer
                   │ (B) create WalContents
                   │ (C) object_store.put(...) => retries
                   │ (D) notify / notify_and_snapshot => snapshot?
                   └────────────────────┐
                                       ▼
                                snapshot concurrent
                                        |
                                        v
                          remove_snapshot_wal_files(...)

```

---

## 7. Summary & Notes

### Key Takeaways:

- **`WalObjectStore`** leverages an **in-memory** WAL buffer plus an **object store** for durable writes.  
- **Replay** recovers from existing WAL files on startup.  
- **Snapshot** logic periodically merges older WAL files, removing them upon completion.  
- **Synchronization** is done via `Mutex<FlushBuffer>` plus a `Semaphore` for one active snapshot.  
- **Retries** on object store errors ensure eventual success or error reporting to waiting callers.

This design provides a **robust** solution for time-series ingest with durability guarantees. By decoupling memory buffering from the periodic object store writes and snapshot tasks, the system can handle **high-throughput** ingestion while ensuring data remains safe and recoverable upon restarts.