#### Overview
The `Compact` method in Prometheus TSDB is responsible for merging smaller blocks of time series data into larger blocks. This process is known as compaction and helps in optimizing storage and improving query performance.
#### Sequence Diagram
Here is a detailed ASCII sequence diagram for the `Compact` method that showcases the compaction data process in the `DB` struct:

```
+------------------+       +------------------+       +------------------+
|      Caller      |       |        DB        |       |      System      |
+------------------+       +------------------+       +------------------+
         |                          |                          |
         |        Compact(ctx)      |                          |
         |-------------------------->                          |
         |                          |                          |
         |                          |  Lock cmtx               |
         |                          |------------------------->|
         |                          |                          |
         |                          |  defer Unlock cmtx       |
         |                          |------------------------->|
         |                          |                          |
         |                          |  defer truncateWAL       |
         |                          |------------------------->|
         |                          |                          |
         |                          |  Check head.compactable  |
         |                          |------------------------->|
         |                          |                          |
         |                          |  if not compactable      |
         |                          |------------------------->|
         |                          |  return nil              |
         |                          |<-------------------------|
         |                          |                          |
         |                          |  if waitingForCompactionDelay |
         |                          |------------------------->|
         |                          |  return nil              |
         |                          |<-------------------------|
         |                          |                          |
         |                          |  Create RangeHead        |
         |                          |------------------------->|
         |                          |                          |
         |                          |  WaitForAppendersOverlapping |
         |                          |------------------------->|
         |                          |                          |
         |                          |  compactHead(rh)         |
         |                          |------------------------->|
         |                          |                          |
         |                          |  if err                  |
         |                          |------------------------->|
         |                          |  return err              |
         |                          |<-------------------------|
         |                          |                          |
         |                          |  truncateWAL(lastBlockMaxt) |
         |                          |------------------------->|
         |                          |                          |
         |                          |  if err                  |
         |                          |------------------------->|
         |                          |  return err              |
         |                          |<-------------------------|
         |                          |                          |
         |                          |  compactOOOHead(ctx)     |
         |                          |------------------------->|
         |                          |                          |
         |                          |  if err                  |
         |                          |------------------------->|
         |                          |  return err              |
         |                          |<-------------------------|
         |                          |                          |
         |                          |  compactBlocks()         |
         |                          |------------------------->|
         |                          |                          |
         |                          |  if err                  |
         |                          |------------------------->|
         |                          |  return err              |
         |                          |<-------------------------|
         |                          |                          |
         |                          |  return nil              |
         |<--------------------------                          |
         |                          |                          |
```

#### Code Explanation
The `Compact` method performs the following steps:
1. **Locking and Defer Statements**:
   - Locks the `cmtx` mutex to ensure no other compaction or deletion runs simultaneously.
   - Defers the unlocking of `cmtx` and the truncation of the WAL (Write-Ahead Log).
2. **Check Compactability**:
   - Checks if the head is compactable. If not, it returns `nil`.
   - Checks if the system is waiting for a compaction delay. If so, it returns `nil`.
3. **Create RangeHead**:
   - Creates a `RangeHead` which is a subset of the head block that needs to be compacted.
4. **Wait for Appenders**:
   - Waits for any overlapping appenders to finish.
5. **Compact Head**:
   - Calls `compactHead(rh)` to compact the head block.
6. **Truncate WAL**:
   - Truncates the WAL up to the maximum time of the last block.
7. **Compact Out-of-Order Head**:
   - Calls `compactOOOHead(ctx)` to compact the out-of-order head block.
8. **Compact Blocks**:
   - Calls `compactBlocks()` to compact the blocks.
9. **Return**:
   - Returns `nil` if all operations are successful, otherwise returns the encountered error.
#### Code Example
Here is a simplified version of the `Compact` method:

```go
func (db *DB) Compact(ctx context.Context) error {
    db.cmtx.Lock()
    defer db.cmtx.Unlock()
    defer db.truncateWAL()

    if !db.head.compactable() {
        return nil
    }

    if db.waitingForCompactionDelay() {
        return nil
    }

    rh := db.createRangeHead()
    db.waitForAppendersOverlapping()

    if err := db.compactHead(rh); err != nil {
        return err
    }

    if err := db.truncateWAL(db.head.lastBlockMaxt()); err != nil {
        return err
    }

    if err := db.compactOOOHead(ctx); err != nil {
        return err
    }

    if err := db.compactBlocks(); err != nil {
        return err
    }

    return nil
}
```

#### Equations and Metrics
- **Compaction Delay**: `CompactionDelay = db.generateCompactionDelay()`
- **Metrics**:
  - `db.metrics.compactionsTriggered.Inc()`
  - `db.metrics.compactionsFailed.Inc()`

#### Block Compactions
Block compactions involve merging smaller blocks into larger ones to optimize storage and improve query performance. The process includes reading data from blocks, merging overlapping time series, and writing the merged data into new blocks.

#### Detailed Steps
1. **Reading Data**:
   - Reads data from the identified blocks, including index, chunk, and tombstone files.

2. **Merging Data**:
   - Merges the data from the blocks, combining overlapping time series and removing deleted data.

3. **Writing Data**:
   - Writes the merged data into a new block, including new index and chunk files.

4. **Cleaning Up**:
   - Marks the old blocks as deletable and removes them if necessary.

#### ASCII Diagram for Block Compaction
```
+------------------+       +------------------+       +------------------+
|      Caller      |       | LeveledCompactor |       |      System      |
+------------------+       +------------------+       +------------------+
         |                          |                          |
         |       Compact(dest, dirs, open)                     |
         |---------------------------------------------------->|
         |                          |                          |
         |                          |  Plan(dir)               |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  plan(dms)               |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  selectOverlappingDirs   |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  selectDirs              |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  readMetaFile(dir)       |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  OpenBlock(logger, dir)  |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  CompactBlockMetas       |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  write(dest, meta, ...)  |
         |                          |------------------------->|
         |                          |                          |
         |                          |  chunks.NewWriterWithSegSize |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  index.NewWriterWithEncoder |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  blockPopulator.PopulateBlock |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  writeMetaFile(logger, tmp, meta) |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  tombstones.WriteFile    |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  fileutil.OpenDir(tmp)   |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |                          |  fileutil.Replace(tmp, dir) |
         |                          |------------------------->|
         |                          |                          |
         |                          |<-------------------------|
         |<----------------------------------------------------|
         |                          |                          |
```

This diagram represents the sequence of operations performed by the `LeveledCompactor` during the compaction process, including interactions with the file system and the various steps involved in reading, merging, and writing blocks.