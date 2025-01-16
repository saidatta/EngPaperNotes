Below is a **PhD-level** exploration of the **`SnapshotTracker`** module in InfluxDB 3.0’s Write-Ahead Log (WAL) subsystem. We will explain **why** and **how** the **snapshot logic** is tracked, **when** snapshots are triggered, and **how** this code helps manage WAL files. Concrete examples, **code snippets**, and **visual diagrams** demonstrate how the `SnapshotTracker` ensures old WAL data is snapshotted and later removed from the object store.

---

# Table of Contents
1. [High-Level Overview](#high-level-overview)  
2. [Key Data Structures & Concepts](#key-data-structures--concepts)  
   1. [SnapshotTracker](#snapshottracker)  
   2. [WalPeriod](#walperiod)  
   3. [SnapshotInfo & SnapshotDetails](#snapshotinfo--snapshotdetails)  
3. [Detailed Logic & Flows](#detailed-logic--flows)  
   1. [add_wal_period](#1-add_wal_period)  
   2. [snapshot](#2-snapshot)  
4. [Configuration & Heuristics](#configuration--heuristics)  
5. [Example Walkthrough](#example-walkthrough)  
6. [Diagram](#diagram)  
7. [Edge Cases & Testing](#edge-cases--testing)  
8. [Summary & Further Notes](#summary--further-notes)

---

## 1. High-Level Overview

In **InfluxDB 3.0**, the **Write-Ahead Log** may generate multiple WAL files over time, each storing a *range of data* (timestamps, etc.). To **compact** or **persist** these older WAL files as Parquet (or other forms), the system triggers **snapshots**. The **`SnapshotTracker`**:

- Maintains a **list** of WAL “periods,” each representing a single WAL file.  
- Decides **when** to generate a **snapshot**.  
- Determines **which** WAL files can be considered “ready for snapshot,” based on time constraints and maximum size thresholds.  

Once a snapshot is triggered, older WAL files can be safely deleted, reclaiming object store space and preventing unbounded growth.

---

## 2. Key Data Structures & Concepts

### 2.1 `SnapshotTracker`

```rust
pub(crate) struct SnapshotTracker {
    last_snapshot_sequence_number: SnapshotSequenceNumber,
    last_wal_sequence_number: WalFileSequenceNumber,
    wal_periods: Vec<WalPeriod>,
    snapshot_size: usize,
    gen1_duration: Gen1Duration,
}
```

**Key fields**:
- **`last_snapshot_sequence_number`**: The ID of the most recent snapshot that was taken.  
- **`last_wal_sequence_number`**: Tracks the highest WAL file sequence number seen so far.  
- **`wal_periods`**: A list of periods (one per WAL file).  
- **`snapshot_size`**: The threshold controlling how many WAL periods we typically want to accumulate before snapshotting.  
- **`gen1_duration`**: The chunk duration for grouping data into time blocks (used to compute final snapshot boundaries).

### 2.2 `WalPeriod`

```rust
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct WalPeriod {
    pub(crate) wal_file_number: WalFileSequenceNumber,
    pub(crate) min_time: Timestamp,
    pub(crate) max_time: Timestamp,
}
```

- Each WAL file has a **sequence number** and tracks the **minimum** and **maximum** data timestamps in it.  
- **No Overlaps**: The code assumes each new `WalPeriod` starts after the last WAL file’s sequence number.  

### 2.3 `SnapshotInfo` & `SnapshotDetails`

```rust
pub struct SnapshotInfo {
    pub(crate) snapshot_details: SnapshotDetails,
    pub(crate) wal_periods: Vec<WalPeriod>,
}
```

- **`SnapshotDetails`**: Has the **snapshot_sequence_number** plus an **end_time_marker** (all data older than this is included in the snapshot) and **last_wal_sequence_number** (which WAL file is fully snapshotted).  
- **`wal_periods`**: The set of WAL files that are included in this snapshot, which can be safely deleted later.

---

## 3. Detailed Logic & Flows

### 3.1 `add_wal_period`

Each time a new WAL file is created (and finalized), the system calls:

```rust
pub(crate) fn add_wal_period(&mut self, wal_period: WalPeriod) {
    if let Some(last_period) = self.wal_periods.last() {
        assert!(last_period.wal_file_number < wal_period.wal_file_number);
    }
    self.last_wal_sequence_number = wal_period.wal_file_number;
    self.wal_periods.push(wal_period);
}
```

- Ensures **strictly increasing** WAL file sequence numbers.  
- Appends the newly created WAL file’s `WalPeriod` to the internal list.  
- Updates the `last_wal_sequence_number`.

### 3.2 `snapshot`

**Core logic** for deciding if/when to snapshot:

```rust
pub(crate) fn snapshot(&mut self) -> Option<SnapshotInfo> {
    // 1) If no WAL periods or not enough to trigger a snapshot, return None
    if self.wal_periods.is_empty()
        || self.wal_periods.len() < self.number_of_periods_to_snapshot_after()
    {
        return None;
    }

    // 2) If too many WAL files are piling up (3x snapshot_size),
    // we forcibly snapshot all but the last WAL period
    if self.wal_periods.len() >= 3 * self.snapshot_size {
        // snapshot all but the last file
        ...
        return Some(SnapshotInfo { ... });
    }

    // 3) Otherwise, attempt a normal snapshot up to some boundary "t"
    let t = self.wal_periods.last().unwrap().max_time;
    let t = t - (t.get() % self.gen1_duration.as_nanos());
    let periods_to_snapshot = self
        .wal_periods
        .iter()
        .take_while(|p| p.max_time < t)
        .cloned()
        .collect::<Vec<_>>();

    // 4) If there are old periods that end before "t", we can snapshot them:
    periods_to_snapshot.last().cloned().map(|period| {
        // remove these from the in-memory list
        self.wal_periods.retain(|p| p.wal_file_number > period.wal_file_number);

        SnapshotInfo {
            snapshot_details: SnapshotDetails {
                snapshot_sequence_number: self.increment_snapshot_sequence_number(),
                end_time_marker: t.get(),
                last_wal_sequence_number: period.wal_file_number,
            },
            wal_periods: periods_to_snapshot,
        }
    })
}
```

#### Explanation

1. **Early exit** if not enough WAL periods exist to justify a snapshot.  
2. If the number of WAL periods is **3x** the `snapshot_size`, we assume data is accumulating without older data being snapshotted. We forcibly snapshot all but the newest WAL period.  
3. Otherwise, we compute a time boundary `t` (aligned to `gen1_duration`).  
4. We gather all WAL periods whose `max_time < t`; those can be snapshotted. Everything after that remains in the WAL.  
5. Return a `SnapshotInfo` describing which WAL files to remove once the snapshot completes.

---

## 4. Configuration & Heuristics

1. **`snapshot_size`**: How many WAL periods typically accumulate before normal snapshot attempts.  
2. **`gen1_duration`**: Data is chunked into fixed time intervals (e.g., 5m blocks). If `t` is 120.9 seconds, it might be truncated to 120 seconds for alignment.  
3. **3x Forced Snapshot**: If the WAL can’t proceed with normal snapshots (e.g., data is out-of-order or we keep seeing future timestamps), we eventually force a snapshot to avoid indefinite WAL growth.

---

## 5. Example Walkthrough

We have `snapshot_size = 2`, `gen1_duration = 1m`, and we create these WAL periods:

- Period #1: `wal_file_number=1`, min_time=0, max_time=60_000000000 (60s).  
- Period #2: `wal_file_number=2`, max_time=119.9s.  
- Period #3: `wal_file_number=3`, max_time=120.9s.  
- Period #4: `wal_file_number=4`, max_time=240.0s.  

We repeatedly call `snapshot()`.  
1. After adding #1, #2, and #3, the code checks if we have at least `2 + (2/2)=3` = 3 WAL periods. Yes, we do.  
2. We compute `t = 120.9 - (120.9 % 60) = 120`.  
3. We gather all WAL periods with `max_time < 120` → #1 (max_time=60.0) and #2 (max_time=119.9).  
4. We remove them from the list and produce a `SnapshotInfo` with `end_time_marker=120`, and `last_wal_sequence_number=2`. That means WAL files 1 and 2 are safely snapshot-ready and can be deleted once the snapshot completes.  
5. The next time we check, we see that #3 remains. We keep going until #4, etc.

---

## 6. Diagram

```
 ┌───────────────────────────────────────────────────┐
 │  SnapshotTracker                                 │
 │   - wal_periods: Vec<WalPeriod>                 │
 │   - snapshot_size: 2                            │
 │   - gen1_duration: 60 seconds                   │
 │   - last_snapshot_seq_num, last_wal_seq_num     │
 └────────────────────────┬─────────────────────────┘
                          │  (1) add_wal_period(...)
                          v
   +------------------------------------------------------+
   | wal_periods = [ Period( #1, 0..60 ),                 |
   |                   Period( #2, 0..119.9 ),            |
   |                   Period( #3, 60..120.9 ), ... ]     |
   +------------------------------------------------------+
                          │
                          │ (2) snapshot() => checks threshold
                          v
 ┌────────────────────────────────────────────────────┐
 │ 1) # of WAL periods >= snapshot_size+(size/2)?    │
 │ 2) forced if >= 3*snapshot_size?                  │
 │ 3) boundary t = floor( max_time / gen1_dur ) * dur │
 │ 4) gather all periods with max_time < t           │
 └────────────────────────────────────────────────────┘
                          │
                          v
                returns SnapshotInfo (some or none)
                includes list of old WAL periods
```

---

## 7. Edge Cases & Testing

### 7.1 Large Future Timestamps

If new data with a **very large** timestamp arrives, `max_time` jumps significantly, potentially preventing earlier WAL periods from being snapshotted. The code eventually hits the **3x** rule, forcing a snapshot.

### 7.2 Overlapping or Backdated Timestamps

The code expects strictly increasing WAL file sequence numbers, but times can jump around. The logic focuses on `max_time` for each WAL file. Overlapping times can accumulate, but eventually either the forced snapshot or normal snapshot logic will handle it.

### 7.3 `number_of_periods_to_snapshot_after`

```rust
fn number_of_periods_to_snapshot_after(&self) -> usize {
    self.snapshot_size + self.snapshot_size / 2
}
```

This ensures a little buffer (1.5x snapshot_size) so we don’t snapshot too frequently.

---

## 8. Summary & Further Notes

**`SnapshotTracker`** elegantly handles older WAL files and ensures they are:

- **Properly chunked** by time (`gen1_duration`).  
- **Collected** into snapshots once enough data accumulates or we detect an outlier scenario.  
- **Safely** removed once the snapshot completes, preventing indefinite WAL growth.

This approach allows InfluxDB 3.0 to **balance** real-time ingestion with the efficient, time-based chunking of data into Parquet files. By maintaining minimal state (`wal_periods`, `snapshot_size`, etc.), the `SnapshotTracker` ensures snapshot decisions are fast, consistent, and predictable for both typical ingestion patterns and edge-case timestamp scenarios.