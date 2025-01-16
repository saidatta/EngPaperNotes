Below are detailed Obsidian notes derived from the article on MySQL's Write-Ahead Logging (WAL) mechanism, formatted to suit the information needs of a Software Engineer. This includes technical details, code snippets, and performance considerations crucial for understanding and implementing WAL in MySQL environments.
https://www.jianshu.com/p/f242bc1e95ff
---
- **WAL Introduction**: Ensures data integrity by writing logs before actual data writes to disk.
- **Key Concepts**:
  - **Dirty Page**: A memory data page not synced with its disk counterpart.
  - **Clean Page**: Synced memory and disk data pages.
  - **Flush Process**: The action of writing dirty pages to disk, making them clean.
### Timing of InnoDB Flushing Dirty Pages
1. **Redo Log Full**: Triggers a checkpoint advance, necessitating a flush of corresponding dirty pages to disk.
2. **Memory Insufficiency**: Necessitates flushing of dirty pages before eviction.
### Performance Considerations
- **Dirty Page Management**: InnoDB utilizes `innodb_io_capacity` to gauge disk IOPS and optimizes flushing strategy based on this along with `innodb_max_dirty_pages_pct`.
- **Binlog Writing Mechanism**: Uses a binlog cache per thread, controlled by `binlog_cache_size`, flushing to binlog files upon transaction commit.
- **Redo Log Writing Mechanism**: Managed via the `innodb_flush_log_at_trx_commit` parameter, influencing log buffer writes and fsync frequency.
- ![[Screenshot 2024-02-24 at 12.59.19 PM.png]]
### Code Snippets and Commands
- **Checking Disk IOPS**: Use the `fio` tool.
  ```bash
  fio -filename=$filename -direct=1 -iodepth 1 -thread -rw=randrw -ioengine=psync -bs=16k -size=500M -numjobs=10 -runtime=10 -group_reporting -name=mytest
  ```
### InnoDB Control Strategies
- **IO Capacity Management**: Use `innodb_io_capacity` to set disk IOPS.
- **Dirty Pages Proportion**: Governed by `innodb_max_dirty_pages_pct`.
- **Flushing Speed Adjustment**: Based on dirty pages ratio and redo log write speed.
### Optimization Techniques
- **Neighbor Page Flushing**: Controlled by `innodb_flush_neighbors`, optimizing disk IO based on storage type.
- **Binlog and Redo Log Settings**:
  - **Binlog Sync Control**: Adjust `sync_binlog` for balancing performance and data safety.
  - **Redo Log Persistence**: `innodb_flush_log_at_trx_commit` settings impact performance and data resilience.
### Group Submission Mechanism
- The log logical sequence number (LSN) is a monotonically increasing value that corresponds to each writing point of the redo log. Each time a redo log of length length is written, length will be added to the LSN value.
- Utilizes log sequence numbers (LSN) for efficient disk IOPS utilization, controlled by `binlog_group_commit_sync_delay` and `binlog_group_commit_sync_no_delay_count`.
### Performance Improvement Methods
1. **Adjust Binlog Group Commit Settings**: Modifying `binlog_group_commit_sync_delay` and `binlog_group_commit_sync_no_delay_count`.
2. **Tune Sync Binlog**: Setting `sync_binlog` to a higher value (100-1000) to reduce disk writes at the risk of potential data loss.
3. **Optimize Redo Log Flushing**: Setting `innodb_flush_log_at_trx_commit` to 2 for performance, with consideration for power loss scenarios.

---