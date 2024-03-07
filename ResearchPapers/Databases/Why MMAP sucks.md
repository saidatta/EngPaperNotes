https://www.youtube.com/watch?v=1BRGU_AS25c
## Overview
The transcript discusses the pitfalls of using Memory-Mapped I/O (MMAP) in Database Management Systems (DBMS). The findings are based on experimental setups testing performance metrics, particularly focusing on read operations.
## Experimental Setup
- **Baseline**: Uses the `fio` benchmarking tool with the `odirect` flag to bypass the OS page cache. This setup ensures data is read directly into user space buffers.
- **Database Size**: 2 TB with only 100 GB allocated for the OS page cache, creating a larger than memory workload scenario.
- **Experiments**: Focused on read-only operations, which should ideally benefit from MMAP.
## Findings
### Random Reads
- **Experiment**: Measured reads per second in a workload with random access patterns, typical in OLTP applications.
- **Phases Observed**:
  1. MMAP comparable to `fio`.
  2. Performance drops to near zero after page cache fills up.
  3. Rebounds to about half of `fio`'s performance but with instability.
- **Causes**:
  1. Single process for page eviction in OS becomes CPU bound. (kswapd)
  2. Overhead from synchronizing the page table and data structures under high contention (100 concurrent threads).
  3. TLB (Translation Lookaside Buffer) shootdowns causing significant performance issues.
### Sequential Scan
- **Experiment**: Measured read bandwidth in sequential scan, typical in OLAP workloads.
- **Findings**:
  1. On a single SSD, MMAP performed worse than other methods.
  2. Performance drop-off when the page cache fills up.
  3. MMAP levels off at about 2 times worse performance than `fio`.
  4. With 10 SSDs in software RAID 0, the gap widens, and MMAP is about 20 times worse.
## Technical Insights
- **TLB Shootdowns**: This involves costly inter-processor interrupts to remove entries from the TLB when pages are evicted, leading to a performance hit.
- **Page Eviction**: Managed by a single process in the OS, which can become a bottleneck.
- **Page Table Synchronization**: High contention leads to overhead and performance degradation.
## Conclusion
- **MMAP and DBMS**: The study strongly suggests that MMAP is not suitable for DBMS due to inherent performance issues, especially under high-load conditions.
- **Buffer Pool Alternative**: Recommends using a traditional buffer pool for better performance and stability in DBMS.
- **Parting Thought**: Despite the attractiveness of MMAP for its simplicity and direct access to storage, its drawbacks in a DBMS context outweigh its benefits.
## Implications
This research has significant implications for database architecture and system design. It challenges the notion that MMAP, despite its ease of use and initial performance benefits, is a viable long-term solution for high-performance DBMS. The study underscores the importance of traditional database management techniques, particularly in handling large-scale, high-concurrency environments.