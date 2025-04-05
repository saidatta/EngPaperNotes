> **Note:** This write-up includes detailed discussion from the transcript around join ordering, the reasons DataFusion has a simple built-in optimizer, concurrency and scheduling with Tokio, pull-based vs. push-based execution, performance comparisons (DuckDB, TPC-H, etc.), and how advanced optimizations (e.g. cost-based join reordering) might be added to DataFusion via extension APIs.

---
## 1. Semantic vs. Cost-Based Optimization

### 1.1 Join Ordering in DataFusion
- **Current State**  
  - DataFusion’s join ordering is effectively *semantic* or *syntactic*: the plan executes joins in the order they appear in the query.  
  - There is no *full-blown cost-based join reordering*.  
  - **Motivation:** True cost-based join ordering requires accurate cardinality estimation, which is famously hard to get right (remains an open research area).
- **Minimal Heuristics**  
  - DataFusion includes “just enough” joint reordering logic to avoid catastrophic performance on TPC-H or other typical queries.  
  - Example: certain star-schema style queries or subqueries that would lead to “join ordering disasters” are partially mitigated with heuristic rewrites.  
- **Reasoning**  
  - Many OLAP workloads today use denormalized data or “flattened” tables, thus fewer multi-join queries.  
  - If advanced reordering is needed, **DataFusion has an API** so custom or third-party optimizers can be plugged in.  
  - *Quote (paraphrased)*: “We’re not investing in a massive cost-based join optimizer now, but you can do it yourself if your application requires it.”
### 1.2 Optimizer Complexity & Catalog Statistics
- **Lack of Full Statistics**  
  - DataFusion’s default catalog is minimal and does not maintain advanced histograms or correlation metrics.  
  - The built-in optimizer is mostly rule-based (filter pushdown, constant folding, limit pushdown, etc.).  
- **Extension Points**  
  - You can supply custom statistics or implement a cost-based planner with your own rules.  
  - _Some mention of “OPD”_ (a new Cascades-style approach in development) might introduce cost-based rewrites in the future, but it’s still highly experimental.
---
## 2. Physical Planning Review
- **Recap**  
  - Logical plan → Physical plan. Physical plan enumerates how to implement each operator (hash join, merge join, partial aggregate, sorting, etc.).  
  - “Execution plan” in DataFusion is the tree of physical operators.  
- **Execution Model**  
  - Volcano-style, *pull-based* streaming. Each operator requests the next `RecordBatch` from its children.  
  - Operators produce streams of `RecordBatch`es asynchronously, using Rust’s async/await via the [Tokio](https://tokio.rs/) runtime.  

---
## 3. Executor Model: Pull-Based, Streaming, and Tokio

### 3.1 Pull-Based Model vs. Push-Based (Morsel-Driven)

DataFusion uses a **Volcano-style pull** approach:
1. An operator calls `child.execute(partition).next_batch()` (in an async fashion).  
2. It processes that batch of ~8,000 rows (default batch size).  
3. Produces a new batch or calls next again.  

**Why not push-based morsel-driven scheduling?**
- **Historical Paper:** “Morsel-Driven Parallelism” paper influenced systems like DuckDB and Umbra. They claim advantages in CPU utilization and cache locality.  
- **DataFusion’s Experience**  
  - Implementing a push-based engine from scratch (with advanced morsel scheduling) is complex.  
  - DataFusion tried a partial push-based variant with [Rayon](https://github.com/rayon-rs/rayon). They found no big performance gains vs. Tokio’s thread pool.  
  - The nature of Rust’s async/await _already_ yields many of the co-routine style benefits (e.g. not blocking threads on I/O, easy scheduling of tasks).  

### 3.2 Tokio for Parallel Execution
- **Tokio** is a popular asynchronous runtime in Rust. It provides:
  1. **Work-Stealing Thread Pool:** Efficient scheduling across CPU cores.  
  2. **Async I/O** integration: if a DataFusion operator must do I/O, it suspends, and Tokio runs other tasks.  
  3. **Async/await** continuation mechanism: the compiler transforms your synchronous-looking code into state machines.
- **Single Data Source**: If your operator must do an I/O call or wait for data, it uses `await` and yields automatically.  
- **Advantages**:  
  - Eliminates the need to write a custom thread pool scheduler for DataFusion.  
  - Good performance for CPU-bound data processing with occasional I/O.  
  - Automatic support for “cancellation” or “time-out” if a query is aborted—Rust’s drop rules clean up resources.  
### 3.3 Partitions & Exchange Operators
- **Partitions**:  
  - If the plan says “4 partitions” for a `FilterExec`, then at runtime, you effectively have 4 independent filter tasks.  
  - Each runs in parallel on a subset of data (e.g., separate Parquet files or row groups).
- **Exchange Operator**:  
  - Repartition data across tasks or shuffle data for distributed or multi-core joins.  
  - Potentially the most complex operator in DataFusion; tries to limit thread contention. 
---
## 4. Memory Management
- **Current Implementation**:  
  - Each operator declares large allocations (e.g., building a hash table, storing a sort run).  
  - Cooperative memory pool (e.g., “global memory limit”): either you get the memory or must spill to disk.  
  - Fine-grained tracking of every small intermediate buffer is not done (some overhead vs. minimal gain).  
- **Spilling**:  
  - DataFusion supports external sorting and external hashing, but the performance is not yet heavily optimized.  
  - DuckDB invests in robust out-of-core strategies, so DataFusion can be slower on extremely large data sets that exceed memory.  
---
## 5. Performance Comparisons
### 5.1 ClickHouse “ClickBench” Aggregation Queries
- **Benchmark**: A set of queries focusing on aggregations, filters, minimal joins.  
- **Scaling to 172 Cores**:  
  - DataFusion’s performance curve parallels DuckDB’s quite closely.  
  - Both scale linearly and see similar overhead at high concurrency.  
  - Some queries favor DataFusion, some favor DuckDB.  

> **Conclusion**: “No fundamental performance wall or scheduling bottleneck is present in DataFusion up to 172 cores.”
### 5.2 Single-Core Efficiency
- **Same ClickBench**:
  - For highly selective queries or minimal columns scanned, DataFusion was sometimes faster.  
  - For medium cardinality aggregates, they’re similar.  
  - For high cardinality group-bys, DuckDB had a notable edge.  
- **Ongoing Work**: DataFusion community focuses on improving group-by performance with specialized hash table strategies.  
### 5.3 TPC-H
- **Join Ordering “Disasters”**:  
  - Some TPC-H queries can blow up if you get the join order drastically wrong.  
  - DataFusion only does a naive, “semantic” approach, so certain queries were slow.  
  - Minor heuristics were added to avoid the worst plans.  
- **Takeaway**: TPC-H performance is decent if the queries are not deeply nested or if you provide some manual reordering. True cost-based joint reordering is not built-in.
### 5.4 CSV-Focused Benchmarks
- **H2O.ai “grouping” Benchmark**:  
  - DataFusion’s CSV reader is quite efficient.  
  - On certain queries, DataFusion outperformed DuckDB significantly, mainly due to faster CSV parsing.  
  - Other queries requiring advanced statistical aggregates (like `CORR` or `MEDIAN`) were slower in DataFusion because those functions are not as optimized.  

---
## 6. Q&A Highlights

### 6.1 Catalog Statistics & Cost-Based Plans

- **Q:** “Are you maintaining any advanced stats in the built-in catalog?”  
- **A:** “No. You can supply your own table-level or partition-level stats, but the default engine does no real cost-based planning. We rely on heuristic rewrites. If you need deeper rewriting, you can plug in your own optimizer.”

### 6.2 OPD (Ongoing Prototype)
- **OPD** is a new experimental project (Cascades-based approach) that might bring cost-based optimization and advanced rewrites to DataFusion.  
- Very early; no official release. Possibly open-source in the future.
### 6.3 Morsel-Driven Parallelism
- **Q:** “Push-based morsels (DuckDB style) claim better cache locality.”  
- **A:** “DataFusion’s pull-based approach combined with Rust’s async/await already obtains the primary benefits. In practice, the overhead of passing record batches between operators is small compared to the actual compute time. No fundamental barrier to scaling.”  
- **Engineering**: “We tested an alternate push-based approach using [Rayon](https://github.com/rayon-rs/rayon). It was complicated and didn’t show major speedups.”
### 6.4 Out-of-Core / Big Data
- **Q:** “What about large data that doesn’t fit in memory?”  
- **A:** “DuckDB invests heavily in out-of-core approaches. DataFusion does have basic external sorting and hashing, but it’s not deeply tuned. If you need more advanced external memory algorithms, you can implement them or help the community extend them.”
### 6.5 “How many Germans?”
- Joke referencing the numerous highly skilled DB devs from TUM (Technical University of Munich) or other German universities who often make major performance improvements in other open-source engines.  
---
## 7. Final Takeaways
1. **Minimal Built-In Optimizer**  
   - DataFusion’s default “semantic” join ordering might cause performance issues on queries with many joins.  
   - Basic rewrites (pushdown, constant folding) are present and effective for common OLAP.  

2. **Simple & Composable**  
   - The entire pipeline (logical → physical → execution) is *highly extensible*.  
   - If you need sophisticated cost-based reordering or specialized group-by operators, you can add them.  

3. **Tokio & Pull-Based**  
   - Rust’s async/await and Tokio’s work-stealing scheduler provide a robust, easy-to-maintain concurrency model.  
   - The overhead difference vs. push-based morsels is minimal for typical analytics, and the code is simpler to maintain.  

4. **Performance & Future Directions**  
   - DataFusion scales up to hundreds of cores with no fundamental bottlenecks.  
   - Ongoing improvements for advanced group-by algorithms, external memory, and cost-based planning.  

5. **Community & Contributions**  
   - Many PRs come from developers with specialized domain needs.  
   - Open to new collaborator implementations (e.g., OPD cost-based planner, advanced indexing, or stats-based rewrites).  

> **Andrew Lamb's Closing Thought**:  
> “It’s all about engineering effort. The system is modular enough to let you add or replace pieces. If you invest the time (like DuckDB did in group-by or out-of-core), you can match or exceed their performance in a shared open-source ecosystem.”
---
## 8. References and Links
- **[Apache Arrow DataFusion GitHub](https://github.com/apache/arrow-datafusion)**  
- **SIGMOD 2023 Paper**: “DataFusion: A Fast, Embeddable, Modular Query Engine Built on Apache Arrow.”  
- **Tokio**: [Tokio.rs](https://tokio.rs/)  
- **Rayon**: [Rayon GitHub](https://github.com/rayon-rs/rayon) (an alternative Rust concurrency library).  
- **Morsel-Driven Parallelism Paper**: _Sompolski, Podkopaev, et al. Umbra/DuckDB references._  
- **H2O.ai GroupBy Benchmark**: Showcases CSV parsing performance.  

---

## Appendix: Conceptual Diagram of DataFusion’s Planning & Scheduling

```mermaid
flowchart TB
    subgraph Logical Level
    A[User Query: SELECT ...] --> B[SQL Parser / Front-End]
    B --> C[Logical Plan]
    C --> LOpt[Logical Rewrite <br/>(Rule-based)]
    LOpt --> P[Physical Plan]
    end

    subgraph Physical Execution
    P --> POpt[Physical Rewrite]
    POpt --> EP[ExecutionPlan (Operators)]
    EP --> SCHED[Tokio Scheduler <br/>(Pull-based, async/await)]
    SCHED --> Exec[Multi-core Execution <br/> Partitions + Streams]
    Exec --> R[Results (RecordBatches)]
    end
```

1. **Logical Plan**: Basic or custom front-end → parse → rewrites (predicate pushdown, constant folding).  
2. **Physical Plan**: Decide operators (hash join, sort, partial aggregates).  
3. **Execution**: Each physical operator executes in parallel partitions. Data is pulled in batches. Tokio coordinates tasks.  