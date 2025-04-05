## Overview

**Video Context**  
- This talk is part of a “Database Building Block Seminar Series.”  
- DataFusion is introduced as one of the hottest, trending modular systems for analytics.  
- Andrew Lamb is a Staff Engineer at InfluxDB, focusing on DataFusion, Apache Arrow, and the InfluxDB IOx engine.  
- Andrew Lamb is also the Chair of the DataFusion Project Management Committee (PMC) and a member of the Apache Arrow PMC.

**Key Topics Covered**  
1. _What is DataFusion?_  
2. _Why build on a shared query engine ecosystem?_  
3. _High-level architecture and design._  
4. _Implementation details in Rust._  
5. _Performance and benchmarking highlights._  
6. _Use cases and community ecosystem._  
7. _Comparison to other OLAP database designs._

The purpose of these notes is to provide a **PhD-level** deep dive into DataFusion, including references, code examples, architectural diagrams, and potential avenues for further exploration.

---

## 1. Introduction to DataFusion

### 1.1 Motivation

- **Analogy:** DataFusion is for database systems what LLVM is for programming languages.  
  - LLVM unified many compiler back-end optimizations (loop unrolling, register allocation, etc.), empowering language creators (e.g., Rust, Swift, Julia) to focus on language design rather than low-level details.  
  - **DataFusion** aims to be that shared foundation for building OLAP (Online Analytical Processing) systems.  

- **Why a shared foundation?**  
  - Building a high-performance OLAP query engine from scratch requires huge engineering resources (often hundreds of millions of dollars across years).  
  - Many subproblems (e.g., vectorized execution, expression evaluation, query planning) are common across different projects.  
  - DataFusion provides “out-of-the-box” SQL support, plus a flexible and modular architecture to adapt or extend the engine for specialized domains.

### 1.2 Key Attributes

- **Embeddable & Modular:**  
  - Developers can use everything from front-end parsing and SQL planning to the low-level vectorized execution.  
  - Or they can integrate only certain parts—like the expression evaluator or the physical execution engine.

- **Performance-Oriented:**  
  - DataFusion uses Apache Arrow as its memory model and arrow kernels for efficient vectorized computation.  
  - Designed to handle large-scale analytical queries with minimal overhead.

- **Extensibility:**  
  - You can customize function libraries, add custom relational operators, or plug in custom data sources.  
  - The community invests in making sure standard relational rewrites and industrial best practices are well-supported.

- **Rust Implementation:**  
  - Provides memory and thread safety guarantees.  
  - Modern tooling (Cargo, crates.io) simplifies building, integrating, and distributing code.  
  - Encourages a vibrant community of contributors, as the compiler rules out many classes of memory-management errors.

---

## 2. Why DataFusion? 

### 2.1 Shared Infrastructure for OLAP
- **Reduce Reinventing the Wheel:**  
  - New systems can focus on domain-specific logic rather than redoing well-known optimizations.  
  - Reusable vectorized kernels, query planner, and expression evaluator save time.
- **Community-Driven Innovation:**  
  - Large open source contributor base.  
  - Frequent monthly releases with new features, bug fixes, and performance improvements.  
  - Encourages collaboration on the “boring but critical” parts of query engines.

### 2.2 Use Cases
1. **Full Database Systems**  
   - *InfluxDB IOx:* Time-series engine built atop DataFusion for real-time, high-volume ingest and analytic queries.  
   - *Ballista:* A distributed compute platform using DataFusion at each executor node.  
   - Others building specialized analytics engines or next-gen distributed systems.

2. **Component Reuse**  
   - *Execution Engine Integration:* Some projects only need a vectorized execution layer, with their own planning or front-end.  
   - *SQL Parser & Planner:* Others might combine DataFusion’s parser/planner with a custom distributed execution framework.

3. **Table Formats**  
   - Rust implementations of *Iceberg, Delta Lake, Hudi, and Lance* use DataFusion to evaluate expressions for compaction, tombstones (deletions), and incremental data transformations.

4. **Academic Projects**  
   - Rapid prototyping for specialized indexing, novel rewrites, or advanced transformations.  
   - Example: Implementing advanced time series indexing for queries, hooking it into DataFusion’s optimizer to demonstrate performance gains.

---

## 3. Architectural Overview

Below is a conceptual diagram adapted from Andrew Lamb’s SIGMOD 2023 (formerly stated as “the Sigma paper,” but it’s officially SIGMOD) paper.  

```
 ┌───────────────────────────┐
 │         Catalog           │
 │  (Data Sources, Schemas)  │
 └───────────┬───────────────┘
             │
 ┌───────────────────────────┐
 │         Frontends         │
 │ (SQL Parser, DFBuilder,   │
 │  Custom DSLs, etc.)       │
 └───────────┬───────────────┘
             │ (LogicalPlan)
 ┌───────────────────────────┐
 │      Logical Planner      │
 │   (Algebra, Rewrites)     │
 └───────────┬───────────────┘
             │ (PhysicalPlan)
 ┌───────────────────────────┐
 │     Physical Planner      │
 │   (Optimizations, Exec    │
 │    Operators, Partition)  │
 └───────────┬───────────────┘
             │ (Arrow Batches)
 ┌───────────────────────────┐
 │   Execution Engine (Vec)  │
 │ (Operators: Projection,   │
 │  Filter, HashAgg, etc.)   │
 └───────────────────────────┘
```

### 3.1 Main Components
1. **Catalog & Data Sources**  
   - Provides *namespaces* (databases/schemas) and references to data.  
   - Data sources can be in-memory tables, Parquet files, CSV, object stores, or even custom connectors.

2. **Frontends**  
   - A standard SQL parser and a DataFusion builder API.  
   - Tools or languages can generate logical plans using DataFusion’s public interfaces.  
   - Supports extension points for custom DSLs (domain-specific languages).

3. **Logical Planner**  
   - Converts SQL or builder calls into a high-level logical plan (relational algebra).  
   - Applies typical query rewrites, such as predicate pushdown, constant folding, or projection pruning.

4. **Physical Planner**  
   - Converts logical plans into physical operator pipelines, choosing specific algorithms (hash join, sort merge join, etc.).  
   - Applies optimizations for parallelism, partitioning, or resource usage.

5. **Execution Engine**  
   - A vectorized operator pipeline.  
   - Uses Apache Arrow record batches as the universal in-memory representation between operators.  
   - Incorporates specialized kernels for filtering, projections, aggregations, etc.

### 3.2 Modularity and Extensibility

- **Every layer can be customized:**  
  - **Custom Catalog & Data Sources:** E.g., hooking in an object store or specialized format.  
  - **Custom Logical Nodes & Physical Operators:** For domain-specific transformations (e.g., time-series bucketing or advanced geo queries).  
  - **Custom Execution Streams:** Overriding or adding specialized kernels or aggregator implementations.

---

## 4. Implementation Details
### 4.1 Apache Arrow Integration

- **Arrow All the Way Down**: DataFusion uses the Arrow columnar format **internally** for operator boundaries, not just at the input/output edges.  
- **Why Arrow?**  
  1. Performance: Highly optimized, cache-friendly layouts.  
  2. Interoperability: The open-source Arrow ecosystem shares tools (e.g., `arrow-rs` kernels).  
  3. Simplified Development: Reuse well-tested compute kernels for expression evaluation, casts, comparisons, etc.
- **Trade-Off**:  
  - The top-level operator interfaces pass `RecordBatch` (Arrow).  
  - Certain internal data structures (e.g., a hash aggregator’s hashtable) may not be strictly in Arrow format, but the final operator output is always Arrow-based.  
  - This strategy reduces overhead for conversions and leverages the broader Arrow community’s performance improvements.
### 4.2 Rust
- **Why Rust?**  
  - Memory/thread safety helps avoid classic concurrency pitfalls and segmentation faults.  
  - Cargo’s modern dependency management simplifies building large-scale systems.  
  - Growing popularity draws contributors who want to learn or practice Rust.
- **Lessons Learned**:  
  - Rust’s compiler enforces correctness for memory usage and thread safety, drastically reducing certain bug classes.  
  - *Potential Downsides*: The learning curve can be steep, especially around Rust’s ownership and borrowing rules.  
  - Overall, the community sees major benefits in code reliability and maintainability for a high-performance engine.
### 4.3 Multi-Threading and Tokio
- DataFusion queries often run with multi-threaded parallelism.  
- Rust’s concurrency approach prevents data races with strict borrowing rules.  
- Integration with [Tokio](https://tokio.rs/) (or other async runtimes) is possible in distributed settings, though the *core* DataFusion engine does not strictly require async I/O for local queries.
### 4.4 Basic Usage Example

Below is a **simplified** Rust snippet demonstrating how to start a DataFusion project using Cargo:

```rust
// 1) Create a new Rust project:
//    cargo new my_datafusion_project && cd my_datafusion_project
// 2) Add DataFusion to the Cargo.toml:
//    cargo add datafusion
//
// Now, an example usage:

use datafusion::prelude::*;  // standard prelude for DataFusion

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    // Create a new execution context
    let mut ctx = SessionContext::new();
    
    // Register a CSV file as a table
    ctx.register_csv("mytable", "path/to/file.csv", CsvReadOptions::new()).await?;
    
    // Run a SQL query
    let df = ctx.sql("SELECT some_col, COUNT(*) as cnt FROM mytable GROUP BY some_col").await?;
    
    // Collect results into memory (Vec of RecordBatch)
    let results = df.collect().await?;
    
    // Print results
    for batch in &results {
        println!("{:?}", batch);
    }
    
    Ok(())
}
```

**Key Points**  
- `SessionContext` manages the catalog and the query execution environment.  
- DataFusion automatically infers CSV schema (or you can specify).  
- The final results are Arrow `RecordBatch` structures, which can be iterated or converted for output.
---
## 5. Performance
### 5.1 Industrial Best Practices
- **Query Execution**  
  - Vectorized operators operating on Arrow-formatted data in cache-friendly batches.  
  - Hash-based group-bys, joins, dictionary optimizations, etc.  
  - Predicate pushdown if the data source supports it (e.g., Parquet, CSV partial pushdown).
- **Benchmarking**  
  - TPC-H or TPC-DS style queries used to measure baseline performance.  
  - Internal microbenchmarks for kernel-level operations (e.g., filter, arithmetic, cast).  
  - Comparisons with systems like ClickHouse show DataFusion is competitive and sometimes better on certain queries.
### 5.2 Tuning
- **Parallel Execution**  
  - Automatic multi-threaded scheduling of partitions for CSV, Parquet, or custom data. 
  - Manual control over concurrency in advanced use cases.
- **Plan Optimizations**  
  - Simplified cost-based rewrites (predicate pushdown, projection pruning, etc.).  
  - Extensible rule-based optimization.  
  - Ongoing development for more advanced optimizers (e.g., join reordering, multi-join heuristics).
---
## 6. Code & Examples
### 6.1 Creating a Custom Function
You can extend DataFusion’s function registry with your own scalar or aggregate functions:
```rust
use datafusion::prelude::*;
use datafusion::logical_expr::{create_udf, Volatility};
use datafusion::arrow::array::{Int64Array, ArrayRef};
use std::sync::Arc;

fn my_add_function(args: &[ArrayRef]) -> datafusion::error::Result<ArrayRef> {
    let left = args[0].as_any().downcast_ref::<Int64Array>().unwrap();
    let right = args[1].as_any().downcast_ref::<Int64Array>().unwrap();
    let res: Int64Array = left.iter().zip(right.iter())
        .map(|(l, r)| l.unwrap() + r.unwrap())
        .collect::<Vec<i64>>()
        .into();
    Ok(Arc::new(res))
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let mut ctx = SessionContext::new();

    // Create a User-Defined Function
    let my_add = create_udf(
        "my_add",
        vec![DataType::Int64, DataType::Int64], // input types
        Arc::new(DataType::Int64),
        Volatility::Immutable,
        Arc::new(my_add_function),
    );

    // Register the UDF
    ctx.register_udf(my_add);

    // Example usage:
    let df = ctx.sql("SELECT my_add(10, 32) AS result").await?;
    let results = df.collect().await?;
    println!("{:?}", results);

    Ok(())
}
```

### 6.2 Extending with a Custom Physical Operator

- **Goal:** Suppose you want a specialized filter operator that uses an external GPU library.  
- **Approach:**  
  1. Create a custom logical plan node.  
  2. Create a corresponding physical plan node.  
  3. Override the `execute()` method to run your GPU-accelerated filter kernel.  
  4. Register the custom planner rule so that DataFusion’s query planner knows when to substitute your custom operator.

Pseudo-code for your custom operator might look like:

```rust
// Pseudocode outline

struct MyGpuFilterExec {
    input: Arc<dyn ExecutionPlan>,
    gpu_filter_expr: Arc<dyn PhysicalExpr>,
}

impl ExecutionPlan for MyGpuFilterExec {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }
    fn execute(&self, partition: usize, context: Arc<TaskContext>)
        -> Result<SendableRecordBatchStream> 
    {
        // 1) Get the stream from input
        let input_stream = self.input.execute(partition, context.clone())?;

        // 2) Wrap in a GPU filter stream
        let stream = MyGpuFilterStream {
            input_stream,
            gpu_filter_expr: self.gpu_filter_expr.clone()
        };
        Ok(Box::pin(stream))
    }
    // Implement statistics, schema, etc.
}
```

---

## 7. Ecosystem & Community

### 7.1 Community Growth

- Large GitHub contributor base with hundreds of contributors.
- Monthly releases, each with a well-documented CHANGELOG.
- User guides for *both* end-users (SQL queries, engine usage) and integrators (custom operators, UDFs, data sources).

### 7.2 Active Projects Built on DataFusion

- **InfluxDB IOx**: Time series engine with DataFusion for SQL-based queries.  
- **Delta Lake, Iceberg, Hudi, Lance**: Rust-based table formats using DataFusion for expression evaluation and I/O tasks.  
- **Ballista**: Distributed compute engine originally part of the DataFusion repository; since separated but remains closely integrated.  
- **Spark DataFusion “Comet”**: (Presented by Andy Grove) bridging Spark queries to DataFusion’s execution engine.  

---

## 8. Comparison with Traditional Systems

- **Tightly Integrated DBs**: Classic DBs often have highly specialized, single-purpose code for scanning, expression evaluation, etc. Ties them to a single storage format or concurrency model.  
- **DataFusion Approach**: “All battery included,” but each piece is modular. The use of Arrow standardizes the memory representation.  
- **Performance Myths**:  
  - Skepticism: “Modularity kills performance.”  
  - Practice: DataFusion demonstrates that carefully chosen boundaries (Arrow’s columnar format) and optimized kernels can yield near state-of-the-art performance while retaining extensibility.

---

## 9. Potential Research & Extension Ideas

1. **Cost-Based Optimization (CBO)**  
   - Further exploration into advanced join ordering or multi-join optimization strategies. 
   - Integration of advanced histograms or sampling-based cardinality estimation.

2. **Vectorized or GPU Operators**  
   - Official frameworks to incorporate GPU libraries for specialized computations.  
   - Potential synergy with Arrow’s CUDA support in the arrow-rs ecosystem.

3. **Custom Data Formats**  
   - Time-series or multi-dimensional indexing with specialized query rewrites.  
   - Extending the file format plugin system for new HPC or scientific data formats.

4. **Streaming or Real-Time**  
   - DataFusion as a streaming engine: Combining batch and incremental logic.  
   - Adapting relational operators for continuous queries (some prototypes exist already in user code).

---

## 10. Final Takeaways

- DataFusion’s main value is **reusable, high-performance building blocks** for **OLAP** queries.  
- Leverages **Apache Arrow** in-memory format to provide vectorized execution.  
- **Rust** ensures memory/thread safety, improving reliability and maintainability in open-source development.  
- Offers a flexible entry point for both **research** (prototyping new DB ideas) and **production** (time-series, distributed analytics, table formats).  

**Quote from Andrew Lamb:**  
> “It’s basically the LLVM for databases... so you don’t have to reimplement the same vectorized query engine all over again.”

---

## 11. References & Further Reading

- **DataFusion Official Repository**  
  [Apache Arrow DataFusion (GitHub)](https://github.com/apache/arrow-datafusion)

- **Apache Arrow**  
  [Apache Arrow Official Site](https://arrow.apache.org/)

- **DataFusion at SIGMOD 2023**  
  - Paper: *“DataFusion: A Modern, Extensible Query Engine Built on Apache Arrow”* by Andrew Lamb et al.

- **Related Talks**  
  - _Andy Grove’s “DataFusion Comet”_ (next in the Database Building Blocks Seminar Series).  
  - _InfluxDB IOx_ design docs and references for a production system using DataFusion heavily.

---

## Appendix: Visualization of a Typical Query Lifecycle

```mermaid
flowchart TB
    A[User SQL Query] --> B[SQL Parser<br/>Logical Plan]
    B --> C[Logical Optimizer<br/>(Pushdown, Pruning)]
    C --> D[Physical Planner<br/>(HashJoin, Partitioning)]
    D --> E[Execution Engine<br/> (Arrow Batches)]
    E --> F[Results as RecordBatches]
```
- **SQL Parser**: Converts raw SQL string into an abstract syntax tree.  
- **Logical Plan**: Converts AST into relational operators (projection, filter, join, etc.).  
- **Logical Optimizer**: Rewrites the logical plan (e.g., filter pushdown).  
- **Physical Planner**: Chooses physical algorithms (hash join vs. sort-merge join, etc.).  
- **Executor**: Runs vectorized operators in parallel, producing Arrow record batches.  
- **Result**: Can be displayed in console, stored in memory, or converted to other data formats.  
---
## Linked Notes
- [[Apache Arrow Overview]]  
- [[Building a Custom Data Source in DataFusion]]  
- [[Optimizing Queries with DataFusion Logical Rewrites]]  
- [[Rust Memory Safety in Data-Intensive Systems]]  