## 1. Intro and Background

### 1.1 Speaker’s History

- Speaker (Andy Grove) has **decades** of experience in software engineering:
  - 1st decade:
    - Building enterprise apps w/ RDBMS (Relational DB) backends.
  - Next ~decade:
    - Focus on data frameworks (no specific mention of which, but presumably Hadoop, Spark, etc.).
  - Most recent **16 yrs**: 
    - Focus on database infrastucture
      - Created a **database sharding** product.
      - Built a custom SQL engine on top of Spark.
      - Learned about Spark deeply, started hacking on DataFusion.
  - Past 5 yrs: 
    - Acceleration of Spark (NVIDIA’s Spark RAPIDS, Apple, etc.).
    - Specifically, using DataFusion’s Rust-based approach.

### 1.2 DataFusion Project History (brief)

- **Started** in 2017 as a personal project. 
  - **Initial goal**: A “modern Apache Spark” with Rust, but found it *too ambitious* for a weekend project.
- **Pivoted** to a simpler approach: an embeddable query engine.
  - Inspiration from “Designing Data-Intensive Applications,” etc.
  - Strives for a modular design w/ a well-defined separation between:
    - Parser & Planner
    - Logical Plan
    - Physical Plan
- **Community Growth**:
  - Donated to Apache Arrow in 2019.
  - Eventually top-level in 2023 (in Apache).
- **Ballista**: 
  - Attempt to build a distributed query engine **on top** of DataFusion.
  - Did not gain as much traction; is not very active now.

---

## 2. DataFusion as a Building Block

### 2.1 Quick Recap

- DataFusion’s **Core**:
  - A set of **libraries** (Rust) for:
    - SQL parsing, logical & physical planning
    - Execution (vectorized, columnar data)
    - In-memory data or streaming from disk
- Already well-known for being a foundation to build new systems on.

### 2.2 Distributed / User-Facing Components

1. **DataFusion Python Bindings**
   - Access DataFusion from a Jupyter/Colab environment.
   - Great for single-node usage or local dev.

2. **Comet** (Focus of this talk)
   - Not a “standalone distributed DataFusion,” but an **acceleration layer** that hooks into Spark’s APIs. 
   - Replaces Spark’s row-based operators w/ Rust-based DataFusion operators under the covers.
1. **DataFusion-Ray** 
   - Another distributed approach using **Ray** for scheduling + DataFusion for query execution.
1. **Ballista** 
   - The older distributed approach that is mostly inactive now.
---
## 3. Performance Benchmarks (TPCH)

> *Speaker references a small-scale TPC-H benchmark on single node to demonstrate relative performance.*
- **Baseline**: Spark
- **Comet**: faster than Spark, but not as fast as other DataFusion-based solutions yet.
- **Ray** and **Ballista**: ~2.5x speedup over Spark.
- **DataFusion Python**: ~3.6x faster than Spark (single process, no overhead of distributed shuffle, etc.).
### 3.1 Reasoning for Differences
- Spark is inherently distributed → overhead of partitioning, shuffling, writing files between stages.
- **DataFusion Python**: 
  - No cluster overhead. 
  - Intra-process data transf is basically passing pointers, so minimal overhead.
- Distributed systems pay the cost of **exchanges** (shuffle, broadcast, etc.).
---
## 4. Parallel vs. Single-Process Execution
### 4.1 Exchange Operators
> **Key difference**: single-process flow can simply pass memory pointers among threads, while distributed execution must “materialize” data, often to disk, plus network overhead.
- On a typical distributed plan (e.g., TPC-H Q1):
  - **Parallel Hash Aggregate** runs in each partition → partial group results.
  - A **repartition** step occurs (exchange op).
  - A **final aggregate** merges partial groups → single result set.
  - Possibly another partition shuffle for final operations (e.g., a sort).

### 4.2 Broadcast vs. Shuffle
- **Shuffle**: 
  - re-distributes data among all executors (expensive).
- **Broadcast**:
  - sends smaller build side of a join to all executors.
  - can help reduce overhead if it’s small.
---
## 5. Overview of Apache Spark
![[Screenshot 2025-04-05 at 9.52.59 AM.png]]
### 5.1 Spark Origins
- ~10 yrs old from AMP lab.
- Mature cost-based optimizer, widely used, integrates w/ many filesystems, large community.
### 5.2 Strengths
- **One-liner** dev experience:
  - Can test on laptop (`spark.local[*]` or `spark-shell`).
  - Scale up to hundreds or thousands of nodes as needed.
### 5.3 Downsides
1. **Row-based** (Volcano model)
   - Lots of overhead: multiple function calls per row.
   - Spark uses “Whole Stage CodeGen” to mitigate overhead by generating Java code for pipelines → big performance improvement but still has limitations (e.g., code size, complexity, corner cases).
2. **JVM-based** 
   - Memory overhead, GC can hamper performance in large scale scenarios, often slower than native code.

---

## 6. Trend: Accelerating Spark Instead of Replacing

### 6.1 Why Accelerate Spark?
- Spark is entrenched in many organizations.
- Re-writing entire infrastructure on a new engine is rarely feasible.
- DataBricks **Photon** is a prime example:
  - C++ vectorized engine behind Spark APIs.
  - Paper claims ~3x speedups on average.
### 6.2 Open-Source Accelerators
1. **Comet** (focus of talk)
2. **Apache Gluten** 
   - Evolved from “Gazelle” project by Intel.
   - Has multiple backends (e.g., Velox, ClickHouse).
3. **NVIDIA Spark RAPIDS** 
   - Uses GPU-based libraries (cuDF, etc.).
---
## 7. Comet: Spark Acceleration via DataFusion
### 7.1 High-Level Concept
- Comet hooks into Spark’s **physical plan**.
- Instead of letting Spark’s row-based CPU operators run, it translates to **DataFusion** behind the scenes.
#### Goals
- Maintain **100% spark compabitliy**.
  - This includes matching spark’s function edge-cases, output types, etc.
  - Must handle version differences (Spark 3.2 vs 3.3 vs 3.4…).
- Currently supports **11 operators** and **111 expressions** natively (Spark has 200+ expressions, so more coverage needed).
### 7.2 Architecture Diagram (Textual Representation)
```
 Spark Driver
   |  (Spark Plan) 
   v
 +-----------------------------------+
 | Comet Translator (Scala -> Proto) |
 +-----------------------------------+
     |  (protobuf IR)
     v
 Spark Executor
   |  JNI -> Rust
   v
 +-----------------------------------+
 |  Comet Planner (Rust)             |
 |   -> DataFusion Physical Plan     |
 |   -> DataFusion Execution         |
 +-----------------------------------+
    ...
    [Shuffle Files or Memory] 
    ...
   [Results to Next Stage]
```
![[Screenshot 2025-04-05 at 9.53.13 AM.png]]

- **Driver** (left):
  - Receives user queries via DataFrame or Spark SQL.
  - Spark’s own plan (in Scala class structure) is created.
  - Comet uses its translator to generate an IR in **protocol buffers**.
- **Executors** (right):
  - Receive the IR (protobuf).
  - Comet’s Rust side translates that IR → DataFusion physical plan.
  - Execution done in native, vectorized DataFusion.
  - Output is either returned or written to shuffle for next stage.
### 7.3 Typical Flow
1. **Spark** builds the physical plan (Scala classes).
2. **Comet** “hijacks” that plan, rewriting operator references into DataFusion forms in an IR.
3. IR is sent to the executor.
4. On the executor, we do: 
   - Rust code → build an actual DataFusion plan.
   - Evaluate operators, produce results.
### 7.4 Full Native vs Partial Acceleration
- *Ideal:* entire plan is recognized by Comet → fully vectorized w/ no fallback to Spark operators.
- *Non-ideal:* if an operator or expression is *not yet supported*, we partially accelerate. Some operators fallback to Spark's row-based approach → some overhead from crossing the JNI boundary multiple times.
---

## 8. Q&A Snippets

> The transcript has a short Q&A about how the IR is shaped, how it’s transmitted, and how Comet compares to other solutions.

1. **Sidecar vs. in-process?**
   - Comet code runs **within** the **Spark executor** process (via JNI).  
   - No separate process.
1. **Substrait?** 
   - Currently, Comet uses a custom protobuf-based IR. 
   - Could switch to Substrait in the future (it’s a promising standard). 
   - For now, narrower scope = simpler engineering.
1. **Matching Spark Semantics** 
   - This is tricky. Spark has many corner cases (especially in type conversions, date/time funcs, etc.).
   - DataFusion tries for Postgres-like semantics but must adapt to Spark’s quirks.
---
## 9. Examples & Code Snippets

### 9.1 Basic Spark Query Example

Below is a simple Spark snippet to illustrate how it might get accelerated by Comet:

```scala
// Scala Spark code
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("SparkCometExample")
  .config("spark.plugins", "org.apache.arrow.comet.CometPlugin") // hypothetical plugin enabling
  .getOrCreate()

// Simple TPC-H Q1 style query
val df = spark.read
  .format("parquet")
  .load("/path/to/lineitem.parquet")

df.createOrReplaceTempView("lineitem")

val result = spark.sql("""
    SELECT
      l_returnflag,
      l_linestatus,
      sum(l_quantity) as sum_qty,
      sum(l_extendedprice) as sum_price
    FROM lineitem
    GROUP BY l_returnflag, l_linestatus
""")

result.show()
```

- **Comet** intercepts the execution plan after Spark’s Catalyst Optimizer does its job.
- Translates to a DataFusion plan in Rust.

### 9.2 Potential Rust DataFusion Plan (Pseudo-Code)

(*This is approximate pseudo-code to show how DataFusion might see the plan.*)

```rust
use datafusion::prelude::*;

fn run_query_in_datafusion() -> Result<()> {
    // Create a DataFusion session
    let mut ctx = SessionContext::new();

    // Register table from parquet
    ctx.register_parquet("lineitem", "/path/to/lineitem.parquet", ParquetReadOptions::default())?;

    // DataFusion SQL
    let df = ctx.sql(r#"
        SELECT
          l_returnflag,
          l_linestatus,
          SUM(l_quantity) as sum_qty,
          SUM(l_extendedprice) as sum_price
        FROM lineitem
        GROUP BY l_returnflag, l_linestatus
    "#)?;

    let results = df.collect()?;
    
    // results is a vector of RecordBatches
    println!("{:?}", results);
    Ok(())
}
```

- In the Comet approach, you never actually write this code. It's done behind the scenes. 
- The logical plan is built from Spark’s instructions → compiled to DataFusion’s plan → executed natively.

---

## 10. Known Challenges

1. **Spark Behavioral Differences**: 
   - E.g., Spark's cast from string to date might produce different results than Postgres.
   - Comet must replicate these corner cases to remain “drop-in” for Spark customers.

2. **Operator Coverage**: 
   - Only 111 expressions so far (Spark has 200+). 
   - More complex UDFs, time zone manipulations, window functions might need special handling.

3. **Shuffle Overhead**: 
   - Even with a fast engine, distributed queries still require data to be written to disk & read over network.
   - Large queries with many stages might be slower if partial fallback to Spark row-based operators occurs in multiple stages.

---

## 11. Visual Diagram of Data Flow

A conceptual (ASCII) diagram you might put in your notes:

```
+------------------------------------+
| Spark Driver                       |
|  - Build Spark Plan                |
|  - Pass plan to Comet translator   |
+-----------------------+------------+
                        |
                        v (protobuf)
+------------------------------------+
| Spark Executor                     |
|  - JNI boundary calls to Rust      |
|  - Comet engine -> DataFusion Plan |
|  - Execution in DataFusion         |
+-------------+----------------------+
              |
              | (shuffle, broadcast)
              v
+------------------------------------+
| Next Stage or final result         |
+------------------------------------+
```

---

## 12. Summary Points / Next Steps

- **Comet** is an **exciting** new open source accelerator for Spark that leverages **Apache DataFusion** in Rust for improved performance.
- Currently sees noticeable **speedups** (but not yet matching purely in-process DataFusion).
- Plan to expand operator coverage and fix corner-case semantics so it can be a **drop-in** for all typical Spark queries.

**Future directions**:
- Potential integration with **Substrait** for a more generalizable IR.
- Possibly support GPU acceleration or other specialized hardware (not explicitly mentioned, but an open possibility).
- Further alignment with standard query semantics for full fidelity with Spark’s behavior.

---

## 13. Personal Reflections / Implementation Details

- **PhD-level concerns**:
  - Potential research on bridging the gap between row-based Catalyst operators and vectorized batch operators in a fully dynamic environment.
  - Hybrid execution strategies or partial codegen for narrower usecases.
  - Evaluate the cost of shuffle + network IO vs. purely local pointer-based columnar operations.
- **Implementation language**:
  - Heavy use of Rust for DataFusion’s core → memory safety and performance benefits over Java.
- **Testing**:
  - Must confirm wide coverage of Spark’s built-in functions (especially date/time and string manipulations).
  - Must ensure correctness for corner cases (NaN, Infinity, locale differences, etc.).

---

## 14. Further Reading / Resources

- **Apache DataFusion**: [https://github.com/apache/arrow-datafusion](https://github.com/apache/arrow-datafusion)
- **Comet Source Code**: (Likely within the same repo or under “arrow-compute-comet” in the Arrow GitHub sub-tree)
- **Photon White Paper** (by Databricks): [https://www.databricks.com/blog/2020/06/22/introducing-photon.html](https://www.databricks.com/blog/2020/06/22/introducing-photon.html) (older blog post, not exactly a formal paper, but references).
- **Gazelle / Gluten**: [https://github.com/oap-project/gluten](https://github.com/oap-project/gluten)

---

## 15. Possible Experiments

1. **Local TPC-H**:
   - Try scale factor 1 → run Spark native vs. Spark + Comet → measure speedups.
2. **Compare** with DataFusion Python:
   - See overhead from distributed vs. in-process approaches.
3. **Profiling**:
   - Use `async-profiler` or `perf` to see JNI transitions overhead or memory usage under heavy aggregation.

---

## 16. Key Takeaways

- Using Comet to accelerate Spark is a **pragmatic** approach (rather than rewriting entire pipelines).
- Native vectorized execution (rust-based) generally outperforms Spark’s row-based approach, albeit partial coverage is a limitation.
- The overhead of **distributed** scheduling and shuffle remains a fundamental cost, but **Comet** can reduce CPU overhead significantly.

---

> **Note**: These notes contain some minor typos by design (e.g., “compabitliy,” “IO,” “Spark's scolar classes” if that was used, etc.) to reflect typical note-taking. Adjust as desired in your Obsidian vault for clarity!