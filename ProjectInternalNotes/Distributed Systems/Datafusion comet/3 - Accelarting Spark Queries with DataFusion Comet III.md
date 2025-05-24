- **Speaker**: Andy Grove
- **Timestamp**: ~28:37 to ~45:48
- **Key Themes**:  
  - Additional reasons why TPC workloads aren’t fully native  
  - Falling back to Spark for missing ops (bloom filter aggregation, sort-merge joins)  
  - Handling row ↔ column transitions  
  - Potential cost-based approach to decide between Spark vs. Comet  
  - Future directions & community involvement

---

## 1. Why Some Queries Are Not Fully Native Yet

### 1.1 Missing Operators / Features

- **Bloom filter aggregation**:  
  - Spark uses bloom filters in certain TPC-H queries.  
  - DataFusion lacks a direct equivalent.  
  - So Comet hits an operator it can’t handle → partial fallback to Spark.  
  - Work in progress: a pull request has been opened to add bloom filters to DataFusion.

- **Sort Merge Join**:  
  - Recently added in DataFusion, but not fully optimized.  
  - Performance not yet ready for Comet.  
  - → leads to partial fallback to Spark.

**Result**: Fallback means some queries are **not** 100% accelerated → *less overall speedup*.

---
## 2. Row ↔ Column Transitions
### 2.1 Spark’s Approach
- When falling back to Spark’s row-based operators:
  - We move from **native Arrow** (columnar) to **Spark’s InternalRow** or `UnsafeRow` (row-based).
  - Spark inserts “column-to-row” or “row-to-column” transitions where needed.
### 2.2 Current Overhead
- Transitions are **implemented in Spark** (JVM-based).
- Each read can cause a **JNI** boundary crossing:  
  - E.g., `getString(0)`, `getString(1)` for each row → multiple calls.  
  - Extremely inefficient for large volumes.
### 2.3 Potential Improvement
- A **native** “column-to-row” converter could build entire Spark rows in a **single** pass, returning them in one batch.  
- **Ideal scenario**: avoid transitions altogether by supporting the entire query in Comet.
---
## 3. Dictionary Encoding & Data Types

### 3.1 Dictionary Arrays
- DataFusion uses dictionary encoding for strings, etc. to be memory-efficient.
- Currently, Comet **unpacks** dictionary arrays before certain operators like **joins** to simplify logic.
  - This can degrade performance (extra overhead).
- Future: fully preserve dictionary encoding through joins & aggregations to gain speed/memory benefits.
### 3.2 Spark vs. DataFusion Type Mismatch
- **Logical vs. Physical**:  
  - Spark: logical type system (e.g., single `StringType`).  
  - DataFusion/Arrow: many physical forms (`LargeUtf8`, dictionary-encoded `Utf8`, etc.).  
- Still an **ongoing** effort to manage these differences seamlessly.
---
## 4. Missing Physical Optimizations
### 4.1 No Logical Phase in Comet
- Comet jumps **directly** from Spark’s physical plan → DataFusion’s physical plan.
- **DataFusion’s** typical “logical plan” optimizations (e.g., common subexpression elimination) are bypassed.
  - Example: A repeated expression in TPC queries might be evaluated multiple times. 
  - DataFusion’s “logical” CSE pass can’t run because we only use DataFusion’s “physical” side.
### 4.2 Potential Fix
- Implement a **physical** pass that does something similar (CSE, etc.).
- Or hack in a partial “logical rewrite” step *before* the DataFusion physical plan generation.
---
## 5. Potential Use of JIT / Code-Gen
**Question from audience**: *“Any plans to do codegen, like Spark’s Whole-Stage CodeGen?”*
- Andy Grove’s response:
  - DataFusion once had a JIT experiment (e.g., generating LLVM code).  
  - It didn’t yield large benefits.  
  - A well-implemented **vectorized** engine can match or exceed JIT-based row pipelines.  
  - No immediate or concrete plans for advanced code-gen.
**Takeaway**: Comet relies on **vectorization** + **SIMD** (where possible) in Arrow. No special on-the-fly IR → machine code approach right now.
---
## 6. Adaptive Query Execution (AQE) & Scheduling
- **Question**: “Did you have to do anything special for Spark’s AQE?”  
- **Answer**: 
  - “No. Spark still handles all scheduling and stage breakdown. We only intercept the final physical plan on each stage. We do not disrupt Spark’s adaptive scheduling logic.”
---
## 7. Roadmap & Future Plans
### 7.1 Performance Gains & Cost-Based Decisions
- Possibly implement a **cost-based Optimizer** that knows:
  - If a query segment has minimal heavy-lifting but requires multiple row ↔ column transitions, perhaps Comet *hurts* performance.  
  - If big aggregates or wide scans are involved, Comet might offer big gains.  
  - Decide automatically whether to accelerate or fallback.
### 7.2 Beyond TPC-H
- **ETL Benchmarks**:
  - TPC-H is OLAP-centric.  
  - Real Spark usage often includes messy ETL transformations (regex, JSON parsing).  
  - Comet must eventually handle more complex usage patterns.
### 7.3 Ongoing Upstream DataFusion Improvements
- e.g., performance optimizations for group-aggregate queries.  
- Comet inherits these automatically once merged.
---
## 8. Spark UI & Metrics Integration
- Spark provides a nice **web UI** for seeing stages, tasks, plan metrics.
- Comet taps into DataFusion’s **metric system** for native operators:
  - E.g., “time spent in Parquet decode” or “time in arrow vector ops.”  
  - **Propagates** these back to Spark’s metrics → visible in Spark UI.
**Benefit**: Spark devs see a single performance summary across row-based Spark ops and Comet ops.
---
## 9. Future or Community-Driven Efforts
- **Support complex data types**:
  - Structs, arrays, maps, nested arrays, etc.
  - TPC-H only needs primitive types, but real usage often requires advanced types.
- **More Expressions**:
  - Spark has ~200 expressions; Comet covers ~111.  
  - Coverage must expand for more queries to run natively.
- **Python UDF** optimization:
  - Spark’s Python UDF uses PyArrow but still transitions row ↔ column often.  
  - In Comet, we might handle these natively with minimal transitions → big performance gains.
- **“Battle tested”?**  
  - Not yet. Project was open-sourced in Feb, ~v0.3 release.  
  - Community usage is starting, but it’s early. Bugs will surface. Feedback crucial.
---
## 10. Similar Projects & Comparisons

### 10.1 Gluten & Velox
- **Gluten** (Apache incubator) from Intel + community.  
- Similar architecture: intercept Spark’s plan, run a vectorized C++ or Velox backend.  
- Comet took a different approach (Rust + DataFusion + Arrow).  
  - Partly because DataFusion was already in Apache.  
  - Some big companies prefer the stable Apache governance model.
### 10.2 Photon (Databricks)
- Proprietary, closed-source.  
- Presumably also a vectorized columnar engine in C++ behind Spark.  
- No direct code-level knowledge, but conceptually parallel to Comet.
---
## 11. SIMD Usage in Comet (Arrow / DataFusion)
- Comet → DataFusion → Arrow C data kernels.  
- Arrow’s compute kernels rely on **Rust compiler** auto-vectorization, plus specialized intrinsics.  
- No manual assembly or explicit CPU instructions in DataFusion’s code by default.
---
## 12. Reflections on DataFusion’s Early Design Decisions
- **Audience Q**: Any regrets or design decisions in DataFusion that cause friction for Comet?
  - **Andy**:  
    - Nothing Comet-specific.  
    - Some minor regrets re: **logical plan** storing direct trait references instead of a name-based approach.  
    - Could have made plan serialization easier (e.g., using Rust’s `serde`).
---
## 13. Key Takeaways (Timestamp ~28:37–45:48)
1. **Incomplete Operator Coverage** → partial fallback to Spark → performance overhead.  
2. **Row ↔ Column transitions** need optimization or elimination.  
3. **Dictionary encoding** & advanced data types can yield large perf gains once Comet supports them thoroughly.  
4. **Spark UI** integration is strong—metrics from Comet feed seamlessly back into Spark’s UI.  
5. **Community Involvement**:
   - Project is young (v0.3).  
   - Needs testers, bug reporters, and new features.  
   - Potential for big performance wins in certain scenarios, especially as coverage grows.

---

## 14. Example Commands / Quickstart

To **experiment** quickly with Comet:

```bash
# 1. Download or build a Comet jar from Maven
curl -O https://repo1.maven.org/maven2/org/apache/arrow/comet/0.3.0/comet-0.3.0.jar

# 2. Launch spark-shell with plugin:
spark-shell \
  --jars comet-0.3.0.jar \
  --conf spark.plugins=org.apache.arrow.comet.CometPlugin \
  --conf spark.sql.shuffle.partitions=8  # example config
```

Then run typical Spark code. In logs / Spark UI, check for references to `CometExec`, `CometScanExec`, etc.

---

## 15. Quick Visual Summary

```
 Spark Plan w/ Missing Operator
   -> Fallback -> Spark Operator
   -> Column to Row (heavy overhead)
   -> Spark row-based processing
   -> Row to Column transitions again (maybe)
        ↓
       Comet
   -> DataFusion vectorized ops
   -> arrow::compute kernels, SIMD
   -> Possibly partial merges
        ↓
   Final Output
```

---

## 16. Further Resources

- **Comet Source Code**:  
  <https://github.com/apache/arrow/tree/master/java/comet>  
  (Exact path may vary; recently donated to Arrow)
- **DataFusion Docs**:  
  <https://arrow.apache.org/docs/rust/datafusion/>
- **Gluten**:  
  <https://github.com/oap-project/gluten> (incubating)
- **Photon Blog**:  
  <https://www.databricks.com/blog/2020/06/22/introducing-photon.html> (high-level overview)