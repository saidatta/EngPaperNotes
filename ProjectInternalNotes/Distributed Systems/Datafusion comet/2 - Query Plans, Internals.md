## 1. Configuration & Usage
### 1.1 How to Use Comet
- **Maven distribution**:
  - Obtain the Comet `.jar` (or set of jars) from Maven.
  - Add it to your Spark application’s classpath.
- **Spark plugin**:
  - In `spark-defaults.conf` or via code, specify:
    ```bash
    spark.plugins=org.apache.arrow.comet.CometPlugin
    ```
  - This effectively replaces portions of Spark’s physical execution with Comet’s vectorized Rust-based operators.

**Takeaway**: Setup is straightforward—**one config** line and a **library** on the classpath.

---
## 2. Example Query Plans (TPC-H Q1)

### 2.1 Top-Level Spark Physical Plan
![[Screenshot 2025-04-05 at 2.49.14 PM.png]]
Using **TPC-H Query 1** as an example:
1. Spark’s Catalyst builds a **Physical Plan** (Scala classes).
2. Comet *translates* that plan into a Comet-labeled version of the same structure (still Scala classes, but “prefixed” or “wrapped” by Comet classes).
![[Screenshot 2025-04-05 at 2.49.39 PM.png]]
Below are textual excerpts of how the plan might look:

- **Spark Physical Plan**:
  ```
  *(2) HashAggregate(keys=[...], functions=[...])
  ...
  Exchange ...
  ...
  *(1) FileScan parquet ...
  ```
![[Screenshot 2025-04-05 at 2.50.07 PM.png]]
- **Comet Translated Plan (Driver side)**:
  ```
  CometHashAggregate(...)
  CometExchange(...)
  CometColumarExchange(...)
  CometScanExec(...)
  ...
  ```
  - Note that some operators have `CometExchange` and others `CometColumnarExchange`.  
  - **Reason**: Comet has two flavors of exchange:
    1. **Native** arrow-based shuffle (ideal, but limited partitioning).
    2. A fallback “mixed” approach that still uses some JVM-based mechanisms.
### 2.2 On the Executor Side
- The Spark executor receives a **sub-plan** (i.e., per-stage plan).
- Comet code (via JNI) translates that **Scala sub-plan** → **DataFusion** Physical Plan in Rust.
- For TPC-H Q1, there might be **3 query stages**, each with a **native** Comet plan:
  - **Stage 1**: Read Parquet, partial aggregate, shuffle output.
  - **Stage 2**: Read shuffle output, final aggregate or sort, produce next shuffle, etc.
  - **Stage 3**: Possibly final merges or final sort.
**Important**: 
- If an operator in the plan is unsupported, it may fall back to Spark’s row-based execution.  
- The ideal scenario is a *fully* Comet-translated plan.

---
![[Screenshot 2025-04-05 at 9.53.42 AM.png]]
## 3. Accelerated Parquet Read

### 3.1 Motivation

- Spark has two data source APIs:
  - **V1**: Row-based Parquet scanning.
  - **V2**: Columnar-based, but columns are loaded into Spark’s internal columnar format, not Arrow.
- Comet / DataFusion use **Arrow** columns.
  - Eliminating an extra conversion step (Spark columnar format → Arrow) can significantly reduce overhead.
- **Native** decoding with Rust further speeds up Parquet I/O.

### 3.2 Diagram: Comet Parquet Reader

```
Spark Executor Process
+--------------------------------------------------+
| JVM (Spark)                        NATIVE (Rust) |
|   - CometParquetReader  --------->   [Decoder]   |
|   - Reads filesystem bits           -> Arrow RB   |
|   - Volcano "next()" calls <------- CometExec...  |
+--------------------------------------------------+
```

1. **Top-left**: Comet’s Parquet Reader (JVM side) reads raw bytes using Spark’s built-in FS connectors (HDFS, S3, etc.).
2. **JNI call** into Rust code, which **decodes** those bytes → **Arrow** `RecordBatch`.
3. Batches flow through the “Volcano” iteration model (JVM calls `next()`, which triggers the next portion of data).
4. The final RecordBatch can:
   - Write to shuffle if needed.
   - Potentially pass back to Spark if partial fallback is required.

**Note**: This approach ensures:
- We leverage Spark’s existing FS ecosystem (security, distributed reads).
- We decode data in **native** code for efficiency.

---

## 4. Challenges & Quirks

### 4.1 Data Types: Logical vs Physical

- **Spark**: Has a “logical” type system (e.g., `StringType`).
- **Arrow/DataFusion**: Has multiple *physical* representations (e.g., `LargeUtf8`, `Dictionary<Utf8>`, etc.).
- **Issue**: When jumping directly from Spark’s physical plan → DataFusion’s physical plan, we skip the normal data type coercion logic that *would* occur in DataFusion’s logical planning.
- **Current Workaround**: Comet must replicate some type-coercion logic. This can lead to suboptimal usage of dictionary-encoded arrays.

### 4.2 Spark Compatibility

- Comet aims for **100%** Spark behavior for edge cases:
  - Some expressions are *almost* correct but differ in corner conditions.  
  - They can be disabled by default or put behind flags (user chooses to enable if safe).

#### Examples of Quirks

1. **Negative Zero**:
   - Rust & Java handle `-0.0` vs `+0.0` differently (IEEE-754 subtlety).
   - Spark has special logic to “normalize” negative zeros in certain arithmetic ops → Comet must match that logic.
2. **Spark `AN` mode**:
   - If enabled, numeric overflow triggers an exception.
   - If disabled, Spark produces `null` on overflow.
   - Comet must replicate both behaviors.
3. **Casting**:
   - Spark supports many partial date/time formats, e.g., `"T2"` meaning `hour=2`.
   - Postgres/DataFusion do not (by default).  
   - Comet must implement custom parsing logic to mirror Spark exactly.
4. **String Ops**:
   - `upper()`, `lower()` might differ with non-ASCII characters between Java & Rust’s UTF-8 handling.  
   - Often turned off by default or labeled “ASCII only.”

---

## 5. Testing for Compatibility

### 5.1 Spark Tests

- **Run Spark’s entire test suite** with Comet plugin enabled.
- If Comet is fully translating an operator, verify the results match Spark’s baseline.

### 5.2 Comet’s Unit and Integration Tests

- **Random query generation**:
  - Create random schemas, random Parquet data, random queries.
  - Compare Spark native vs. Spark+Comet results → must match bit-for-bit (or near enough, e.g., floating rounding).
- **Fuzz Testing**:
  - More intense version of random queries & data.
  - Catches weird corner cases (e.g., negative zero division).
- **Example of a found bug**:
  - `SELECT a, b, a/b` where `b` is `-0.0`. 
  - Spark returned `null`, Comet originally returned `Infinity`.
  - Real fix: match Spark’s behavior for negative zero divisions.

```text
# Pseudocode excerpt of fuzz test
Query = "SELECT col_a, col_b, (col_a / col_b) as div_val
         FROM random_table
         ORDER BY div_val"
```
- => Ensuring Comet & Spark match numeric edge-cases (`Infinity`, `NaN`, etc.).

---

## 6. Performance Snapshot (Revisited)
### 6.1 Comparing Against Other DataFusion-Based Engines
- **Comet** not yet as fast as:
  - **Ballista** (single node or distributed).
  - **DataFusion-Ray**.
- **Reason**: Overhead from partial fallback to Spark operators, plus general overhead of distributed shuffle inside Spark’s architecture.
### 6.2 Larger Scale (TPC-DS 1TB on Cloud)
- Even on more “enterprise-scale” 1TB TPC-DS:
  - Comet shows definite **acceleration** on many queries vs. vanilla Spark.
  - Overall speedup is still modest.
  - Additional operator coverage & performance tuning in Comet are ongoing.
---
## 7. Example Visualizations & Code Snippets

### 7.1 Minimal “Hello Comet” Spark App

```scala
import org.apache.spark.sql.SparkSession

object HelloComet {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("HelloComet")
      .config("spark.plugins", "org.apache.arrow.comet.CometPlugin")
      .getOrCreate()

    // Some sample data
    val df = spark.range(0, 1000000).toDF("id")
    df.createOrReplaceTempView("t1")

    // Simple query
    val result = spark.sql("""
      SELECT id, id+1 AS next_id
      FROM t1
      WHERE id % 100 = 0
    """)

    result.show()
    
    spark.stop()
  }
}
```

- This code will invoke Comet for the *filter*, *projection*, possibly *scan*.
- Check logs for references to `CometExec`, verifying acceleration occurred.
### 7.2 Quick ASCII Diagram: Comet + Parquet + Shuffle
```
+------------------+            +----------------------+
| Spark Driver     |            | Spark Executor       |
|  - Plan via SQL  |  Plan -->  |  - Comet runs w/ JNI |
+------------------+            |  - Reads Parquet     |
                                |  - Shuffle Write     |
                                +---------+------------+
                                          |
                                          | Shuffle
                                          v
                                +----------------------+
                                | Next Spark Stage     |
                                |  Possibly Comet too  |
                                +----------------------+
```
---
## 8. Summary of This Section
1. **User Experience**:  
   - As simple as including a `.jar` and adding `spark.plugins=...CometPlugin`.
2. **Plan Translation**:  
   - Driver side: from Spark’s Catalyst physical plan → *Comet-labeled* plan in Scala.  
   - Executor side: further translation to DataFusion physical plan.
3. **Accelerated Parquet**:  
   - Reads & decodes natively into Arrow, skipping a costly format conversion step.
4. **Spark Quirks**:  
   - Must handle negative zeros, partial date/time casts, `AN` mode, etc.
   - Testing is done with Spark’s own tests + fuzz tests → ensures behavior matches.
5. **Performance**:  
   - Encouraging speedups; still not as optimized as a pure DataFusion engine (due to Spark overheads + partial coverage).  
   - TPC-DS 1TB results show real (but not massive) gains, with more improvements on the way.
---
## 9. Key Takeaways & Next Steps

- **Implementation Maturity**:  
  - Comet is still under heavy development. Operator coverage and specialized partitioning are big areas of improvement.
- **Testing**:  
  - Vital for ensuring Spark equivalence; fuzz testing has caught corner-case issues around floating arithmetic, negative zero, etc.
- **Long-Term Vision**:  
  - Achieve near-complete Spark coverage → minimal or no fallback to Spark row-based operators.
  - Contribute some advanced type-coercion logic back to DataFusion’s physical layer.
---
## 10. Further Reading
1. [Comet Project in Apache Arrow (GitHub)](https://github.com/apache/arrow/tree/master/...)
2. [DataFusion Documentation](https://arrow.apache.org/docs/rust/datafusion/)
3. [Spark Plugin Mechanisms](https://spark.apache.org/docs/latest/core-api-guide.html#plugins)
4. [TPC-DS 1TB Benchmark Examples](https://github.com/databricks/tpcds)