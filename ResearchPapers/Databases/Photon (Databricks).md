Tags: #databricks #spark #photon #sql #c++ #systems #query-engine #vectorized #lakehouse

---
## Table of Contents
1. [[#Introduction|Introduction]]  
2. [[#Lakehouse-Architecture-Recap|Lakehouse Architecture Recap]]  
3. [[#Motivation-and-Bottlenecks|Motivation and Bottlenecks]]  
4. [[#High-Level-Design|High-Level Design]]  
   1. [[#Columnar-Execution|Columnar Execution]]  
   2. [[#Native-C++|Native C++]]  
   3. [[#Pull-Based-Operator-Pipeline|Pull-Based Operator Pipeline]]  
   4. [[#Interpreted-Vectorized-Execution|Interpreted Vectorized Execution]]  
5. [[#Code-Generation-vs.-Interpreted-Engines|Code Generation vs. Interpreted Engines]]  
6. [[#Photon-Implementation-And-Optimizations|Photon Implementation And Optimizations]]  
   1. [[#Operator-Fusion|Operator Fusion]]  
   2. [[#Position-Vectors|Position Vectors]]  
   3. [[#Compiler-Hints-in-C++-(restrict-and-template)|Compiler Hints in C++ (restrict and template)]]  
   4. [[#Vectorized-Hash-Tables|Vectorized Hash Tables]]  
7. [[#Adaptive-Execution|Adaptive Execution]]  
8. [[#Memory-Allocation-and-Management|Memory Allocation and Management]]  
   1. [[#Buffer-Pool|Buffer Pool]]  
   2. [[#Spilling|Spilling]]  
9. [[#Integration-with-Spark|Integration with Spark]]  
10. [[#Conclusion|Conclusion]]  

---

## Introduction
**Photon** is a query engine developed by Databricks (paper published in 2022) to run atop the Spark environment. It is written in **C++** for performance, focusing on:
- **Vectorized (columnar) execution** to exploit modern CPU features (e.g., SIMD).  
- Integration with the broader **Spark** ecosystem, but avoiding many overheads of the JVM.  
- Aggressive **adaptive execution** strategies.
Photon competes with other cloud data warehousing/lakehouse engines like **Snowflake**. While it leverages Spark’s scheduling and resource model (the “Databricks Runtime”—fork of Apache Spark), Photon replaces parts of Spark’s built-in operator execution with a **native** code path in C++.

---
## Lakehouse Architecture Recap
A typical **lakehouse** architecture has the following layers:
1. **Storage Layer (Data Lake)**  
   - Stores large amounts of data on distributed object stores (e.g., AWS S3, Azure Blob, GCS).  
   - Often uses columnar file formats like **Parquet**.  
   - Metadata or table formats (e.g., **Delta Lake**, **Iceberg**) track schema/partition info.
2. **Execution Layer**  
   - Multiple worker nodes run computations on the stored data.  
   - A **task scheduler** (Spark driver) launches tasks that read from the storage layer and run transformations/queries.
3. **Data Caching**  
   - Workers may cache frequently accessed data in local SSD (NVMe) to reduce repeated remote I/O.
**Visual Diagram:**
```
          +----------------------+
          |    Cloud Storage    |  <-- e.g., S3, GCS, Azure
          +--------+-----------+
                   |
         (Parquet  |   Delta/metadata)
                   |
        +--------------------------------+
        |       Compute/Execution        |
        |  Spark / Photon Query Engine   |
        +--------------------------------+
                   |
            [  Data Caching on NVMe  ]
```
- As technology improves, many workloads become **CPU-bound** rather than strictly I/O-bound. Photon addresses CPU inefficiencies by employing **C++** vectorization and advanced compiler optimizations.

---
## Motivation and Bottlenecks
- **Existing Spark constraints**:
  1. Spark is largely **JVM-based** (Scala/Java).  
  2. The JVM’s **JIT** compilation is powerful, but can be difficult to optimize deeply for specialized data processing.  
  3. **Garbage Collection** overhead leads Spark developers to store critical data “off-heap” anyway, partly defeating the convenience of Java memory management.  
  4. Large, “compiled” SQL operators in Spark can exceed the JVM’s method-size limits, causing fallback to an **interpreted** code path.  

- **Photon’s Approach**:
  1. Implement the query engine in **C++** for direct control over memory usage, CPU instructions (SIMD), etc.  
  2. Use **columnar** data paths end-to-end, no “row pivoting” unless absolutely necessary.  
  3. **Vectorized** query execution to handle data in batches for better cache utilization and simpler loop constructs.
	---
## High-Level Design

### Columnar Execution
- **Columns** are processed in contiguous memory blocks (batches) rather than row-by-row.  
- Benefits include:
  - Exploit CPU cache lines more effectively.  
  - Skip reading unneeded columns (projection pushdown).  
  - Use **SIMD** instructions to process many column values in parallel.
### Native C++
- Photon is written in **C++** (instead of JVM languages).  
- Allows precise control of:
  - Memory allocations (no GC overhead).  
  - Low-level CPU instructions (SIMD, SSE/AVX, etc.).  
  - Code-level optimizations via **compiler hints**, advanced debugging/profiling with standard C++ toolchains.

### Pull-Based Operator Pipeline
- Each operator in the query plan has a `getNext()` method, requesting a batch of rows from its child operator.  
- This is a **pull**-based design:  
  ```
  Operator (JOIN) => calls -> child1.getNext()  
                                 child2.getNext()
  ```
- Contrasts with “push-based” designs (like Snowflake) where data flows upwards from the scan node automatically.  
- Both designs are valid; Databricks chose pull-based for Photon.

### Interpreted Vectorized Execution
- Instead of generating custom code for every incoming SQL (as pure code-generation engines might), Photon has a **library** of C++ operator implementations.  
- Each operator runs in an **interpreted** style: one function call per operator batch.  
- Additional internal logic fuses or optimizes certain operator sequences (like `BETWEEN`).
---
## Code Generation vs. Interpreted Engines

### Code-Generation Approach
1. **SQL -> Plan -> Compile to Assembly**  
2. Single specialized code path can be very fast (no interpretive overhead).  
3. Downsides:  
   - Harder to debug individual operators (fusion merges them).  
   - Hard to adapt mid-query (assembly is fixed).  
   - Potentially large code blow-up for complex queries or sub-operators.

### Interpreted Vectorized Approach
1. **SQL -> Plan -> Operators** (each a compiled C++ class/function).  
2. Fewer “mega-methods”; each operator is an object that processes batches.  
3. Downsides:  
   - Overhead of repeated virtual function calls (`operator->getNext()` calls).  
   - Potentially slower if not heavily optimized.  
4. Upsides:  
   - **Easy to debug** and measure operator-level performance.  
   - **Adaptive** optimizations possible mid-query.

---

## Photon Implementation And Optimizations

### Operator Fusion
- Photon can create specialized operator variants for frequently combined patterns.  
- **Example**: `BETWEEN` is logically `>= AND <=`. Instead of two filters, a single fused operator checks both conditions.  
- Reduces overhead from multiple iteration loops and function calls.

### Position Vectors
- Photon stores an internal notion of which rows in a batch are still “active” (not filtered out).  
- Two main strategies to track active vs. filtered rows:
  1. **Bit Vector**  
     - 1-bit per row indicates active/inactive.  
     - Great if most rows remain active, but requires branching on each row.  
  2. **Position Vector**  
     - An array of indices pointing to active rows only.  
     - Great if many rows are filtered out (fewer rows to iterate over).  
- Photon uses **Position Vectors** primarily. If a batch has many inactive rows, the position vector is short. If nearly all rows are active, Photon employs specialized logic to skip overhead.

#### Example Visualization

```
 Original Column:  [ Jordan,  Sophie,   SarahJessica,  Parker ]
 Filter Condition:  attractiveness=10 => SarahJessica=0 
 Position Vector:  [0, 1, 3]
                   // Means indexes 0,1,3 in original array are active
```

### Compiler Hints in C++ (restrict and template)

In C++, **compiler hints** can generate more efficient assembly:

1. **`restrict`** keyword  
   - Asserts that pointers do not overlap.  
   - Enables more aggressive vectorization because the compiler knows data writes won’t alias reads.

2. **Templates** with compile-time booleans  
   - By parameterizing operators (e.g., `template<bool HAS_NULLS, bool ALL_ACTIVE>`), the compiler can statically eliminate branches for known-true or known-false conditions.

**Example**: A simplified Photon function for `sqrt()` operation on a batch:

```cpp
template<bool HAS_NULLS, bool ALL_ACTIVE>
void sqrtVector(
    const int32_t* restrict positionVector, 
    int size,
    const float* restrict input,
    float* restrict output,
    const uint8_t* restrict nullBitmask
) {
    for (int i = 0; i < size; ++i) {
        
        // If ALL_ACTIVE=false, we need to look up the row index:
        int rowIdx = ALL_ACTIVE ? i : positionVector[i];

        // If HAS_NULLS=true, check if current row is null:
        if constexpr (HAS_NULLS) {
            bool isNull = (nullBitmask[rowIdx >> 3] >> (rowIdx & 7)) & 1;
            if (isNull) {
                // skip or output null
                continue;
            }
        }

        float val = input[rowIdx];
        // With 'restrict', the compiler can safely assume no overlap
        // Possibly compile down to a simd sqrt operation
        output[rowIdx] = __builtin_sqrtf(val);  // or simd intrinsics
    }
}
```

- **`if constexpr`** in C++17 eliminates the branch entirely for a given template instantiation.  
- This code, although simplified, demonstrates how Photon avoids repeated branching for every row.

### Vectorized Hash Tables
- **Hash Joins** are central to big data queries. Photon tries to process multiple probe rows in parallel (SIMD).
- Key strategies:
  1. **Batch compression**: If 90% of the batch is null or filtered out, compress the batch so only active rows remain contiguous for better SIMD usage.  
  2. **Prefetch**: When checking a hash bucket, also load neighboring buckets to reduce cache misses during collisions.  
  3. **Multi-column**: If the join is on (colA, colB, …), Photon can simultaneously compare all columns with SIMD.

---

## Adaptive Execution
- Photon dynamically adjusts operator behavior based on observed data patterns:
  - If all values in a batch are **non-null**, it uses simpler code paths (skips null checks).  
  - If a batch is extremely sparse, compress it to a smaller contiguous array.  
  - If a “string column” is discovered to contain only ASCII characters, Photon uses a cheaper uppercase/lowercase transformation than a full UTF-8 approach.  
  - If the column looks like a “UUID in string form,” Photon can interpret it as a fixed-size binary, enabling vectorized comparisons.

**Contrast**: a purely code-generated engine must fix its strategy before execution begins.

---

## Memory Allocation and Management

### Buffer Pool
- Rather than calling `malloc`/`free` repeatedly (which can be expensive in a tight loop), Photon reuses pre-allocated memory buffers:
  - **Column batch** data structures are reused once processing finishes for that batch.  
  - This often hits hot CPU caches (L1/L2) for better performance.  

### Spilling
- Some data (like hash table for joins, partial aggregates) must outlive the batch. Photon notifies Spark of memory usage so Spark can decide to **spill** other operators’ data to disk if memory is tight.
- In typical “warehouse” systems, memory is allocated more deterministically, but in a “lakehouse” with less predictable data, dynamic “spill or not” decisions are crucial.

---

## Integration with Spark
- Despite Photon’s native code, some parts of a user’s query plan may still rely on Spark’s original operators (e.g., unimplemented corner-case functions, certain advanced library features).
- Photon:
  1. Processes as many operators as it can in its **columnar** pipeline.  
  2. If it encounters an unsupported operator, it **hands back** the data to Spark’s row-based plan.  
- **Data Format Conversions**:
  - If Spark is about to handle data, Photon must pivot from columnar to row format.  
  - Minimizing back-and-forth is key. Often, the plan is designed so Photon runs an entire stage, then yields data to Spark.  
- **Semantic Equivalence**:
  - C++ numeric and datetime libraries can differ from Java’s. Photon invests engineering effort to replicate Spark’s semantics (so results match user expectations).

---

## Conclusion
Photon is Databricks’ native C++ query engine, providing:

1. **High Performance**  
   - Vectorized (columnar) execution with heavy SIMD usage.  
   - Minimizing overheads from the JVM (Garbage Collection, large code-gen fallback, etc.).  

2. **Interpreted Vectorized Execution**  
   - Each operator is a C++-level class or function, with smaller compiled code.  
   - **Extensive compiler hints** and dynamic instrumentation (adaptive execution).

3. **Adaptive Capabilities**  
   - Real-time detection of data properties (null patterns, ASCII-only strings, etc.) to switch to optimized code paths.  
   - Memory reservation/spilling integrated with Spark’s memory manager.

4. **Integration with Spark**  
   - The “Databricks Runtime” can seamlessly call Photon or Spark’s Scala operators.  
   - Photon typically handles the performance-critical portion of the plan, returning data to Spark for corner cases.

**Key Takeaway**  
While pure code generation can theoretically yield peak performance for specific queries, Photon’s **interpreted vectorized** approach is more maintainable, debuggable, and adaptively optimized. Combining C++ power with Spark’s flexibility, Photon aims to accelerate enterprise analytics on the modern **lakehouse** stack.

---
```md
## Extended Discussion and Examples

This section expands on the previous notes by providing **additional examples**, **mini code snippets**, and **visualizations** to illustrate Photon’s concepts in more depth. This content is especially useful for those who want a more “hands-on” understanding of how Photon might be implemented or integrated within a large-scale, C++-based query engine.

---

### 1. Example: Photon’s Flow in a Simple Query

Let’s imagine a simplified Spark/Photon execution flow for the following SQL:

```sql
SELECT
  city,
  COUNT(*) AS visit_count
FROM user_logs
WHERE status = 'OK'
GROUP BY city
```

1. **Data Source**  
   - The data is stored in a Parquet file under `user_logs/` in an S3 bucket.  
   - Spark (Databricks Runtime) decides to let **Photon** handle the scan of Parquet since Photon provides a specialized Parquet reader.

2. **Photon Operators**  
   - **Scan Operator**: Reads `Parquet` columns (`city`, `status`) in columnar batches.  
   - **Filter Operator**: Checks if `status` == `'OK'`.  
     - If the data is stored as a dictionary-encoded column for `status`, Photon can quickly skip non-`OK` blocks.  
   - **Aggregate Operator (Group By)**: Groups rows by the `city` column, counting rows. This is often implemented with a **hash-aggregate** data structure.

3. **Adaptive Execution**  
   - Photon checks if `status` is mostly `'OK'`, or if it’s a small fraction. If a small fraction is `'OK'`, the filter might reduce the batch drastically, so Photon can switch to a **position vector** approach.  
   - If `city` is an ASCII-only string, Photon can store it in a more compact representation (potentially 1-byte per char) internally.

4. **Memory Reservation and Spilling**  
   - If the group-by cardinality is large, Photon requests more memory from Spark’s memory manager.  
   - Spark might spill older partial aggregates to disk, ensuring Photon can build the in-memory hash table for the new partial aggregates.

5. **Result**  
   - Once Photon finishes the aggregation, it returns the final columnar result or row-based result (depending on the downstream consumer).  
   - If Spark has further operators that Photon does **not** handle, the data is pivoted to row format and handed back to Spark.

**Mini ASCII Diagram**:

```
 Parquet(Columns: city, status)  -->  [Photon Scan] 
        --> [Photon Filter: status='OK'] 
        --> [Photon Hash Aggregate on city]
        --> [Result Batches or Rows]
        --> [Spark final ops if needed]
```

---

### 2. Detailed Code Snippets

Below are **toy** or **pseudo-code** examples illustrating some of the photon-like mechanics.

#### 2.1 Parquet Reader (Pseudocode)

```cpp
class ParquetReader {
public:
    ParquetReader(const std::string& filePath);
    // parse file metadata, column schema, etc.

    // Fetch a single column batch
    bool getNextColumnChunk(
        ColumnChunk* outChunk, 
        int columnId, 
        int batchSize
    ) {
        // 1. If no more data or at column boundary, return false.
        // 2. Read up to 'batchSize' values from parquet column stripes.
        // 3. Decode them into a columnar buffer (e.g., int32_t* or float*).
        // 4. Set outChunk->dataPtr to the newly filled buffer.
        // 5. Return true if we read data, false if we’re done.
    }
    
private:
    // internal state: file pointer, offsets, metadata, etc.
};

```

**Explanation**:  
- For each column, Photon retrieves “chunks” of data—often in compressed, columnar format (Parquet).  
- Decoding logic might produce **uncompressed, contiguous arrays** of values that Photon’s operators consume.

#### 2.2 Filter Operator with Position Vector

```cpp
template<bool HAS_NULLS, bool MATCH_STRING>
class FilterOp : public Operator {
public:
    // constructor omitted for brevity
    
    Batch getNext() override {
        Batch childBatch = childOp->getNext();
        if (childBatch.empty()) {
            return Batch(); // end of data
        }

        // Prepare an output position vector (filteredRows)
        filteredRows.clear();

        // Example: "status = 'OK'"
        // We'll assume column index for 'status' is known.

        for (int i = 0; i < childBatch.size; ++i) {
            int rowIdx = childBatch.isAllActive ? i : childBatch.posVector[i];

            // optional null check if HAS_NULLS
            if constexpr (HAS_NULLS) {
                if (childBatch.isNull(rowIdx, statusColId)) {
                    // skip
                    continue;
                }
            }

            if constexpr (MATCH_STRING) {
                // e.g., compare string in childBatch
                auto valPtr = childBatch.columns[statusColId]->getPtr(rowIdx);
                if (isEqual(valPtr, "OK")) {
                    filteredRows.push_back(rowIdx);
                }
            } else {
                // numeric compare or some other logic
            }
        }

        // create a new batch from childBatch
        // but with a smaller posVector if we filtered out many rows
        Batch outBatch = childBatch;
        outBatch.isAllActive = false;
        outBatch.posVector = filteredRows;
        outBatch.size = filteredRows.size(); 
        return outBatch;
    }

private:
    std::vector<int> filteredRows;
    int statusColId;
};
```

**Key Points**:
- We rely on `posVector` for active rows.  
- The template arguments (`HAS_NULLS`, `MATCH_STRING`) let the compiler remove unneeded checks.  
- If most rows pass the filter, `outBatch.posVector` still references nearly all child rows. If almost everything is filtered, `posVector` is short, enabling fast iteration for subsequent operators.

---

### 3. Visualization of SIMD Execution

**Simd** (Single Instruction Multiple Data) is crucial for Photon. Below is an ASCII depiction of how an operator might process multiple values in a single CPU instruction:

```
 Traditional Approach (scalar):
    for i in [0..batchSize):
       output[i] = sqrt(input[i])

 SIMD Approach (conceptual):
    // Each register can hold 4 floats at once (for example).
    load 4 floats: [in[i], in[i+1], in[i+2], in[i+3]]
    apply sqrt on all 4 in one CPU instruction
    store the results into output
```

- In modern x86_64, instructions like SSE/AVX can do 4 or 8 floats at once.  
- Photon tries to feed these intrinsics with contiguous, type-homogeneous data arrays.

---

### 4. Advanced Adaptive Optimization Example

Suppose Photon is reading a “city” column that is typically freeform text. Mid-query, Photon notices that **all** strings in a particular batch are ASCII only and have length <= 20. It can:

1. Convert the batch into a **fixed-length** array of 20 chars (with null terminators if shorter).  
2. Possibly apply SIMD-based `toupper()` or `tolower()` across these 20-char blocks.  
3. Revert to normal string handling if the next batch includes multi-byte UTF-8.

This technique **reduces branching** and data variability within a single batch.

---

### 5. Integration with Spark: A More Detailed Diagram

In production, Spark’s driver orchestrates tasks. Each task can launch Photon operators or fall back to Spark’s native engine. The flow might look like:

```
 Spark Driver (Task Scheduler)
     |
     v
 +-----------+   +--------------------------------+
 | Spark Exec|-> | PhotonEngine (C++ Operators)    |
 | (Scala)   |   |   - ParquetReaderOp            |
 |           |   |   - FilterOp                   |
 |           |   |   - HashAggOp                  |
 |           |   +--------------------------------+
 |           |
 | [some ops not in Photon?] -> row-based Spark ops
 |           |
 +-----------+
```

When an operator chain is all Photon-compatible, the data is kept in **columnar** format. If an unsupported operation arises (for example, certain obscure window functions not yet implemented in Photon), the columnar result is **pivoted** back to Spark’s row format.

---

### 6. Extended Conclusion

1. **Photon’s Core Idea**  
   - Replace large chunks of Spark’s row-based JVM code with a **C++** columnar engine.  
   - Provide general-purpose vectorized operators that can adapt to data distribution at runtime.

2. **Challenges Addressed**  
   - **Memory**: Bypassing the JVM GC, using specialized buffer pools, and dynamic memory reservations integrated with Spark.  
   - **Performance**: Minimizing interpretive overhead through operator fusion, position vectors, SIMD intrinsics, and compile-time templates.

3. **Future Directions**  
   - **Expanded Operator Coverage**: More complex window functions, exotic data types, etc.  
   - **More Fine-Grained Adaptation**: Deeper heuristics on data cardinality mid-query, dynamic re-planning.  
   - **GPU Acceleration?** Some queries could run well on GPUs if Photon can offload vectorizable steps.

---

## Final Takeaways

- Photon exemplifies how a **vectorized, interpreted** approach in C++ can rival or beat pure code-gen engines when combined with thorough optimization, extensive compiler hints, and adaptive execution.  
- It seamlessly integrates with Spark’s scheduling, memory management, and partial row-based operations, striking a balance between performance and practical maintainability.  
- As **data** and **queries** grow ever more varied (the Lakehouse scenario), **adaptive** strategies and careful memory usage become key to extracting top-level performance.
```
