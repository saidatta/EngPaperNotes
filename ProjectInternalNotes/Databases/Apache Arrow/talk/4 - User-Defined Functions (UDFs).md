https://www.youtube.com/watch?v=bZOvAKGkzpQ
**UDFs**, **extensibility**, **out-of-core execution**, **resource management**, and other advanced topics. As before, we include:

DuckDB has growing support for user-defined functions in different languages (e.g., **Python**, **JavaScript**, and potentially **WebAssembly** UDFs). Extensions allow embedding custom functions written in C++ or other languages.
#### 1.1 Future of UDFs
- **Strong Demand**: Many users want to embed custom logic that is not easily expressed in pure SQL.  
- **Less Critical than in Traditional Systems?**  
  - Because DuckDB is embeddable with fast data transfer between client and engine, some “UDF-like” operations can be done by the host application.  
  - Still, native UDFs can bring performance benefits by avoiding round trips.

> [!info] **Example**:  
> - **Python UDF**: A Python function that can be registered and called from within DuckDB SQL.  
> - **JS/WASM UDF**: Possible for in-browser scenarios, letting you define custom logic directly in JavaScript.

---

### 2. Extensions & Extensibility

DuckDB embraces **extensions** for optional functionality. Rather than bloating the core engine, many features are implemented as loadable modules:

1. **File System Extensions**  
   - **HTTP/S3**: Remote data access.  
   - **Custom** file systems for specialized I/O.

2. **Data Format Extensions**  
   - **Parquet** (already a bundled extension).  
   - **JSON** (extension for JSON parsing/querying).  
   - **Geo** extension for geospatial types and functions.

3. **Catalog Extensions**  
   - You can write a **custom catalog** that masquerades as DuckDB tables.  
   - **Example**: A “SQLite catalog” so that a SQLite database can be attached as if it were a native DuckDB database.

4. **Language-Specific Extensions**  
   - E.g., JavaScript or Python-based user-defined functions.  

> [!tip] **Version Compatibility**  
> - C++ extension APIs **can break** as DuckDB evolves.  
> - A smaller set of **C-level APIs** is more stable but less powerful.

#### 2.1 Example: Building an Extension in C++
- DuckDB allows hooking into nearly **all** parts of the engine (parsing, planner, execution, file I/O, function registration).  
- Because the internal engine is still evolving quickly, extension authors must **track** DuckDB version changes.

```cpp
// Pseudocode: Minimal extension that registers a custom function
#include "duckdb.hpp"
using namespace duckdb;

extern "C" {

DUCKDB_EXTENSION_API void my_extension_init(duckdb::DatabaseInstance &db) {
   Connection con(db);
   // Register your function or custom logic here
   con.CreateVectorizedFunction("myfunc", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MyFunction);
}

DUCKDB_EXTENSION_API const char *my_extension_version() {
   return DuckDB::LibraryVersion();
}

}
```

Then you’d compile and load this shared library via `LOAD 'my_extension';`.

---

### 3. Out-of-Core Execution

DuckDB aims to maintain good performance even if working sets exceed available memory. Traditional engines might revert to “slow path” external sorts or merges. DuckDB’s approach tries to **gracefully degrade** performance without an abrupt cliff.

#### 3.1 Dynamic Partitioning & Partial Spill
- **Hybrid Hash Join**: If data grows too large, DuckDB dynamically partitions/spills subsets to disk, continuing to use hash join strategies.  
- **SSD Speed**: Modern SSDs are so fast that out-of-core operations can remain *surprisingly* efficient.

> [!important]  
> DuckDB’s approach avoids a separate “big data” code path. Instead, it adaptively spills or partitions as needed—still employing vectorized methods.

---

### 4. Resource Management

#### 4.1 Memory Management
- **Buffer Manager**:  
  - Maintains a fixed memory budget (configurable).  
  - Data blocks can be pinned in memory or “unpinned” and written back to disk when memory is tight.  
- **No Global LRU**:  
  - Inspired by **LeanStore**-like designs, avoiding a heavy centralized lock.  
  - Minimizes contention by letting each thread handle local queues or pinned blocks.

> [!info] **File Reuse vs. Compaction**  
> - Freed row groups can be **reused** in the single file, though the file size might not shrink (no automatic “shrink” compaction yet).  
> - Full compaction or vacuum-like operations may come in future updates.

#### 4.2 Thread Management
- DuckDB uses an **internal thread pool** for parallel query execution.  
- Alternatively, an application embedding DuckDB can supply its **own** thread pool if it wants to unify resource management across multiple services.

```plaintext
+----------------------------+
|      Application           |  <- (Manages resources at a higher level)
|   [Thread Pool, Memory]    |
+-------------^--------------+
              |
+-------------v--------------+
|         DuckDB Engine      |
|  [Push-based Pipelines,    |
|   Buffer Manager, etc.]    |
+----------------------------+
```

---

### 5. Extra Q&A Highlights

1. **Query Cancellation**  
   - Easy in push-based: after each vector, control returns to the scheduler, which can check a cancellation flag.

2. **DataFrame-like APIs**  
   - DuckDB (in Python, R, and C++ clients) provides a *dataframe-style* interface.  
   - Under the hood, it still builds logical plans; but it’s more convenient for row/column manipulation.

3. **Most Surprising Use Case**  
   - **WASM in the browser**: People run DuckDB **client-side**, enabling SQL queries on data loaded in-memory, all within the user’s web browser—*no server round-trip* needed.

4. **Next Steps**  
   - **Continuous internal rewrites**: Many subsystems (vector formats, push-based engine, etc.) have been revisited and optimized as DuckDB matures.  
   - **World Domination**: The team envisions DuckDB on every platform where analytics is needed—browsers, phones, serverless, etc.

---

### 6. Example: Pseudocode for a Memory-Spill Hash Join

Below is a *conceptual* snippet illustrating an adaptive hash join that partitions when memory is exceeded.

```cpp
void AdaptiveHashBuild(Chunk &input) {
    // Build in-memory
    for (auto &row : input) {
       if (!AddToHashTable(row)) {
          // If out-of-memory or threshold exceeded, spill
          SpillPartition(row);
       }
    }
}

bool AddToHashTable(Row &row) {
    if (hash_table_memory_usage < memory_limit) {
       hash_table.insert(row);
       return true;
    } else {
       return false;
    }
}

void SpillPartition(Row &row) {
    // Write row to temporary partition file
    // Potentially re-hash or partition by some key
}
```

- The actual code in DuckDB is more complex, but conceptually it tries to keep as much in memory as possible while gracefully spilling the rest.

---

### 7. References & Further Reading

- **UDF Documentation**:  
  - DuckDB’s [Extensions & UDFs on GitHub](https://github.com/duckdb/duckdb/tree/master/extension)  
  - [DuckDB Python UDF Guide](https://duckdb.org/docs/guides/python/udf.html) (in development).
- **Push-Based Execution**:  
  - [Morsel-Driven Parallelism Paper (HyPer)](https://www-db.in.tum.de/~leis/papers/morseldriven.pdf)
- **LeanStore Paper** (inspiration for concurrency-friendly buffer management):  
  - [“LeanStore: In-Memory Data Management beyond Main Memory,” SIGMOD 2018](https://dl.acm.org/doi/10.1145/3183713.3196890)
- **Geospatial Extension**:  
  - DuckDB’s official [GIS extension](https://duckdb.org/docs/extensions/gis.html).
- **Browser WASM**:  
  - [DuckDB WASM usage docs](https://duckdb.org/docs/api/wasm/).

---

## Summary

1. **UDFs & Extensions**: DuckDB is highly extensible—both for user-defined functions and for hooking into deeper internals (catalog, file system, custom data formats).  
2. **Out-of-Core & Resource Management**: Adaptive approaches allow queries to run even when data exceeds memory, with minimal performance degradation.  
3. **Push-Based Execution**: Still central, enabling easy cancellation, partial results, and advanced parallel scheduling.  
4. **Future Direction**: DuckDB continues evolving rapidly, aiming to serve analytics *everywhere*—including the browser, mobile, and any environment that benefits from embedded analytical SQL.