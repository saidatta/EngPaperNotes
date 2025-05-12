https://www.youtube.com/watch?v=bZOvAKGkzpQ
### 1. Vector Types with Different Data Formats

#### 1.1 Scalar / Integer Vectors
- **Integer vectors** are straightforward: typically contiguous arrays of integer values (e.g., `int32_t[]`).
- The notable part is how DuckDB **internally** supports multiple compression formats, but logically in the engine, these are either flat, constant, dictionary, or sequence vectors (discussed in previous notes).

#### 1.2 String Vectors
- DuckDB (inspired by Umbra) uses a **16-byte string entry** format, where:
  - **Inline short strings** (12 bytes or fewer).  
    - The first 12 bytes store the string contents.  
    - The remaining 4 bytes store the length and potentially other metadata.  
  - **Prefix for long strings** (over 12 bytes).  
    - First 4 bytes store a prefix (e.g., first 4 characters).  
    - A pointer references the full string in a separate buffer.  
- **Advantages**:
  1. **No extra indirection** for short strings → less pointer chasing.
  2. **Fast string comparisons** by comparing the first 8 bytes (length + initial characters). If they differ, we immediately know the strings differ.

> [!info] **Reference**:  
> The Umbra paper (Neumann et al.) details this compact string representation.

---

### 2. Nested (Complex) Data Types

Nested types are increasingly common (e.g., due to JSON and array-like operations). DuckDB supports **list** and **struct** types (and by extension **map** via a list-of-struct approach). Instead of storing nested data as BLOBs, DuckDB **fully vectorizes** these structures, enabling efficient and composable query execution.

#### 2.1 Structs
- A **struct** contains multiple fields, each field is itself a vector.
- **Same cardinality**: Each field vector in a struct must match the number of rows in the parent struct.
- **Null handling**: Each field can have its own null bitmap (validity mask). The entire struct can be null, or individual fields can be null.
- **Example**: A struct called `product_info(item STRING, price INT)` might be stored as:
  - Vector `[item]`: `["pants","t-shirt","shoes",...]`
  - Vector `[price]`: `[20, 15, 45, ...]`

```plaintext
Struct Vector: product_info
   ├─ "item"  -> ["pants","t-shirt","shoes"]  (Vector<string>)
   └─ "price" -> [20, 15, 45]                 (Vector<int>)
```

#### 2.2 Lists (Arrays)
- A **list** can have different lengths per row (some may be empty, some may have multiple elements).
- **Representation**:
  - A parent vector of type `LIST<T>` holds **offset-length pairs** pointing into the **child vector** of type `T`.
  - The child vector’s length can be *larger* than the parent vector’s length, since a single row in the parent might reference multiple elements in the child.
- **Example**: A list of integers `[ [1,2], [], [5,6,7] ]` might store:
  - Offsets & lengths in the parent vector, e.g.:
    - Row 0: Offset=0, Length=2  (covers child[0..1] = {1,2})
    - Row 1: Offset=2, Length=0  (covers child[2..1], i.e. empty)
    - Row 2: Offset=2, Length=3  (covers child[2..4] = {5,6,7})
  - Child vector holds the flattened data: `[1, 2, 5, 6, 7]`.

```plaintext
List Vector: list<int>
   ├─ Offsets: [0, 2, 2]
   ├─ Lengths: [2, 0, 3]
   └─ Child Vector (int): [1, 2, 5, 6, 7]
```

> [!important]  
> Because both **struct** and **list** types themselves can contain further structs/lists, DuckDB can represent arbitrarily nested data in a fully vectorized fashion.

---

### 3. Revisiting the Execution Models

#### 3.1 Volcano Model (Pull-Based)
- Traditional “volcano” approach: each operator has a `GetChunk()` method that **pulls** data from child operators.
- **Vector Volcano**:
  - Instead of pulling one row at a time, operators pull an entire **vector (or chunk)** at a time.
  - Example: A `HashJoin` operator might do:
    1. **Build** phase: repeatedly call `child_right.GetChunk()` to build a hash table until no more data.
    2. **Probe** phase: call `child_left.GetChunk()` and probe the hash table.

```plaintext
+--------------+
| RootOperator |  <-- call GetChunk() 
+--------------+
       |
       v
+--------------+
|   HashJoin   |  <-- internally calls: RightChild.GetChunk() for build,
+--------------+                       LeftChild.GetChunk() for probe
       |
       v
+--------------+
|   Scan(...)  |
+--------------+
```

> [!info] Volcano was historically single-threaded (one operator calls the next). Modern hardware with many cores demands more advanced parallel strategies.

#### 3.2 Adding Parallelism with Exchange Operators
- One approach to parallelize Volcano: **Exchange operators** inserted into the plan.
- The plan is physically divided into multiple partitions, and each partition runs in parallel.  
- **Drawbacks**:
  1. **Plan Explosion**: If you have 200 cores, you can end up with a plan that’s 200x bigger.
  2. **Load Imbalance**: Partitions are determined upfront; if data distribution is skewed, some threads do more work than others.
  3. **Materialization Overhead**: Communication between operators often requires writing out intermediate rows.

---

### 4. Morsel-Driven Parallelism

#### 4.1 Motivation
- Modern systems like Umbra propose **morsel-driven** parallelism:
  - Each operator **is aware of** parallelism (not oblivious like in exchange-based).
  - Data is broken down into “morsels” or chunks that can be scheduled dynamically across threads.
  - Avoids plan explosion because parallelism isn’t baked directly into the logical/physical plan nodes.

> [!quote]  
> *“Instead of parallelism-unaware operators, each operator can handle parallel input chunks, enabling adaptive partitioning and load balancing.”*

#### 4.2 Pipelines in Morsel-Driven Systems
- A query plan is decomposed into **pipelines** by identifying “pipeline breakers” (e.g., a `HashJoin` build phase that materializes a hash table).
- **Example**:

```plaintext
        Query Plan
          ┌──────┐
          │GroupBy│   <- Pipeline #2 (probe + aggregate)
          └──────┘
             ^
             |
          ┌───────┐
          │HashJoin│ <- Pipeline #2 continues
          └───────┘
             ^
             |
  ┌────────────────────┐
  │ Build side (Sales) │ <- Pipeline #1 (build hash table)
  └────────────────────┘
             ^
             |
  ┌─────────────────────┐
  │ Probe side (Orders) │ <- Pipeline #2 (probe)
  └─────────────────────┘
```

- **Pipeline #1**: Builds the hash table from `Sales` (fully parallel within itself).  
- **Pipeline #2**: Probes the hash table with `Orders`, then performs `GroupBy`.

> [!info]  
> In a push-based engine, we can schedule pipelines more flexibly (e.g., “finish building, then notify the next pipeline to start probing”), rather than weaving everything together in a single pull-based call stack.

---

### 5. Transition to Push-Based Execution

#### 5.1 Why Not Volcano?
- In a traditional Volcano model, everything is **entangled** via `GetChunk()` calls; splitting pipelines for parallelism is awkward.
- Parallel aware operators (like a parallel hash table build) need **control**: “Wait until build is complete, then switch to probe.”
- This control flow is more naturally expressed when the data is **pushed** from one stage to the next, rather than recursively pulled.

#### 5.2 DuckDB’s Shift to Push
- DuckDB originally used a **(Vector) Volcano** approach.
- Recognizing the limitation for **multicore** and **pipeline-based** scheduling, DuckDB **moved to a push-based** model. 
  - This model effectively integrates ideas from morsel-driven parallelism and pipeline-based scheduling.

> [!important]  
> *Modern parallel architectures need flexible scheduling where an operator can say: “I’m done building; next pipeline can start.” The push model orchestrates these transitions more directly.*

---

## 6. Code & Illustrations

### 6.1 Volcano-Style Pseudocode (for reference)

```cpp
// A simplified Volcano hash join
class HashJoinOperator {
public:
    Chunk GetChunk() {
        // if build phase not done, pull from right child
        while(!done_build) {
            auto chunk = right_child.GetChunk();
            if(chunk.empty()) {
                done_build = true;
                break;
            }
            hash_table.Build(chunk);
        }

        // now probe phase
        auto probe_chunk = left_child.GetChunk();
        if(probe_chunk.empty()) {
            return EmptyChunk();
        }
        // join matching rows with hash table
        return Join(probe_chunk, hash_table);
    }

private:
    bool done_build = false;
    HashTable hash_table;
    Operator right_child;
    Operator left_child;
};
```

### 6.2 Morsel-Driven / Push-Based Sketch

```cpp
// Sketch of push-based approach
// Operators register tasks that "push" data forward.

class ParallelHashJoinOperator {
public:
    void BuildPhase(TaskScheduler &scheduler) {
        // schedule build tasks in parallel
        for (auto &morsel : sales_morsels) {
            scheduler.AddTask([=] {
                auto chunk = ReadMorsel(morsel);
                hash_table.Build(chunk);
            });
        }
        scheduler.WaitAll();
        // once done, move to next pipeline
    }

    void ProbePhase(TaskScheduler &scheduler) {
        // schedule probe tasks in parallel
        for (auto &morsel : orders_morsels) {
            scheduler.AddTask([=] {
                auto chunk = ReadMorsel(morsel);
                auto joined = Join(chunk, hash_table);
                // push or buffer for next operator (e.g. aggregator)
            });
        }
        scheduler.WaitAll();
    }

private:
    HashTable hash_table;
    vector<Morsel> sales_morsels;
    vector<Morsel> orders_morsels;
};
```

---

## 7. Key Takeaways

1. **String Inlining**: DuckDB’s short-string optimization reduces pointer chasing and speeds up comparisons.  
2. **Nested Types**: By recursively storing structs and lists as vectors, DuckDB performs efficient vectorized operations on complex data (a big win for JSON, arrays, etc.).  
3. **Volcano vs. Push**:  
   - Volcano’s pull-based approach is simpler for single-threaded execution.  
   - Parallelism requires more dynamic scheduling → push-based with morsel-driven parallelism.  
4. **Performance Gains**: On modern hardware with large core counts, a push-based parallel engine can see *orders of magnitude* improvements over single-thread or naive Volcano models.  

> [!info]  
> For HPC scenarios (e.g., AWS machines with 96–200 cores), DuckDB’s parallel pipelines scale effectively, turning what could be multiple minutes of runtime into seconds.

---

## 8. References & Further Reading

- **Umbra Paper** (Neumann et al.) for inline string format:  
  [“Umbra: A Disk-Based System with In-Memory Performance” – CIDR 2020](http://cidrdb.org/cidr2020/papers/p18-neumann-cidr20.pdf)
- **Morsel-Driven Parallelism** (HyPer, Umbra, etc.):  
  - [“Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age,” CIDR 2013](https://www-db.in.tum.de/~babcock/bakker.pdf)  
- **DuckDB GitHub**: [https://github.com/duckdb/duckdb](https://github.com/duckdb/duckdb) – see `src/execution` for operator and pipeline code.  
- **Vectorized Execution** references:  
  - “Vectorwise: A Vectorized Analytical DBMS,” by Boncz et al.  
  - “MonetDB/X100: Hyper-Pipelining Query Execution,” by Boncz et al.

---

### Summary

DuckDB’s journey from a vectorized Volcano system to a **push-based, morsel-driven** parallel architecture highlights the **importance of parallelism** in modern database engines. By **vectorizing** not just scalar but also nested data types, DuckDB retains high performance across a wide range of workloads—including JSON-like operations and string manipulations. The **next** step is to delve into *how* the push-based approach concretely schedules tasks and handles pipeline transitions, but the concepts outlined above form the *foundation* of DuckDB’s query execution engine.

---

**End of Notes**