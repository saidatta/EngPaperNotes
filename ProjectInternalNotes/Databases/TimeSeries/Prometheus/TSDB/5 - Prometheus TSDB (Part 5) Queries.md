https://ganeshvernekar.com/blog/prometheus-tsdb-queries
---
### **Overview**
This detailed note explores the query mechanisms within **Prometheus TSDB**, focusing on how data is accessed from persistent blocks and the Head block. We delve into three primary query types, discussing matchers, query operations, and implementation details. This information is crucial for understanding how **PromQL** retrieves raw data for its operations.

---
### **Table of Contents**

1. **Introduction to TSDB Queries**
2. **Types of TSDB Queries**
   - A. **LabelNames()**
   - B. **LabelValues(name)**
   - C. **Select([]matcher)**
3. **Detailed Analysis of Query Mechanisms**
4. **Matchers: Understanding the Basics**
   - A. **Types of Matchers**
   - B. **Matcher Operations**
5. **Sample Selection Process**
6. **Querying Multiple Blocks**
7. **Querying the Head Block**
8. **Code References and Internal Flow**

---

### **1. Introduction to TSDB Queries**

- This section focuses on **low-level TSDB queries**, distinct from PromQL.
- PromQL uses these internal TSDB queries to gather raw data and perform computations on it.
- The discussion covers querying data from persistent blocks and briefly touches on the Head block.

```plaintext
+-----------------------------------------------+
|             Key Concepts of TSDB Queries      |
+-----------------------------------------------+
| Directly fetches raw data from TSDB           |
| Supports PromQL operations                   |
| Operates below the PromQL engine level       |
+-----------------------------------------------+
```

---

### **2. Types of TSDB Queries**

We focus on three primary query types executed on persistent blocks:

#### **A. LabelNames()**
- **Function**: Returns all unique label names in the block.
- Uses **Postings Offset Table** to retrieve label names.
- The result is sorted for easy display (e.g., query autocomplete suggestions).

#### **B. LabelValues(name)**
- **Function**: Returns all possible values for a given label name.
- Iterates over the positions of label values stored in the **Postings Offset Table**.
- Provides results lexicographically sorted by label values.

#### **C. Select([]matcher)**
- **Function**: Returns the samples for series that match the specified matchers.
- Involves fetching the relevant series and samples based on matcher conditions.

```plaintext
+----------------------------------------------+
|              Query Types in TSDB             |
+----------------------------------------------+
| LabelNames(): Unique label names in the block|
| LabelValues(name): Possible values for label |
| Select([]matcher): Samples for given matchers|
+----------------------------------------------+
```

---

### **3. Detailed Analysis of Query Mechanisms**

- **Creating a Querier**:
  - A **Querier** is initialized with the `min time (mint)` and `max time (maxt)` for the data to be queried.
  - This scope ensures queries are efficiently executed on a defined time range.

```plaintext
┌───────────────┐
│   Querier     │
├───────────────┤
│ minTime (mint)│
│ maxTime (maxt)│
└───────────────┘
```

---

### **4. Matchers: Understanding the Basics**

#### **A. Types of Matchers**

1. **Equal**: `labelName="<value>"` — Exact match of label name and value.
2. **Not Equal**: `labelName!="<value>"` — Excludes specific label name-value pairs.
3. **Regex Equal**: `labelName=~"<regex>"` — Matches values satisfying a regex.
4. **Regex Not Equal**: `labelName!~"<regex>"` — Excludes values matching a regex.

```plaintext
+---------------------------------------------------+
|              Matcher Types in TSDB                |
+---------------------------------------------------+
| Equal:         Exact match of label value         |
| Not Equal:     Excludes specific label value      |
| Regex Equal:   Matches label value with regex     |
| Regex Not Equal: Excludes label value with regex  |
+---------------------------------------------------+
```

#### **B. Matcher Operations**

- **Combining Matchers**: Multiple matchers are combined using logical AND operations.
- **Example Matcher Operations**:
  - `job=~"app.*", status="501"` results in an intersection of series matching both conditions.

---

### **5. Sample Selection Process**

- **Fetching Series IDs**:
  - Matchers first retrieve **series IDs (postings)** using the **Postings Offset Table** and **Postings i**.
  - An Equal matcher directly locates its position, while Regex matchers iterate over all possible values.

```plaintext
┌───────────────┬─────────────────────────┐
│ Label Matcher │ Series IDs (Postings)   │
├───────────────┼─────────────────────────┤
│ job="app1"    │ (s1)                    │
│ status="501"  │ (s2, s4)                │
└───────────────┴─────────────────────────┘
```

- **Intersecting Matchers**:
  - Combine results of individual matchers to form the final set of matching series.

```plaintext
(status="501") ∩ (job=~"app.*") = (s2)
```

- **Querying the Time Range**:
  - For each series, samples within the specified `mint` and `maxt` range are retrieved.

---

### **6. Querying Multiple Blocks**

- **Merging Results Across Blocks**:
  - When multiple blocks span the query time range, a **merge querier** combines results.
- Operations for each query type:
  - **LabelNames()**: Performs an N-way merge to gather label names from all blocks.
  - **LabelValues(name)**: Merges label values across blocks.
  - **Select([]matcher)**: Creates a combined iterator that lazily processes series across blocks.

```plaintext
+--------------------------------------------------+
|            Merging Queries Across Blocks         |
+--------------------------------------------------+
| LabelNames(): N-way merge of label names         |
| LabelValues(): Merged label values from blocks   |
| Select([]matcher): Lazy merge of sample iterators|
+--------------------------------------------------+
```

---

### **7. Querying the Head Block**

- The **Head block** stores all label-value pairs and postings lists in memory.
- The querying procedure remains the same as for persistent blocks, but data retrieval is faster due to its in-memory representation.

```plaintext
┌───────────────┬──────────────────────┐
│ Label Matcher │ Direct Memory Access │
├───────────────┼──────────────────────┤
│ job="app1"    │ Immediate results    │
└───────────────┴──────────────────────┘
```

---

### **8. Code References and Internal Flow**

- **Index Query Operations**: `tsdb/index/index.go`
  - Handles `LabelNames()` and `LabelValues(name)` queries.
  - Manages merging of postings lists.
- **Select Matcher Queries**: `tsdb/querier.go`
  - Processes `Select([]matcher)` queries on persistent blocks.
- **Head Block Queries**: `tsdb/head.go`
  - Performs all three query types directly on the Head block.
- **Merged Queries Across Blocks**: `tsdb/db.go`, `storage/merge.go`
  - Implements the logic for merging results from multiple blocks.

```plaintext
+-----------------------------------------------------+
|                Code Reference Summary               |
+-----------------------------------------------------+
| Index Queries:       tsdb/index/index.go            |
| Select Matchers:     tsdb/querier.go                |
| Head Block Queries:  tsdb/head.go                  |
| Merged Queries:      tsdb/db.go, storage/merge.go   |
+-----------------------------------------------------+
```

---

### **Conclusion**

Querying in Prometheus TSDB is a highly optimized process that uses **inverted indexes**, **postings lists**, and **lazy evaluation** to efficiently fetch and combine results across blocks. The detailed interaction between matchers, the Head block, and persistent blocks ensures robust and high-performance querying capabilities within the database. This deep understanding forms the foundation for optimizing and extending PromQL query capabilities.