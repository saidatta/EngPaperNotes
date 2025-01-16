https://www.youtube.com/watch?v=_eY_SBbsAdE
---
### **Overview**
**Presto** is a distributed SQL query engine extensively used at **Facebook** to support analytical and interactive workloads. Facebook's internal project, **RaptorX**, is designed to reduce query latency by up to **10X** through **hierarchical caching**. This approach tackles I/O bottlenecks introduced by **disaggregated storage architectures**, ensuring efficient query processing while maintaining scalability for petabyte-scale workloads.

#### **Key Objectives of RaptorX:**
- Achieve **10X faster query latency** using hierarchical cache.
- Mitigate the **I/O bottlenecks** caused by disaggregated storage, a common architecture in cloud-based data systems.
- Maintain the benefits of **scaling compute and storage independently**.

---
### **Presto at Facebook**
- **Scale**: Presto is deployed across **50,000 servers** at Facebook, processing **exabytes of data daily**.
- **Use Cases**: Presto powers various workloads, including:
  - **Interactive** queries from data scientists and engineers.
  - **Analytical** workloads for dashboards and reporting.

### **Challenges of Disaggregated Storage**

In disaggregated architectures, **storage and compute scale independently**, which offers flexibility in managing resources. However, the trade-off is **increased latency** due to frequent remote storage access.

#### **Challenges:**
1. **Remote I/O Bottlenecks**: Network and remote storage access add significant latency to queries.
2. **Multiple I/O Operations**: Queries must communicate with external storage, leading to delays.
3. **Increased Network Overhead**: Each query requires remote metadata and file list fetches, slowing down performance.

#### **Example of the Presto Architecture with Disaggregated Storage**:
1. **Query Planner**: The planner optimizes the query and fetches metadata from the **metastore** (remote I/O).
2. **Scheduler**: The scheduler assigns tasks to Presto workers based on the file list (remote I/O).
3. **Worker Nodes**: Workers read data from remote storage (e.g., **HDFS**, **S3**) and perform query processing.

This architecture highlights how **every I/O operation involves network latency**, making it inefficient for latency-sensitive workloads.

---

### **RaptorX: Hierarchical Caching for Latency Reduction**

Caching is a well-known solution to mitigate the challenges posed by remote I/O. **RaptorX's hierarchical cache** implements a multi-level caching strategy to reduce the frequency of remote calls and optimize query execution.

### **1. Metastore Cache**

**Problem**: Fetching partition and metadata information from the metastore introduces significant overhead because Presto must frequently communicate with the **Hive Metastore**.

**Solution**: Implement a **metastore cache** to cache partition information, table metadata, and other relevant details locally. To ensure data consistency, each partition in the metastore cache is **versioned**.

#### **Versioning**
- **Why**: The metastore is mutable, meaning the metadata can change frequently (e.g., partition updates).
- **How**: For each partition, the cache stores the **version** number. When a query requests data from the cache, the version number is checked against the current version in the metastore to ensure the data is up-to-date.

**Performance Impact**:
- **20% Latency Reduction**: Especially effective for workloads involving large partitioned tables.
- For cases with a **large number of partitions**, latency reductions of up to **60%** have been observed.

---

### **2. File List Cache**

**Problem**: The Presto **scheduler** must fetch the list of files from remote storage (e.g., HDFS or S3) for each partition before assigning work to workers. This is a costly operation, often taking **100ms or more per query**.

**Solution**: Implement a **file list cache** that stores the list of files for each partition locally. Since the files in Facebook's storage system are immutable, caching this information is highly effective.

**Performance Impact**:
- **100ms Reduction per Query**: By eliminating redundant file list fetches from remote storage.

---

### **3. Affinity Scheduling for Data Locality**

**Problem**: In traditional Presto configurations, load balancing is used to assign queries to worker nodes based on CPU availability. This results in **poor cache locality**, as different workers may process the same files across different queries, leading to redundant caching.

**Solution**: **Affinity Scheduling** assigns **file splits** to workers based on locality. The goal is to assign the same file split to the same worker across queries, allowing workers to **reuse cached data** efficiently.

#### **How Affinity Scheduling Works**:
1. **Primary Worker**: When a query for a file split arrives, it is scheduled on the same worker that previously processed the split, enabling cache reuse.
2. **Secondary Worker**: If the primary worker is overloaded, a **secondary worker** is selected based on locality.
3. **Fallback to Load Balancing**: If neither the primary nor secondary workers are available, the traditional load-balancing approach is used.

**Impact**:
- This approach improves the effectiveness of caching by ensuring **data locality**, reducing redundant I/O operations, and improving query performance.

---

### **4. File Descriptor & Footer Cache**

**Problem**: For every file read, Presto must open the file and read its **footer** (index) to determine the locations of data blocks. This operation is costly in terms of both **I/O** and **CPU** because footers must be fetched from remote storage and deserialized.

**Solution**: Cache the **file footer** and **file descriptors** in memory to avoid repeatedly opening files and fetching footers.

**Performance Impact**:
- **40% Reduction in CPU Usage**.
- **Significant Latency Decrease**: Cached footers are reused across queries, reducing remote storage access and eliminating deserialization overhead.

---

### **5. Data Cache Using Alluxio**

**Problem**: Presto must frequently access remote storage (e.g., HDFS) for data, which introduces substantial latency.

**Solution**: Implement **block-level data caching** using **Alluxio**. Presto workers cache **data blocks** on local **SSDs**.

#### **How Data Caching Works**:
- When a worker reads a data block from remote storage, it caches the **1MB data block** on its SSD.
- Future queries for the same data block are served from the local SSD instead of remote storage.
  
**Block Caching Example**:
```plaintext
- Query 1 requests data from 1.1MB to 5.6MB.
- Alluxio fetches and caches 1MB to 6MB from remote storage.
- Query 2 requests data from 4.3MB to 7.6MB.
- Cached blocks from 4MB to 6MB are reused.
- Only 6MB to 7MB is fetched from remote storage.
```

**Performance Impact**:
- **10X to 20X Reduction in Latency**: Dramatic performance improvements are seen for queries that can leverage the local SSD cache.

---
### **6. Fragmented Result Cache**
**Problem**: Queries with small variations often repeat the same expensive computations. For example, a time-based filter might shift slightly (e.g., `date BETWEEN '2023-01-01' AND '2023-01-02'` vs. `date BETWEEN '2023-01-01' AND '2023-01-03'`), but the underlying data remains the same. Here, 2023-01-01 data will be cached.
**Solution**: Implement **fragmented result caching**, where intermediate query results for static portions of the query plan are cached and reused for subsequent queries.
#### **Handling Dynamic Queries**:
For dynamic queries (e.g., `time > NOW() - INTERVAL '3 days'`), **partition stats pruning** is used to identify portions of the query plan that are static and cache only those portions.
#### **Performance Impact**:
- **45% Reduction in Latency**.
- **75% Reduction in CPU Usage**: Fragmented result caching significantly reduces the need for redundant computation.

---

### **Performance Results**

- **TPC-H Benchmark**: RaptorX showed more than **10X performance improvements** in both **CPU efficiency** and **latency** when tested on a **114-node cluster** with **1TB SSDs per node**.
  
- **Production Use**: These caching techniques are **not just theoretical** but have been deployed at scale within Facebook, delivering **consistent wins** across multiple workloads. The project continues to expand its use for **interactive queries**.

---

### **Future Work**

- **Improving Cache Affinity**: Current affinity scheduling uses a **simple hash of file paths**. However, the approach is being refined to handle cases where workers frequently go down, leading to **unstable node counts**.
  
- **Expanding Fragmented Result Caching**: Further optimizations in **fragment normalization** and **dynamic query pruning** are being explored.

- **Optimizing Cache Hit Rate**: Facebook is experimenting with more sophisticated **shadow caching** techniques to improve cache hit rates.

---

### **Conclusion**

RaptorX has transformed Presto at Facebook by implementing **hierarchical caching**. The caching framework includes metadata, file lists, file footers, data blocks, and query fragments. These optimizations have significantly improved query latency, particularly for **interactive and analytical workloads**. By reducing **I/O overhead** and enhancing **cache locality**, RaptorX enables **10X faster query performance** while maintaining the benefits of



