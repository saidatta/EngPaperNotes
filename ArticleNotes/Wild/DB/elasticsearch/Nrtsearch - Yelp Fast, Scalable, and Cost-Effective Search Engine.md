https://engineeringblog.yelp.com/2021/09/nrtsearch-yelps-fast-scalable-and-cost-effective-search-engine.html
https://github.com/Yelp/nrtsearch
#### **Introduction**
- **Context:** Search and ranking are critical features for Yelp, supporting a wide variety of use-cases, such as finding services (e.g., plumbers) and displaying relevant media (e.g., photos of dishes).
- **Problem:** Yelp's Elasticsearch-based platform faced scalability issues and rising costs as more use-cases were onboarded. Specific challenges included inefficient document-based replication, uneven shard distribution, and difficulties in autoscaling.
- **Solution:** Yelp developed **Nrtsearch**, a Lucene-based search engine, to address these challenges. Nrtsearch leverages near-real-time (NRT) segment replication and concurrent searching to improve performance and reduce costs.
---
#### **Why Replace Elasticsearch?**
- **Document-Based Replication:** 
  - Elasticsearch indexes documents individually across replicas, leading to higher CPU usage and the need for more replicas.
  - **Issue:** This method doesn't scale well as the system grows, leading to increased infrastructure costs.
- **Shard Distribution:**
  - Shard distribution controlled by Elasticsearch can result in hot/cold nodes, where some nodes are underutilized while others are overloaded.
  - **Issue:** This uneven load requires manual intervention to redistribute shards and avoid bottlenecks.
- **Autoscaling Challenges:**
  - Elasticsearch’s shard migration process complicates real-time autoscaling, requiring over-provisioning for peak loads.
  - **Issue:** Scaling up or down is restricted by shard and replica counts, making dynamic scaling difficult.
---
#### **Lucene-Based Features That Attracted Yelp**
- **Near-Real-Time (NRT) Segment Replication:**
  - Lucene writes indexed data into immutable segments, which replicas can pull from the primary node instead of re-indexing the data themselves.
  - **Benefit:** This approach reduces CPU load on replicas and speeds up indexing operations.
- **Concurrent Searching:**
  - Lucene allows parallel searching across multiple segments within an index, leveraging multi-core CPUs effectively.
  - **Benefit:** This enhances search performance without the need for distributing the load across multiple shards.
---
#### **Design Goals of Nrtsearch**
1. **Built on Lucene:**
   - Reuse existing custom Java code for features like ML-based ranking, analysis, and suggestions with minimal changes.
   - **Goal:** Achieve near-real-time segment replication and concurrent searching.
2. **Optimize for Search, Not Analytics:**
   - Focus on search-related tasks and reduce overhead on nodes handling search requests.
   - **Goal:** Maintain low latency and high throughput for search queries.
3. **External Storage for Indexes:**
   - Store the primary copy of the index in external storage (e.g., Amazon S3) instead of local storage.
   - **Goal:** Enable fast node startup without the need for rebalancing or backing up local storage.
4. **Fast Node Startup:**
   - Ensure new nodes can initialize quickly to handle increased load without rebalancing.
   - **Goal:** Support autoscaling with minimal downtime.
5. **Stable Extension API:**
   - Provide a consistent API for custom extensions (e.g., custom analysis, ML ranking) without requiring version-specific targeting.
   - **Goal:** Encourage modular development and extensibility.
6. **Point-In-Time Consistency:**
   - Ensure consistency for applications that require it.
   - **Goal:** Support transactional guarantees where needed.
---
#### **Implementation Details**
- **Base on Lucene Server Project:**
   - Used Mike McCandless’ open-source Lucene Server project as the foundation, upgrading it from Lucene 6.x to 8.x.
   - **Changes:** Replaced REST/JSON API with gRPC/protobuf for improved performance and implemented gRPC for NRT segment replication between primary and replicas.
- **Persistent Storage:**
   - Use persistent storage volumes (e.g., Amazon EBS) for primary nodes instead of local disks, allowing for fast recovery on node restarts.
   - **Architecture:**
     ```
     Primary Node 
       | 
    +--+--+
    |     |
  Disk   EBS (persistent storage)
  Backup to S3
     ```

- **NRT Segment Replication Workflow:**
   - **Step 1:** Primary indexes documents and stores segments on persistent storage.
   - **Step 2:** Periodically back up the index to S3.
   - **Step 3:** Replicas download the most recent backup from S3 and sync updates from the primary.
   - **Step 4:** Use gRPC for communication between primary and replicas.
   - **Architecture Diagram:**
     ```
     +-------------------+
     | Primary Node       |
     +-------------------+
             |
             | gRPC (segment sync)
             v
     +-------------------+
     | Replica Nodes      |
     +-------------------+
             |
             | gRPC
             v
        S3 Backup
     ```
- **Autoscaling and Kubernetes Integration:**
   - Deployed Nrtsearch on Kubernetes, using a Kubernetes operator to manage configurations, stateful sets, and service discovery.
   - **Autoscaling:** Horizontal Pod Autoscaler (HPA) scales replicas based on load. Replicas run on cheaper spot instances.
   - **Node Recovery:** Replicas restore from S3 backups and sync with the primary, minimizing downtime.
---
#### **Migration Strategy**
- **Phased Rollouts:**
   - Started with a low-traffic feature, gradually moving to larger use-cases.
   - **Dark Launch:** Sent traffic to both Elasticsearch and Nrtsearch, returning results from Elasticsearch while monitoring Nrtsearch stability.
- **Results Validation:**
   - Compared search results between Elasticsearch and Nrtsearch to ensure accuracy and consistency.
   - **Eval Tool:** Used to identify differences in search responses and diagnose scoring discrepancies.
- **Custom Plugin Migration:**
   - Migrated custom Elasticsearch plugins (e.g., for analysis and scoring) to Nrtsearch using Lucene’s extensibility.
---
#### **Performance Improvements**
- **Timing Improvements:**
   - Improved 50th, 95th, and 99th percentile query timings by 30-50%.
   - **Cost Reduction:** Reduced infrastructure costs by up to 40% through spot instance usage and autoscaling.
- **Visualization of Performance Gains:**
   ```
   +-----------------------------+
   | Percentile | ES Time | NRT Time |
   +-----------------------------+
   | P50        | 50ms    | 30ms     |
   | P95        | 200ms   | 120ms    |
   | P99        | 500ms   | 250ms    |
   +-----------------------------+
   ```
---
#### **Challenges and Learnings**
- **Caching Compiled Scripts:** Improved performance by reducing re-compilation overhead.
- **Virtual Sharding:** Distributed segments more evenly across search threads.
- **Parallel Fetch:** Enhanced performance for high-recall use-cases.
- **Replica Sync:** Synced replicas with the primary before full index startup to avoid slow queries during startup.
- **Geosharding Issues:**
   - Smaller geoshards for faster bootstrapping, but dense geoshards required optimization.
   - **Compression:** Switched from gzip to LZ4 for faster backup/restore.

---

#### **Future Work**
- **Open Source Contributions:** Continue improving Nrtsearch and open-sourcing features.
- **Additional Features:**
   - Support for search suggestions and highlights.
   - Integration of MLeap for ML model execution without performance degradation.
   - Scatter-gather functionality for non-geoshardable use-cases, enabling queries across multiple clusters.

---

### **Conclusion**
Nrtsearch provided Yelp with a fast, scalable, and cost-effective search solution, outperforming Elasticsearch in both performance and cost metrics. Through careful planning, phased rollouts, and iterative optimizations, Yelp successfully migrated its search use-cases to Nrtsearch, demonstrating its potential as a robust Lucene-based alternative for large-scale search applications.

--- 

These detailed notes encapsulate the technical aspects and learnings from Yelp’s journey with Nrtsearch, providing a comprehensive overview for advanced engineering audiences.