## Abstract:
-   CockroachDB is a scalable SQL DBMS designed for global OLTP workloads with high availability and strong consistency.
-   Provides fault tolerance and resilience through replication and automatic recovery mechanisms.
-   Presents a novel transaction model for consistent geo-distributed transactions on commodity hardware.
-   Discusses how CockroachDB replicates and distributes data for fault tolerance and high performance.
-   Features a distributed SQL layer that scales automatically and provides a standard SQL interface.
-   Includes a comprehensive performance evaluation and case studies of CockroachDB users.
-   Concludes with lessons learned over five years of building CockroachDB.

##  1 Introduction

CockroachDB is designed to support modern transaction processing workloads that are increasingly geo-distributed. The paper discusses the challenges faced by global companies in building scalable applications while controlling where data resides for performance and regulatory reasons. The introduction presents a case study of a large company with a core user base in Europe and Australia and a fast-growing user base in the US. To power its global platform while reducing operational costs, the company decides to migrate to a cloud-based database management system.

The company has specific requirements to comply with the EU's General Data Protection Regulation (GDPR), to avoid high latencies due to cross-continental communication, to provide an 'always on' experience, and to support SQL with serializable transactions. The company chose CockroachDB (CRDB) to fulfill its requirements, and the paper presents the design and implementation of CRDB.

##### Figure 1 - Strategic Vision for CRDB Deployment
The figure shows the strategic vision for the deployment of CRDB for the global platform of the company discussed in the case study. The deployment includes multiple data centers across different geographic zones, with at least three replicas of every partition in the database. CRDB uses heuristics for data placement, but users can control the partitioning of data across nodes and the location of replicas for performance optimization or data domiciling strategy.

#### Key Features of CRDB
The paper explains how CRDB supports the requirements of global companies and presents its key features:

1.  Fault tolerance and high availability - CRDB maintains at least three replicas of every partition in the database across diverse geographic zones and uses automatic recovery mechanisms whenever a node fails.
    
2.  Geo-distributed partitioning and replica placement - CRDB is horizontally scalable, and users can control data partitioning and replica placement for performance optimization or data domiciling strategy.
    
3.  High-performance transactions - CRDB's transaction protocol supports performant geo-distributed transactions that can span multiple partitions and provides serializable isolation using no specialized hardware.
    
4.  SQL support - CRDB supports the SQL standard with a state-of-the-art query optimizer and distributed SQL execution engine.
    
5.  Production-grade - CRDB includes all the features necessary to run it in production as a system of record, including online schema changes, backup and restore, fast imports, JSON support, and integration with external analytics systems.
    

The paper concludes by stating that all the source code of CRDB is available on GitHub, and the core features of the database are under a Business Source License (BSL). Additionally, CRDB is 'cloud-neutral', meaning a single CRDB cluster can span an arbitrary number of different public and private clouds. These features enable users to mitigate the risks of vendor lock-in and cloud provider outages.

## 2 SYSTEM OVERVIEW

CockroachDB (CRDB) is a scalable SQL DBMS that supports global online transaction processing workloads while maintaining high availability and strong consistency. This section provides an overview of CRDB's architecture and how the system replicates and distributes data to provide fault tolerance, high availability, and geo-distributed partitioning.

### 2.1 Architecture of CockroachDB

CRDB uses a standard shared-nothing architecture, in which all nodes are used for both data storage and computation. Clients can connect to any node in the cluster. Within a single node, CRDB has a layered architecture consisting of the SQL layer, Transactional KV layer, Distribution layer, Replication layer, and Storage layer.

-   CockroachDB uses a standard shared-nothing architecture in which all nodes are used for both data storage and computation.
-   A CRDB cluster can consist of any number of nodes that may be located in the same data center or distributed globally.
-   Within a single node, CRDB has a layered architecture, including the SQL layer, Transactional KV layer, Distribution layer, Replication layer, and Storage layer.
-   The SQL layer is the interface for all user interactions with the database, including parser, optimizer, and SQL execution engine.
-   The Transactional KV layer ensures atomicity of changes spanning multiple KV pairs and is responsible for CRDB's isolation guarantees.
-   The Distribution layer presents the abstraction of a monolithic logical key space ordered by key and uses range-partitioning on the keys to divide the data into contiguous ordered chunks of size ~64 MiB, called Ranges.
-   Ranges start empty, grow, split when they get too large, and merge when they get too small. Ranges also split based on load to reduce hotspots and imbalances in CPU usage.
-   By default, each Range is replicated three ways with each replica stored on a different node, and the Replication layer ensures durability of modifications using consensus-based replication.
-   The Storage layer represents a local disk-backed KV store that provides efficient writes and range scans to enable performant SQL execution.

-   SQL layer: It is the highest-level layer and the interface for all user interactions with the database. It includes the parser, optimizer, and the SQL execution engine that convert high-level SQL statements to low-level read and write requests to the underlying key-value (KV) store.
-   Transactional KV layer: It ensures atomicity of changes spanning multiple KV pairs and is responsible for CRDB's isolation guarantees.
-   Distribution layer: This layer presents the abstraction of a monolithic logical key space ordered by key. CRDB uses range-partitioning on the keys to divide the data into contiguous ordered chunks of size ~64 MiB, that are stored across the cluster. This layer identifies which ranges should handle which subset of each query and routes the subsets accordingly.
-   Replication layer: By default, each range is replicated three ways, with each replica stored on a different node. This layer ensures durability of modifications using consensus-based replication.
-   Storage layer: This is the bottommost level that represents a local disk-backed KV store. It provides efficient writes and range scans to enable performant SQL execution.

#### 2.1.1 SQL Layer

The SQL layer is the interface for all user interactions with the database. It includes the parser, optimizer, and the SQL execution engine that convert high-level SQL statements to low-level read and write requests to the underlying key-value (KV) store. The SQL layer is not aware of how data is partitioned or distributed because the layers below present the abstraction of a single, monolithic KV store.

#### 2.1.2 Transactional KV Layer

The Transactional KV layer ensures atomicity of changes spanning multiple KV pairs and is largely responsible for CRDB's isolation guarantees.

#### 2.1.3 Distribution Layer

The Distribution layer presents the abstraction of a monolithic logical key space ordered by key. CRDB uses range-partitioning on the keys to divide the data into contiguous ordered chunks of size ~64 MiB, that are stored across the cluster. This layer identifies which ranges should handle which subset of each query and routes the subsets accordingly.

#### 2.1.4 Replication Layer

By default, each range is replicated three ways, with each replica stored on a different node. The Replication layer ensures durability of modifications using consensus-based replication.

#### 2.1.5 Storage Layer

The Storage layer is the bottommost level that represents a local disk-backed KV store. It provides efficient writes and range scans to enable performant SQL execution.

## Section 2.2: Fault Tolerance and High Availability

-   CRDB guarantees fault tolerance and high availability through replication of data, automatic recovery mechanisms, and strategic data placement.
-   Replication is achieved using the Raft consensus algorithm.
-   The unit of replication in CRDB is a command, which represents a sequence of low-level edits to be made to the storage engine.
-   Raft maintains a consistent, ordered log of updates across a Range’s replicas, and each replica individually applies commands to the storage engine as Raft declares them to be committed to the Range’s log.
-   CRDB uses Range-level leases, where a single replica in the Raft group (usually the Raft group leader) acts as the leaseholder.
-   CRDB automatically redistributes load across live nodes in case of node failures or membership changes.
-   CRDB creates new replicas of under-replicated Ranges and determines their placement based on node liveness data and cluster metrics.
-   CRDB has both manual and automatic mechanisms to control replica placement.
-   Automatic replica placement spreads replicas across failure domains and uses various heuristics to balance load and disk utilization.

## Section 2.3: Data Placement Policies

-   CRDB's replica and leaseholder placement mechanisms allow for a wide range of possible data placement policies.
-   Geo-Partitioned Replicas: tables can be partitioned by access location with each partition pinned to a specific region.
-   Geo-Partitioned Leaseholders: leaseholders for partitions in a geo-partitioned table can be pinned to the region of access with the remaining replicas pinned to the remaining regions.
-   Duplicated Indexes: indexes can be duplicated on a table and pinned to specific regions to enable fast local reads while retaining the ability to survive regional failures.

## 3 Transactions

-   CRDB transactions provide ACID guarantees and can span the entire key space, touching data resident across a distributed cluster.
-   CRDB uses a variation of multi-version concurrency control (MVCC) to provide serializable isolation.
-   Section 3.1 provides an overview of the transaction model.
-   Section 3.2 describes how CRDB guarantees transactional atomicity.
-   Sections 3.3 and 3.4 describe the concurrency control mechanisms that guarantee serializable isolation.
-   Section 3.5 gives an overview of how follower replicas can serve consistent historical reads.
-   A SQL transaction starts at the gateway node for the SQL connection, which acts as the transaction coordinator.
-   The coordinator employs two important optimizations: Write Pipelining and Parallel Commits.
-   Write Pipelining allows returning a result without waiting for the replication of the current operation.
-   Parallel Commits avoids the extra round of consensus by employing a staging transaction status.
-   The leaseholder provides mutual exclusion between concurrent, overlapping requests and verifies that the operations op depends on have succeeded.
-   If it is performing a write, it also ensures that the timestamp of op is after any conflicting readers.
-   After consensus is reached, each replica applies the command to its local storage engine.
-   The evaluation phase (Line 7) is the period of time when a transaction may encounter uncommitted writes from other transactions or writes so close in time to the transaction’s read timestamp that it is not possible to determine the correct order of transactions.
-   CRDB guarantees both atomicity and serializable isolation.

##### Execution at the transaction coordinator

-   Write Pipelining allows returning a result without waiting for the replication of the current operation, and Parallel Commits lets the commit operation and the write pipeline replicate in parallel.
-   The coordinator tracks operations which may not have fully replicated yet and maintains the transaction timestamp, which selects the point at which the transaction performs its reads and writes.
-   Before committing, the Parallel Commits protocol employs a staging transaction status which makes the true status of the transaction conditional on whether all of its writes have been replicated.
-   The coordinator is free to initiate the replication of the staging status in parallel with the verification of the outstanding writes, which are also being replicated.

##### Execution at the leaseholder

-   When the leaseholder receives an operation from the coordinator, it first checks that its own lease is still valid.
-   Then it acquires latches on the keys of the operation and all the operations the operation depends on, thus providing mutual exclusion between concurrent, overlapping requests.
-   Next, it verifies that the operations the operation depends on have succeeded.
-   If it is performing a write, it also ensures that the timestamp of the operation is after any conflicting readers, incrementing it if necessary.
-   After consensus is reached, each replica applies the command to its local storage engine.
-   Finally, the leaseholder releases its latches and responds to the coordinator if it hasn’t already done so.

##### Scenarios during evaluation phase

-   This is the period of time when a transaction may encounter uncommitted writes from other transactions or writes so close in time to the transaction’s read timestamp that it is not possible to determine the correct order of transactions.
-   The next sections discuss these scenarios, and how CRDB guarantees both atomicity and serializable isolation.

### 3.1 Overview
- The SQL transaction starts at the gateway node and is received interactively by the coordinator, which is responsible for orchestrating and ultimately committing or aborting the associated transaction. The coordinator algorithm employs two important optimizations: Write Pipelining and Parallel Commits. Write Pipelining allows returning a result without waiting for the replication of the current operation, and Parallel Commits lets the commit operation and the write pipeline replicate in parallel.

- When the coordinator receives a requested KV operation from the SQL layer, it initializes the transaction timestamp and tracks operations that may not have fully replicated yet. The coordinator sends the operation to the leaseholder for execution and waits for a response. If the response contains an incremented timestamp, the coordinator verifies whether repeating the previous reads in the transaction at the new timestamp will return the same value. If not, the transaction fails, and may have to be retried.

- When the leaseholder receives an operation from the coordinator, it checks whether its own lease is still valid, acquires latches on the keys of the operation, and verifies that the operations the operation depends on have succeeded. If the leaseholder is performing a write, it also ensures that the timestamp of the operation is after any conflicting readers. After the leaseholder evaluates the operation and determines what data modifications are needed in the storage engine, the leaseholder replicates the write operations. After consensus is reached, each replica applies the command to its local storage engine, and the leaseholder releases its latches and responds to the coordinator if it hasn’t already done so.

### 3.2 Atomicity Guarantees

-   CRDB considers all writes provisional until commit time, using write intents to hold provisional values.
-   An intent is an MVCC KV pair preceded by metadata pointing to a transaction record, which stores the current disposition of the transaction.
-   The transaction record serves to atomically change the visibility of all intents at once, and is durably stored in the same Range as the first write of the transaction.
-   When encountering an intent, a reader reads the intent’s transaction record and considers the intent as a regular value if the transaction is committed, ignores it if the transaction is aborted, and blocks if the transaction is still pending.
-   If the coordinator node fails, contending transactions mark the transaction record as aborted.
-   In the staging state, the reader attempts to abort the transaction if it hasn't been replicated, and otherwise assumes it is committed.

### 3.3 Concurrency Control

-   CRDB is an MVCC system, and each transaction performs its reads and writes at its commit timestamp, resulting in a total ordering of all transactions.
-   Write-read conflicts occur when a read runs into an uncommitted intent with a lower timestamp. The read waits for the earlier transaction to finalize.
-   Read-write conflicts occur when a write to a key at timestamp ta cannot be performed if there’s already been a read on the same key at a higher timestamp tb >= ta. The writing transaction advances its commit timestamp past tb.
-   Write-write conflicts occur when a write runs into an uncommitted intent with a lower timestamp or a committed value at a higher timestamp. The writing transaction waits for the earlier transaction to finalize or advances its timestamp past it.
-   CRDB employs a distributed deadlock-detection algorithm to abort one transaction from a cycle of waiters.

### 3.4 Read Refreshes
- Section discusses the read refresh mechanism used by CockroachDB to maintain serializability while advancing the commit timestamp of a transaction. The read refresh mechanism is necessary when conflicts arise between transactions, which require the commit timestamp of a transaction to be advanced. To maintain serializability, the read timestamp must also be advanced to match the commit timestamp.
- CockroachDB maintains a set of keys in the transaction's read set, and a read refresh request is used to validate whether the keys have been updated in a given timestamp interval. If the data has changed, the transaction needs to be restarted. If no results from the transaction have been delivered to the client, CockroachDB retries the transaction internally. If results have been delivered, the client is informed to discard them and restart the transaction.
- The read refresh mechanism involves re-scanning the read set and checking whether any MVCC values fall in the given interval. This process is equivalent to detecting the rw-antidependencies that PostgreSQL tracks for its implementation of SSI. CockroachDB may allow false positives to avoid the overhead of maintaining a full dependency graph, similar to PostgreSQL.
- Advancing the transaction's read timestamp is also required when a scan encounters an uncertain value, which is a value whose timestamp makes it unclear if it falls in the reader's past or future. In this case, CockroachDB attempts to perform a refresh, and if successful, the value will now be returned by the read.

### 3.5 Follower Reads

-   CRDB allows non-leaseholder replicas to serve read-only queries at specific timestamps using 'AS OF SYSTEM TIME'
-   Non-leaseholder replicas must ensure no future writes can retroactively invalidate the read and that they have all necessary data
-   Leaseholders track incoming request timestamps and periodically emit a closed timestamp, below which no further writes are accepted
-   Closed timestamps and Raft log indexes are exchanged periodically between replicas
-   Nodes keep track of latency with other nodes, forwarding read requests to the closest node with the required data

## 4 Clock Synchronization

-   CRDB uses software-level clock synchronization services like NTP or Amazon Time Sync Service
-   Hybrid-logical clocks (HLCs) are used for timestamp ordering, providing single-key linearizability and handling synchronization bounds violations

### 4.1 Hybrid-Logical Clocks

-   Each node maintains an HLC, combining physical time (system clock) and logical time (Lamport's clocks)
-   HLCs are configured with a maximum allowable offset between physical time components
-   Important HLC properties:
    1.  Causality tracking: Nodes attach HLC timestamps to messages, updating their local clock with received timestamps
    2.  Strict monotonicity: HLCs provide monotonicity within and across restarts, ensuring causally dependent transactions are timestamped correctly
    3.  Self-stabilization: Nodes forward HLCs upon message receipt, allowing HLCs across nodes to converge and stabilize despite individual clock divergences

### 4.2 Uncertainty Intervals

-   CRDB offers single-key linearizability for reads and writes under normal conditions
-   Transactions are given a provisional commit timestamp and an uncertainty interval
-   If a transaction encounters a value within its uncertainty interval, it performs an uncertainty restart
-   Uncertainty intervals allow transactions to maintain real-time ordering without a perfectly synchronized global clock

### 4.3 Behavior under Clock Skew

-   Consistency is maintained through Raft, which does not have a clock dependency
-   Range leases cause complications when clock skew leads to multiple nodes believing they hold a valid lease
-   CRDB employs two safeguards to maintain transaction isolation:
    1.  Range leases contain start and end timestamps, preventing leaseholders from serving reads/writes outside their lease interval
    2.  Writes to a Range's Raft log include the lease sequence number, rejecting writes if the sequence number doesn't match the active lease
-   These safeguards ensure serializable isolation even under severe clock skew
-   Clock skew outside configured bounds can result in single-key linearizability violations for causally-dependent transactions
-   Nodes periodically measure clock offset and self-terminate if exceeding maximum offset by more than 80% compared to the majority of nodes

## 5 SQL

-   User interaction with the database passes through the SQL layer
-   CRDB supports much of the PostgreSQL dialect of ANSI standard SQL with some extensions

### 5.1 SQL Data Model

-   Every SQL table and index is stored in one or more Ranges
-   All user data is stored in ordered indexes, with one designated as the "primary" index
-   CRDB supports hash indexes to avoid hot spots and distribute load across multiple Ranges

### 5.2 Query Optimizer

-   Uses a Cascades-style query optimizer with over 200 transformation rules
-   Transformation rules written in a domain-specific language (DSL) called Optgen, which compiles to Go
-   Optimizer is distribution-aware, taking data distribution and partitioning into account
-   Minimizes cross-region data shuffling by assigning a cost to each index replica based on proximity to the gateway node

Figure 3: Physical plan for a distributed hash join

-   Illustrates the physical plan for executing a distributed hash join in CRDB
-   Demonstrates how the optimizer takes data distribution and partitioning into account to minimize cross-region data movement

### 5.3 Query Planning and Execution

-   SQL query execution in CRDB operates in gateway-only mode or distributed mode
-   Read-only queries can execute in distributed mode
-   SQL layer can perform read and write operations for any Range on any node
-   Decision to distribute is made by a heuristic estimating the quantity of data that would need to be sent over the network
-   Physical planning stage transforms the query optimizer's plan into a directed acyclic graph (DAG) of physical SQL operators
-   Execution engines: row-at-a-time engine (Volcano iterator model) and vectorized execution engine (inspired by MonetDB/X100)

### 5.4 Schema Changes

-   CRDB performs schema changes using a protocol that allows tables to remain online during the schema change
-   Schema change protocol decomposes each schema change into a sequence of incremental changes (similar to F1)
-   Addition of a secondary index requires two intermediate schema versions to ensure index updates on writes across the entire cluster before becoming available for reads
-   Database remains in a consistent state throughout the schema change by enforcing the invariant of having at most two successive schema versions in the cluster at all times


## 6 Evaluation

-   Evaluates CRDB performance, scalability, and multi-region deployment under disaster scenarios
-   Uses CRDB v19.2.2 in all experiments, unless noted otherwise

### 6.1 Scalability of CockroachDB

-   Vertical and horizontal scalability evaluated using Sysbench OLTP suite
-   Throughput per vCPU remains nearly constant as the number of vCPUs increases
-   Vertical scalability demonstrated with varying AWS instance types
-   Horizontal scalability demonstrated with varying cluster sizes

#### 6.1.2 Scalability with cross-node coordination

-   CRDB evaluated using TPC-C with variable percentage of remote warehouses and replication factors
-   Overhead of replication can reduce throughput by up to 48% (three replicas) or 57% (five replicas)
-   Distributed transactions may reduce throughput by up to 46%
-   Workloads scale linearly with increasing cluster sizes

Table 1: TPC-C Benchmark Environment and Results

-   CRDB scales to support up to 100,000 warehouses, corresponding to 50 billion rows and 8 TB of data, at near-maximum efficiency
-   Amazon Aurora achieves only 7.3% efficiency with 10,000 warehouses

### 6.2 Multi-region Availability and Performance

-   TPC-C 1,000 performance measured against multi-region CRDB cluster with AZ and region failures
-   All policies tolerate AZ failures; only geo-partitioned leaseholders tolerate region-wide failures
-   Geo-partitioned leaseholders have higher sustained throughput during region-wide failures but come at a cost of higher p90 latencies during stable operation and recovery
-   Duplicated indexes policy maintains the lowest p90 latencies under stable conditions

### 6.3 Comparison with Spanner

-   CRDB performance compared against Cloud Spanner using the YCSB benchmark suite
-   CRDB configurations: 4, 8, and 16 vCPUs per node
-   CRDB shows significantly higher throughput for most YCSB workloads
-   Both systems demonstrate horizontal scalability
-   CRDB does not scale well on Workload A (update-heavy, zipfian distribution of keys) due to its high contention profile
-   CRDB has significantly lower latencies at all percentiles, partially attributed to Spanner's commit-wait

### 6.4 Usage Case Studies

1.  Virtual customer support agent for a telecom provider:
    
    -   Objective: Reduce customer service costs with a virtual agent providing 24/7 support
    -   Chose CRDB for strong consistency, regional failure tolerance, and performance for geo-distributed clusters
    -   Multi-region CRDB cluster deployed across on-prem data center and AWS regions
    -   Opted for geo-partitioned leaseholders policy for regional failure survivability
2.  Global platform for an online gaming company:
    
    -   Processes 30-40 million financial transactions per day
    -   Requirements: Data compliance, consistency, performance, and service availability
    -   CRDB architecture fit their requirements and is now a strategic component in their long-term roadmap
    -   CRDB deployment designed to isolate failure domains and pin user data to specific localities for compliance and low latencies

## 7 LESSONS LEARNED

### 7.1 Raft Made Live

-   CRDB chose Raft as the consensus algorithm for its ease of use and precise implementation, but found challenges in using it in a complex system like CRDB.
-   Challenges included reducing communication overhead and joint consensus for rebalancing operations.
-   Joint consensus allows for an intermediate configuration and requires the quorum of both old and new majority for writes, resulting in unavailability only if either majority fails.

### 7.2 Removal of Snapshot Isolation

-   CRDB originally offered two isolation levels, SNAPSHOT and SERIALIZABLE, but made SERIALIZABLE the default due to performance advantages and avoiding write skew anomalies.
-   Removing the check for write skews for SNAPSHOT proved to be difficult and would require pessimistic locking for any row updates, even for SERIALIZABLE transactions.
-   CRDB opted to keep SNAPSHOT as an alias to SERIALIZABLE instead.

### 7.3 Postgres Compatibility

-   CRDB adopted PostgreSQL’s SQL dialect and network protocol to capitalize on the ecosystem of client drivers, but behaves differently from PostgreSQL in ways that require client-side intervention.
-   Clients must perform transaction retries after an MVCC conflict and configure result paging.
-   CRDB is considering the gradual introduction of CRDB-specific client drivers.

### 7.4 Pitfalls of Version Upgrades

-   A clear upgrade path between versions with near-zero downtime is crucial for a system that values operational simplicity.
-   Running a mixed-version cluster can introduce additional complexity and serious bugs.
-   CRDB addressed this by moving the evaluation stage first and proposing the effect of an evaluated request, rather than the request itself.

### 7.5 Follow the Workload

-   Follow the Workload is a mechanism built into CRDB to automatically move leaseholders closer to users accessing the data for workloads with shifting access localities.
-   However, this mechanism is rarely used in practice, and operators prefer manual controls over replica placement for expected workloads.
-   Adaptive techniques in databases are difficult to get right for a general-purpose system and can be unpredictable, hindering adoption.
##  8 Related work
-   Distributed transaction models:
    -   Many systems with reduced consistency levels have been proposed to overcome the scalability challenges of traditional relational databases.
    -   Spanner provides the strongest isolation level, strict serializability, but its protocol is significantly different from CRDB's.
    -   Calvin, FaunaDB, and SLOG provide strict serializability but do not support conversational SQL.
    -   H-Store and VoltDB support serializable isolation but perform poorly on cross-partition transactions.
    -   L-Store and G-Store commit all transactions locally but require relocating data on-the-fly.
    -   Recent work has explored minimizing the commit time of geo-distributed transactions.
-   Distributed data placement:
    -   Several papers have considered how to place data in a geo-distributed cluster, with different objectives.
    -   CRDB gives users control by supporting different data placement policies.
    -   CRDB range-partitions based on the original keys, resulting in better locality for range scans than some other systems.
-   Commercial Distributed OLTP DBMSs:
    -   There are many distributed DBMS offerings on the market today for OLTP workloads, each providing different features and consistency guarantees.
    -   Amazon Aurora is a distributed SQL DBMS which can only be deployed on AWS.
    -   F1 is a federated SQL query processing platform from Google, which inspired CRDB's distributed execution engine and online schema change infrastructure.
    -   TiDB is an open-source distributed SQL DBMS that is compatible with the MySQL wire protocol.
    -   NuoDB is a proprietary NewSQL database that scales storage independently from the transaction and caching layer.
    -   FoundationDB is an open-source key-value store from Apple that supports strictly serializable isolation.

### 9 Conclusion
The conclusion and future outlook of the paper focuses on the advantages of CockroachDB, including its novel transaction protocol that achieves serializable isolation, consensus-based replication for fault tolerance and high availability, and geo-partitioning and follow-the-workload features that minimize latency. The authors note that CockroachDB is already providing value to thousands of organizations, but they plan to improve the software with each release, including a completely redesigned storage layer, geo-aware query optimizations, and improvements to other parts of the system. They also plan to improve support for operational automation, making databases truly "serverless" from a user's perspective. Disaggregated storage, on-demand scaling, and usage-based pricing are among the areas they plan to develop in the future. Finally, the authors point out that making a geo-distributed database perform well in such an environment is a problem ripe for independent research, and they look forward to supporting and participating in it.
