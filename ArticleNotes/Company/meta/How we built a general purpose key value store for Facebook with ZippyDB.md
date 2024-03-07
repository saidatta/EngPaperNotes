https://engineering.fb.com/2021/08/06/core-infra/zippydb/
## Introduction to ZippyDB
- **ZippyDB Overview**: A strongly consistent, geographically distributed key-value store at Facebook. 
- **Deployment Year**: First deployed in 2013.
- **Use Cases**: Supports various applications - metadata for distributed filesystem, event counting, product data for app features.
- **Features**: Offers tunable durability, consistency, availability, latency guarantees.
- **Popularity Reason**: Flexibility for storing ephemeral and nonephemeral small key-value data.
## Background and Motivation
- **Prior to ZippyDB**: Teams used RocksDB directly, leading to duplicated efforts in solving consistency, fault tolerance, failure recovery, replication, capacity management.
- **ZippyDB Goal**: Provide a durable, consistent key-value data store. Offload data management challenges to ZippyDB.
## Early Design Decisions
- **Infrastructure Reuse**: Emphasis on using existing infrastructure.
- **Key Components**:
  - **Data Shuttle**: A data replication library.
  - **RocksDB**: Underlying storage engine.
  - **Shard Manager**: Manages shard distribution.
  - **Distributed Configuration Service**: Based on ZooKeeper, handles load balancing, shard placement, failure detection, service discovery.
![[Screenshot 2024-01-17 at 2.08.26 PM.png]]
## Architecture
- **Deployment Units**: ZippyDB operates in tiers spread across regions globally.
- **Tiers**: Includes default "wildcard" tier and specialized tiers.
- **Data Division**: Data is split into shards, replicated across regions.
- **Replication Methods**: Uses Data Shuttle, either multi-Paxos or async replication.
- **Shard Configurations**:
  - **Paxos Quorum Group**: Synchronous replication for durability and availability.
  - **Followers**: Asynchronous replication for low-latency reads with relaxed consistency.
![[Screenshot 2024-01-17 at 2.08.38 PM.png]]
## Data Management
- **Data Model**: Simple key-value with standard APIs (get, put, delete, batch variants, iterate over key prefixes, delete key ranges).
- **Advanced Features**: Test-and-set API, transactions, conditional writes.
- **TTL Support**: Time-to-live for ephemeral data, integrated with RocksDB's compaction.
- **ORM Layer**: Abstracts underlying storage details.
## Shard Management
- **Physical Shards (p-shards)**: Basic unit of data management on server.
- **Micro-Shards (μshards)**: Allow partitioning of key space into smaller units.
- **Shard Assignment Types**:
  - **Compact Mapping**: Static assignments, changed for large/hot shards.
  - **Akkio Mapping**: Dynamic mapping by Akkio service, optimizes for low latency access.

![[Screenshot 2024-01-17 at 2.08.51 PM.png]]
## Data Replication and Consistency
- **Data Shuttle and Multi-Paxos**: Used for synchronous replication.
- **Epochs and Leadership**: Time divided into epochs with unique leaders, managed by ShardManager.
- **Consistency Levels**: Configurable per request - eventual, read-your-writes, strong.
- **Read/Write Trade-offs**: Options like fast-acknowledge mode for performance with lower consistency guarantees.
## Transactions and Conditional Writes
- **Transactions**: Serializable, use optimistic concurrency control.
- **Conditional Writes**: Implemented as server-side transactions with preconditions (key_present, key_not_present, value_matches_or_key_not_present).
![[Screenshot 2024-01-17 at 2.09.03 PM.png]]
## Future of ZippyDB
- **Ongoing Evolution**: Adapting to changing ecosystem and product requirements.
- **Planned Improvements**: Storage-compute disaggregation, changes in membership management, failure detection and recovery, distributed transactions.
## Conclusion
- **Impact**: ZippyDB's flexibility in efficiency, availability, performance trade-offs led to steep adoption.
- **Engineering Efficiency**: Optimizes use of engineering resources and key-value store capacity at Facebook.
## Technical Considerations for Software Engineers
- **Replication Strategies**: Understanding of Paxos vs. asynchronous replication and their impact on system performance and reliability.
- **Shard Management**: Importance of efficient shard distribution and management for scalability and performance.
- **Consistency Models**: Trade-offs between different consistency levels and their practical implications in distributed systems.
- **Transactional Support**: Role of transactions in maintaining data integrity and consistency.
- **System Evolution**: Continuous adaptation and evolution of systems like ZippyDB to meet changing requirements and technological advancements.