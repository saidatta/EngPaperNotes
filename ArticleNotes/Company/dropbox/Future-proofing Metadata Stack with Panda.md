#### Introduction to Panda
- **Background**: Dropbox's expansion necessitated a more scalable and efficient metadata storage solution beyond its existing sharded MySQL systems, given the massive scale of operations involving petabytes of data and tens of millions of queries per second.
- **Challenge**: The primary challenge was the looming need for a storage layer overhaul to accommodate growing metadata, ensure cost-effectiveness, and enhance performance without doubling hardware resources.
#### Panda: A New Layer in Metadata Storage
- **Solution**: Panda is introduced as a petabyte-scale, transactional key-value store designed to abstract MySQL shards, offering features like ACID transactions, incremental capacity expansion, and data rebalancing.
- **Architecture**: Panda serves as a middle layer between the application layer (Edgestore) and the storage layer (sharded MySQL), simplifying data management and allowing independent iteration on storage optimization.
#### Key Features and Advantages of Panda
- **Data Rebalancing and Transactions**: Panda supports ACID transactions and data rebalancing across nodes, enabling seamless capacity expansion and management of uneven data growth.
- **Abstraction and Flexibility**: By abstracting the underlying storage engine, Panda allows for features like two-phase commit and data rebalancing, making it independent of MySQL's limitations.
- **Unification of Backends**: Panda's introduction aims to unify Dropbox's metadata storage backends, enhancing consistency and reducing complexity in managing different systems.
#### Open Source Alternatives Consideration
- **FoundationDB, Vitess, and CockroachDB**: Each of these open-source solutions was evaluated but found lacking in meeting Dropbox's specific requirements for scale, transaction support, and operational model, leading to the development of Panda.
#### Panda's Design and Implementation
- **API Design**: Panda's API is intentionally minimalistic to ensure predictability, handle partial results, and optimize for non-blocking reads without exposing key locks, thus encouraging developers to design key structures for efficient data access.
- **Multi-Version Concurrency Control (MVCC)**: Adopting MVCC enables ACID transactions without read-locks and supports snapshot reads, optimizing the system for the high read/write ratio of Dropbox's workloads.
- **Range Transfers**: Panda employs a protocol for transferring data ranges among storage nodes, allowing for dynamic data rebalancing, incremental capacity expansion, and seamless backend migration.
#### Testing and Verification
- **Simulation Testing**: Extensive testing, including simulation testing with production-like workloads and fault injection, helps ensure Panda's reliability and correctness in managing metadata.
- **Randomized Testing**: Emphasizing randomized testing has been crucial in identifying and addressing potential issues early in the development process, reinforcing Panda's design for correctness and durability.

#### Lessons Learned and Future Directions

- **Trade-offs and Design Choices**: The development of Panda highlighted the importance of understanding trade-offs in distributed system design, such as choosing MVCC despite its write amplification and garbage collection costs.
- **Incremental Value and Flexibility**: Panda's introduction demonstrates the benefit of a layered architecture that delivers incremental improvements while allowing for future adaptability in storage engine choices.
- **Randomized Testing's Value**: The project underscored the effectiveness of randomized testing in building systems where correctness is paramount, advocating for early and robust testing strategies.

#### Conclusion

- **Impact on Dropbox**: Implementing Panda has significantly improved the scalability, efficiency, and reliability of Dropbox's metadata stack, setting a foundation for future growth and innovation.
- **Ongoing Evolution**: As Dropbox continues to evolve its metadata storage solutions, Panda will play a critical role in enabling scalable, reliable, and cost-effective data management across the platform.

### Technical Details and Examples

```go
// Simplified Panda Interface Example
type Panda interface {
    Write([]KeyValue, []Precondition) (Timestamp, error)
    LatestTimestamp(Key) (Timestamp, error)
    LatestTimestampForRange(KeyRange) (Timestamp, error)
    Get(Timestamp, Key) (Record, error)
    Scan(Timestamp, KeyRange) ([]Record, error)
}
```

- **API Design Rationale**: The API's design is aimed at providing a predictable and reliable foundation for building higher-level abstractions, ensuring that developers can leverage Panda to create scalable metadata storage systems with clear operational guarantees.

### Diagrams and ASCII Art

- **Panda's Place in Metadata Stack**: 
```
[Application Layer (Edgestore)] 
                |
           [Panda Layer] 
                |
[Storage Layer (Sharded MySQL)]
```
- **Panda Data Plane Architecture**:
```
[Front End Nodes] --- [Storage Nodes] --- [MySQL]
```
- **Panda Control Plane Components**:
```
[Storage Control] -- [Registry] -- [Keeper]
```

- **Range Transfer Protocol Flowchart**:
```
Begin Transfer -> Lock Range -> Copy Data -> Update Registry -> Unlock Range -> Complete
```

These notes encapsulate Panda's strategic role in Dropbox's metadata infrastructure overhaul, offering a comprehensive

 overview of its design, implementation, and the valuable lessons learned throughout the process.