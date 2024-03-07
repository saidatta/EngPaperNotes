https://www.uber.com/blog/how-uber-serves-over-40-million-reads-per-second-using-an-integrated-cache/
#### Introduction

- **Docstore**: A proprietary, distributed database system by Uber, built atop MySQL®, designed to store and manage tens of petabytes (PBs) of data while serving up to tens of millions of requests per second from microservices across various business units.

	#### Growth and Challenges

- **Expansion**: The rapid growth in the user base and use case complexity for Docstore has led to significantly increased request volumes and data storage needs.
- **Latency and Scalability Demand**: Critical microservices require low-latency reads and high scalability, pushing the boundaries of what traditional disk-based storage databases can provide.

#### Docstore Architecture Overview
- **Three-Layer Structure**:
  - **Stateless Query Engine Layer**: Handles query planning, routing, sharding, schema management, node health monitoring, and authentication/authorization.
  - **Stateful Storage Engine Layer**: Manages consensus (via Raft protocol), replication, transactions, concurrency control, and load balancing across MySQL nodes equipped with NVMe SSDs for enhanced throughput.
  - **Control Plane**: Oversees the orchestration and management of the database infrastructure.
![[Screenshot 2024-02-19 at 9.00.00 AM.png]]
#### CacheFront: Uber's Integrated Caching Solution

- **Goal**: Address the scaling challenges and high latency issues without resorting to costly vertical or horizontal scaling methods.
- **Design Principles**:
  - **Integrated Caching Layer**: Seamlessly integrates with Docstore’s query engine to offer low-latency read capabilities and reduce direct reads from disk-based storage.
  - **Opt-In Consistency**: Allows services to selectively use the cache based on their consistency requirements, providing flexibility between strong and eventual consistency.

#### Detailed Design and Implementation

- **Query Engine Integration**: CacheFront is built into the query engine layer, acting as an intermediary cache between client requests and the storage engine. This design allows for independent scaling of caching and storage resources.
- **Redis Interface**: Implements an interface to Redis within the query engine for caching data and managing cache invalidation, optimizing read performance while maintaining data freshness.
- **Cache Aside Strategy**: Adopts a cache aside approach for handling read operations, checking the cache first and falling back to the storage engine as needed. Missed cache entries are asynchronously populated to ensure future read efficiency.
- **Change Data Capture (CDC)**: Utilizes Docstore's CDC system, Flux, to monitor changes in the storage layer and update the cache accordingly, ensuring cache consistency with minimal lag.

#### Advanced Features for Scalability and Resiliency

- **Cache Consistency Verification**: Introduces a compare cache mode that shadows read requests to verify cache-data consistency against the database, ensuring high cache accuracy.
- **Cross-Region Cache Warming**: Implements a novel cache warming strategy to maintain cache warmth across geographical regions, ensuring high availability and minimizing the impact of regional failovers.
- **Negative Caching**: Incorporates negative caching for non-existent data reads, reducing unnecessary database queries and optimizing cache space utilization.
- **Redis Sharding and Circuit Breakers**: Employs sharding across multiple Redis clusters to manage load distribution and uses circuit breakers to prevent overloading individual Redis nodes, enhancing overall system resilience.
- **Adaptive Timeouts**: Dynamically adjusts Redis operation timeouts based on real-time performance metrics, balancing the trade-off between cache hit rates and latency impacts on tail-end requests.
#### Impact and Results
- **Performance Gains**: Demonstrates substantial improvements in request latencies, with significant reductions at the 75th and 99.9th percentiles, alongside more stable latency behavior during traffic spikes.
- **Operational Efficiency**: Reduces the computational resources required for handling peak read workloads, translating to cost savings and reduced operational complexity.
- **Developer Productivity**: By abstracting caching logic into CacheFront, Uber enhances developer productivity, enabling teams to focus on core product development rather than caching mechanics.
- **Scalability and Fault Tolerance**: Validates CacheFront's ability to handle large-scale operational demands with high cache hit rates and effective failover management during regional outages.

#### Conclusion

CacheFront represents a significant leap forward in Uber's data infrastructure, addressing critical challenges in database scalability and read latency. By integrating a sophisticated caching layer within Docstore and leveraging advanced features like CDC for cache invalidation, adaptive timeouts, and cross-region cache warming, Uber has successfully optimized its database operations for high throughput and low-latency reads. This solution not only enhances the performance and efficiency of Uber's microservices but also sets a new standard for integrating caching solutions in distributed database systems.