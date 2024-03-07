- https://jepsen.io/analyses/hazelcast-3-8-3
## Overview
- **Hazelcast**:
  - Provides distributed data structures with intuitive APIs and optional persistence.
  - Used for synchronization of databases, caches, session stores, messaging buses, service discovery.
  - Can be embedded directly into JVM apps or interact via network clients in multiple languages.
  - Offers transparent distribution with Java APIs for various datatypes.
## Safety Guarantees
- **Documentation Implications**:
  - Suggests that operations on Hazelcast objects provide safety guarantees.
  - Example: AtomicReferences offers guaranteed atomic compare-and-set across a cluster.
  - Reality: compare-and-set on AtomicReferences is **not atomic**.
## Network Partitioning: Split-Brain Syndrome
- **Network Partition Event**:
  - Every component of the network continues to run independently post partition.
  - Potential for lost-update scenario especially with MapStore persistence for Maps.
  - Conflicting entries might be written to their backing database by both clusters.
  - Issue also exists without a backing database, leading to the potential misunderstanding that in-memory storage wouldn’t experience lost updates.
- **Split-brain merging**:
  - Pertains only to Maps.
  - Conflicting values for a key are merged via configurable merge policy.
  - Built-in policies:
    - Non-commutative heuristics (larger or smaller cluster wins).
    - Last-write-wins.
    - Higher-hits-wins.
  - As seen in prior Jepsen analyses:
    - All techniques can result in loss of committed updates.
    - For datatypes other than Maps, updates to the smaller cluster are discarded.
## Split-Brain Protection
- **Hazelcast's Split-Brain Protection**:
  - Specify minimum cluster size for operations.
  - Achieved by defining/configuring split-brain protection cluster quorum.
  - If cluster size < defined quorum:
    - Operations are rejected.
    - Rejected operations throw `QuorumException`.
  - Applications continue operations on remaining operating cluster.
  - Instances connected to smaller clusters receive exceptions (potential for alerts).
  - **Key Point**: Applications prevented from continuing with stale data.
- **False Implication**:
  - Implies safe updates for Maps, Transactional Maps, Caches, Locks, and Queues with a majority quorum policy.
  - Reality: This is **not** the case.
- **Delay in Protection**:
  - Time lag (seconds or tens of seconds) before cluster adjusts to exclude unreachable members.
  - Gap between network partitioning and application of Split-Brain Protection.
  - **Result**: Operations on most Hazelcast datatypes may be lost when network is unreliable.
## Analysis: Experimental Confirmation
- Jepsen analysis aims to test the behavior of various Hazelcast datatypes under network failures.
## Key Takeaways:
1. **Documentation Discrepancy**: There's a difference between Hazelcast's documentation promises and the real behavior, particularly with atomicity.
2. **Network Partitions**: Can lead to data inconsistencies and lost updates even in purely in-memory configurations.
3. **Built-in Merge Policies**: Can result in loss of committed updates.
4. **Split-Brain Protection**: Although a protective measure, there's a time delay before it activates, leaving a vulnerability window.
5. **Experimental Verification**: Jepsen analysis underscores the need to test and understand behavior in network failure scenarios.
## Potential Action Points:
- Consider tightening the timing for split-brain protection.
- Improve documentation to make the limitations clearer.
- Investigate potential improvements to the merging policies.
---

