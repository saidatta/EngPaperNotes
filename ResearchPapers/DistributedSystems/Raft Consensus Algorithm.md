https://supwills.com/post/raft/
## Table of Contents
1. Introduction
2. Replicated State Machines
3. The Raft Consensus Algorithm
4. Raft Basics
5. Leader Election
6. Log Replication
7. Safety
8. Cluster Membership Changes
9. Log Compaction
10. References
## Introduction
In distributed systems, achieving high availability and consistency often involves using a master-slave structure. When failures such as machine downtime or network partitions occur, a backup node must take over to ensure the system remains available. This process must be transparent to the client. The Paxos algorithm was an early solution to this problem but is complex and difficult to implement. 

The Raft algorithm, proposed as an easier-to-understand alternative, simplifies consensus into three key components: leader election, log replication, and safety. This paper summarizes Raft’s approach to these problems and the practical considerations for implementing it.
## Replicated State Machines
![[Screenshot 2024-05-09 at 12.08.36 PM.png]]
### Concept
A replicated state machine ensures consistency across distributed systems. Each replica starts with the same state, processes the same input in the same order, and thus reaches the same state. The Raft consensus algorithm aims to ensure that logs are identical across all replicas, ensuring consistent state machines.
### Process
1. The client sends an operation to the leader node.
2. The leader appends the operation to its log and synchronizes it to a majority of the nodes.
3. Once a majority acknowledge, the leader commits the log entry and applies it to its state machine.
4. The leader then informs the client of the successful operation.
### Characteristics
- **Safety**: Ensures correctness despite non-Byzantine faults like network delays, partitions, packet loss, and reordering.
- **Availability**: As long as a majority of nodes are functional and can communicate, the system remains available.
- **Independence from Timing**: Relies on logical rather than physical clocks to avoid timing-related inconsistencies.
## The Raft Consensus Algorithm
### Challenges with Paxos
- **Complexity**: Paxos is difficult to understand and implement, especially for multi-instance consensus.
- **Implementation Difficulty**: Translating Paxos into practical, working code is challenging.
### Raft’s Approach
Raft simplifies consensus by breaking it into easily understandable sub-problems: leader election, log replication, and safety. Raft reduces the number of states and eliminates non-determinism to simplify implementation and understanding.
## Raft Basics
### Roles
- **Leader**: Manages log replication and handles client requests.
- **Follower**: Passive, responds to requests from the leader or candidate.
- **Candidate**: Runs for leader when an election is triggered.
![[Screenshot 2024-05-09 at 12.08.58 PM.png]]
### Term
- Raft divides time into terms, each starting with an election.
- Terms are consecutive integers.
- Each term has at most one leader.
![[Screenshot 2024-05-09 at 12.18.47 PM.png]]
### Server States
- **Persistent State**:
  - `currentTerm`: The latest term known to the server.
  - `votedFor`: Candidate ID that received the server's vote in the current term.
  - `log[]`: Log entries.
- **Volatile State (All Servers)**:
  - `commitIndex`: Index of the highest log entry known to be committed.
  - `lastApplied`: Index of the highest log entry applied to the state machine.
- **Volatile State (Leader Only)**:
  - `nextIndex[]`: Index of the next log entry to send to each server.
  - `matchIndex[]`: Index of the highest log entry known to be replicated on each server.
### Communication
Raft uses two types of RPCs:
- **RequestVote RPC**: Initiated by candidates to gather votes.
- **AppendEntries RPC**: Used by the leader to replicate log entries and send heartbeats.
## Leader Election
### Process
1. A follower becomes a candidate if it doesn't receive a heartbeat within an election timeout.
2. The candidate increments its term, votes for itself, and sends `RequestVote` RPCs to other servers.
3. Servers respond with their votes. A candidate must receive votes from a majority to become the leader.
4. If no candidate wins (e.g., vote split), candidates increment their terms and start new elections after random timeouts.
### RequestVote RPC
- **Request**:
  - `term`: Candidate's term.
  - `candidateId`: Candidate requesting the vote.
  - `lastLogIndex`: Index of candidate's last log entry.
  - `lastLogTerm`: Term of candidate's last log entry.
- **Response**:
  - `term`: Current term for candidate to update itself.
  - `voteGranted`: `true` if the candidate received the vote.
## Log Replication
![[Screenshot 2024-05-09 at 12.23.57 PM.png]]
### Process
1. The client sends a request to the leader.
2. The leader appends the request to its log and sends `AppendEntries` RPCs to followers.
3. Followers append the entries to their logs and respond to the leader.
4. Once a majority of followers acknowledge, the leader commits the entry and applies it to its state machine, then responds to the client.
### AppendEntries RPC
- **Request**:
  - `term`: Leader's term.
  - `leaderId`: Leader's ID.
  - `prevLogIndex`: Index of log entry immediately preceding new ones.
  - `prevLogTerm`: Term of the preceding log entry.
  - `entries[]`: Log entries to store.
  - `leaderCommit`: Leader’s `commitIndex`.
- **Response**:
  - `term`: Current term.
  - `success`: `true` if the follower contained an entry matching `prevLogIndex` and `prevLogTerm`.

## Safety
![[Screenshot 2024-05-09 at 12.24.25 PM.png]]
### Ensuring Consistency
Raft ensures that logs are consistent across nodes through several mechanisms:
- **Election Restrictions**: A follower only votes for a candidate with an up-to-date log.
- **Log Matching Property**: If logs at different nodes have the same index and term, they are identical up to that index.
- **Commitment Rules**: A new leader commits log entries from the previous term only if they are replicated by a majority in the current term.
![[Screenshot 2024-05-09 at 12.27.26 PM.png]]
### Handling Crashes
- **Follower and Candidate Crashes**: Raft handles these by retrying RPCs until they succeed. Since RPCs are idempotent, retries do not cause issues.
## Cluster Membership Changes
![[Screenshot 2024-05-09 at 12.27.36 PM.png]]
### Joint Consensus Approach
Raft handles membership changes through a two-phase approach:
1. **Joint Configuration**: A transition phase where the configuration includes both old and new servers.
2. **New Configuration**: The system transitions fully to the new configuration once the joint configuration is stable.
![[Screenshot 2024-05-09 at 12.24.43 PM.png]]
### Considerations
- New servers must catch up with the log before gaining voting rights.
- Reducing servers requires careful handling to avoid disruption, including handling server requests during the transition.

## Log Compaction
![[Screenshot 2024-05-09 at 12.27.48 PM.png]]
### Snapshot Mechanism
To manage growing log sizes, Raft uses snapshots to capture the state of the system at a point in time. Snapshots:
- Reduce memory usage.
- Speed up recovery and restart processes.
- Are used when followers lag significantly behind the leader.
### InstallSnapshot RPC
- **Request**:
  - `term`: Leader's term.
  - `leaderId`: Leader’s ID.
  - `lastIncludedIndex`: Index of the last log entry included in the snapshot.
  - `lastIncludedTerm`: Term of the last log entry included in the snapshot.
  - `offset`: Byte offset of the snapshot chunk.
  - `data[]`: Raw bytes of the snapshot chunk.
  - `done`: `true` if this is the last chunk.
- **Response**:
  - `term`: Current term.
## References
- "In Search of an Understandable Consensus Algorithm (Extended Version)" by Diego Ongaro and John Ousterhout
- [Raft Consensus Algorithm Website](https://raft.github.io/)
- [MIT 6.824 Distributed Systems Course](https://pdos.csail.mit.edu/6.824/)