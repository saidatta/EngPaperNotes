## Table of Contents

* [Abstract](#abstract)
* [Introduction](#introduction)
* [Conclusion](#conclusion)


### **Abstract:** {#abstract}

Introduces an updated version of Viewstamped Replication, a replication method that addresses crash failures in nodes. It covers client request handling, group reorganization upon replica failure, and the process of a failed replica rejoining the group. Additionally, the paper highlights essential optimizations and presents a protocol for managing reconfigurations, which can modify both the group membership and the number of failures the group can handle.

### **1 Introduction:** {#introduction}

An updated version of Viewstamped Replication (VR), which is designed to work in asynchronous networks and handle crash failures in nodes. VR supports replicated services running on multiple replica nodes, providing state machine replication that allows clients to observe and modify the service state. This makes it suitable for implementing replicated services like lock managers or file systems.

This updated version of VR has several improvements over previous papers:

-   A simpler and better-performing protocol, with some enhancements inspired by Byzantine fault tolerance work.
-   No need for disk usage, as the protocol now relies on replicated state for persistence.
-   A reconfiguration protocol that allows changing the membership of the replica group and the number of failures the group can handle.
-   The protocol is presented independently of any specific applications, making it easier to understand and separate from application-specific details.

VR was developed in the 1980s, around the same time as Paxos, but without knowledge of that work. It differs from Paxos in that it is a replication protocol rather than a consensus protocol, using consensus as a part of supporting a replicated state machine. Another distinction is that VR did not require disk I/O during the consensus protocol used to execute state machine operations.

The paper is organized into sections covering background, an overview of the approach, the VR protocol, implementation details, optimizations, the reconfiguration protocol, correctness, and conclusions.

### 2 Background:

This section discusses the environment assumptions for VR, the number of replicas needed for correct behavior, and the system configuration for using VR.

#### 2.1 Assumptions:

-   VR handles crash failures, where nodes either function correctly or stop completely.
-   VR does not handle Byzantine failures (arbitrary failures possibly caused by malicious attacks).
-   VR is designed for asynchronous networks like the Internet.
-   Messages may be lost, delivered late, out of order, or delivered more than once, but sending them repeatedly ensures eventual delivery.
-   The network is assumed not to be under attack by malicious parties. If needed, cryptography can be used to obtain secure channels.

#### 2.2 Replica Groups:

-   VR ensures reliability and availability when no more than f (a threshold) replicas are faulty, using replica groups of size 2f + 1.
-   The rationale for this size is based on needing enough replicas to execute a request without waiting for f potentially crashed replicas and ensuring a non-empty intersection between quorums for successive steps in the protocol.
-   For a given threshold f, there's no benefit in having a group larger than 2f + 1. The paper assumes this exact group size.

#### 2.3 Architecture:

-   The VR architecture includes client machines running user code and the VR proxy, and replica nodes running the service code and the VR code.
-   Clients communicate with the VR proxy, which communicates with replicas to carry out operations and return results to the client.
-   Of the 2f + 1 replicas, only f + 1 need to run the service code. This point is discussed further in Section 6.1.
  
### 3 Overview Summary:

State machine replication requires replicas to start in the same initial state and execute operations deterministically. The replication protocol must ensure that operations execute in the same order at all replicas despite concurrent client requests and failures. VR achieves this by using a primary replica to order client requests, with other replicas acting as backups. The system moves through a sequence of views, each with a different primary replica.

VR addresses primary replica failure by allowing different replicas to assume the primary role over time. The system moves through a sequence of views, selecting a new primary for each view. Backups monitor the primary, and if it appears faulty, they execute a view change protocol to select a new primary.

To work correctly across view changes, the system's state in the next view must reflect all client operations executed in earlier views. VR achieves this by having the primary wait until at least f + 1 replicas know about a client request before executing it and initializing the state of a new view by consulting at least f + 1 replicas.

VR also provides a way for failed nodes to recover and continue processing, which is crucial for maintaining the system's reliability. Correct recovery requires the recovering replica to rejoin the protocol only after it knows a state at least as recent as its state when it failed.

VR uses three sub-protocols to ensure correctness:

1.  Normal case processing of user requests.
2.  View changes to select a new primary.
3.  Recovery of a failed replica so it can rejoin the group.

These sub-protocols are described in detail in the next section.

### 4 The VR Protocol Summary:

The VR Protocol consists of three main sub-protocols: normal operation, view change, and recovery. This section focuses on the normal operation of the protocol when the primary is not faulty.

The state of the VR layer at a replica includes the configuration, replica number, current view-number, status, op-number, log, commit-number, and client-table. The client-side proxy also maintains its state, including the configuration, view-number, client-id, and request-number.

During normal operation, the request processing protocol proceeds as follows:

1.  The client sends a REQUEST message to the primary replica, containing the operation, client-id, and request-number.
2.  The primary compares the request-number with the information in the client table and decides whether to process or drop the request.
3.  The primary updates its state, adds the request to the log, and sends a PREPARE message to the other replicas.
4.  Backups process PREPARE messages in order, update their state, and send a PREPAREOK message to the primary.
5.  The primary waits for f PREPAREOK messages before considering the operation committed, executing the operation, and sending a REPLY message to the client.
6.  The primary informs backups about the commit, either through the next PREPARE message or by sending a COMMIT message.
7.  Backups execute the operation and update their state when they learn of a commit.

If a client does not receive a timely response, it resends the request to all replicas. The protocol does not require any writing to disk, and backups execute operations quickly to ensure minimal delay when a replica takes over as primary.

#### Section 4.3 - Recovery:

The paper discusses the recovery protocol for when a replica recovers after a crash. The recovering replica must not participate in request processing and view changes until it has a state at least as recent as when it failed. The recovery protocol does not require disk I/O during normal processing or view changes. The protocol consists of three steps:

1.  The recovering replica sends a RECOVERY message to all other replicas.
2.  A replica replies to a RECOVERY message only when its status is normal, sending a RECOVERYRESPONSE message to the recovering replica.
3.  The recovering replica waits to receive at least f+1 RECOVERYRESPONSE messages from different replicas, updates its state using the information from the primary, changes its status to normal, and the recovery protocol is complete.

#### Section 4.4 - Non-deterministic Operations:

Non-deterministic operations can cause state divergence among replicas. To avoid divergence, the primary can predict the value using local information or by requesting values from backups and computing the predicted value as a deterministic function of their responses and its own. The predicted value is stored in the log and propagated to other replicas. When the operation is executed, the predicted value is used.

#### Section 4.5 - Client Recovery:

If a client crashes and recovers, it must start up with a request-number larger than what it had before it failed. The client fetches its latest number from the replicas and adds 2 to this value to ensure the new request-number is big enough.

### Section 5 - Pragmatics:

This section addresses efficient log management for node recovery, state transfer, and view changes to resolve important issues in a practical system.

#### 5.1 Efficient Recovery:

Efficient recovery is achieved by using checkpoints. Every O operations, the replication code requests the application to take a checkpoint. The application records a snapshot of its state on disk and a checkpoint number. The recovering node obtains the application state from another replica using a Merkle tree to determine which pages to fetch. The recovering node then runs the recovery protocol and informs the other nodes of its state by including the checkpoint number in its RECOVERY message. Checkpoints allow for faster recovery and log garbage collection.

#### 5.2 State Transfer:

State transfer is used by a node that is behind to bring itself up-to-date. It sends a GETSTATE message to another replica, which responds with a NEWSTATE message if its status is normal and it is in the current view. When the replica receives the NEWSTATE message, it appends the log in the message to its log and updates its state. If there is a gap between the last operation known to the slow replica and what the responder knows, the slow replica brings itself up to date using application state and then obtains the log from the point.

#### 5.3 View Changes:

To complete a view change efficiently, replicas include a suffix of their log in their DOVIEWCHANGE messages. Sending the latest log entry or the latest two entries should be sufficient in most cases. Occasionally, if this information isn't enough, the primary can ask for more information or use application state to bring itself up to date.

### Section 6 - Optimizations:

This section discusses several optimizations to improve the performance of the protocol, some of which were proposed in the paper on Harp, and others based on later work on the PBFT replication protocol.

#### 6.1 Witnesses:

Witnesses are used to avoid having all replicas actively run the service. The group of 2f + 1 replicas includes f+1 active replicas and f witnesses. The primary is always an active replica. Witnesses are needed for view changes and recovery, but they do not execute operations. This allows most of the time for witnesses to perform other work.

#### 6.2 Batching:

Batching is used to reduce the overhead of running the protocol. Instead of running the protocol for each request, the primary collects multiple requests and runs the protocol for all of them at once. Batching can provide significant gains in throughput when the system is heavily loaded.

#### 6.3 Fast Reads:

Two ways to reduce the latency for handling read requests and improve overall throughput are discussed.

##### 6.3.1 Reads at the Primary:

The primary can execute read requests without consulting the other replicas, reducing message traffic and delay for processing reads. Harp used leases to ensure that the primary doesn't return results based on stale data. This approach also reduces the system's load, especially for primarily read-only workloads.

##### 6.3.2 Reads at Backups:

If it's acceptable for read results to be based on stale information, read requests can be executed at backups. Clients must inform the backup of their previous operation to support causality. This approach provides a form of load balancing and allows backups to be used as caches that satisfy causality requirements. However, unlike the first approach, it does not provide external consistency.

### 7 Reconfiguration 

The reconfiguration protocol allows for changes in the group of replicas over time. These changes may be necessary to replace failed nodes, upgrade hardware, or adjust replica locations. The protocol also enables changes to the failure threshold (f), allowing the system to adapt to changing circumstances.

Key aspects of the reconfiguration protocol:

1.  Reconfiguration is triggered by a special client request and executed through the normal case protocol by the old group.
2.  When the request commits, the system moves to a new epoch, transitioning responsibility for processing client requests to the new group.
3.  Before the new group can process client requests, it must have up-to-date knowledge of all committed operations from the previous epoch. To achieve this, the new replicas transfer state from the old replicas.
4.  The old replicas do not shut down until the state transfer is complete, ensuring a smooth transition between configurations.

In summary, Section 7 introduces a reconfiguration protocol that enables changes to the group of replicas and the failure threshold to adapt to different situations and requirements. The protocol ensures a seamless transition between configurations by transferring the state from the old group to the new group.

#### 7.1 Reconfiguration Details

The reconfiguration protocol includes additional information in the replica state and introduces a new status called "transitioning." Replicas change their status to transitioning at the beginning of the next epoch. New replicas use the old-configuration for state transfer, and replicas that are members of the new group change their status to normal once they receive the complete log up to the start of the epoch. Replicas that are being replaced shut down after transferring their state to the new group.

Every message now contains an epoch-number. Replicas process messages that match their current epoch and update their epoch if they receive a message with a later epoch number.

Reconfigurations are requested by a client, such as an administrator's node. The primary accepts the request if it meets certain conditions, and then processes it using the normal case protocol, with some differences.

Steps in processing a reconfiguration request:

1.  The primary adds the request to its log, sends a PREPARE message to the backups, and stops accepting client requests.
2.  The backups handle the PREPARE message, add the request to their log when they are up to date, and send PREPAREOK responses to the primary.
3.  The primary increments its epoch-number, sends COMMIT messages to the other old replicas, and sends STARTEPOCH messages to new replicas being added. Then, it executes all pending client requests and sets its status to transitioning.

The transition to the new epoch occurs separately for replicas that are members of the new group and replicas that are being replaced. This detailed reconfiguration process ensures a smooth transition between configurations, enabling the system to adapt to changing circumstances effectively.

##### 7.1.1 Processing in the New Group:
-   Replicas that are part of the new epoch group initialize their state to record the old and new configurations, the new epoch-number, and the opnumber, sets its view-number to 0, and sets its status to transitioning when they receive a STARTEPOCH or COMMIT message.
-   If a replica is missing requests from its log, it sends state transfer messages to the old and new replicas to get a complete state up to the op-number, allowing it to learn all client requests up to the reconfiguration request.
-   Once a replica in the new group is up-to-date with the start of the epoch, it sets its status to normal and executes any requests in the log that it hasn’t already executed. If it is the primary of the new group, it starts accepting new requests. Additionally, it sends EPOCHSTARTED messages to the replicas that are being replaced.
-   Replicas in the new group select the primary using a deterministic function of the configuration for the new epoch and the current view number.
-   Replicas in the new group send an EPOCHSTARTED response to a (duplicate) STARTEPOCH message they receive after they have completed state transfer.

##### 7.1.2 Processing at Replicas being Replaced:
-   When a replica being replaced learns of the new epoch, it changes its epoch-number to that of the new epoch and sets its status to transitioning. If it doesn’t have the reconfiguration request in its log, it obtains it by performing state transfer from other old replicas. Then it stores the current configuration in old-configuration and stores the new configuration in configuration.
-   Replicas being replaced respond to state transfer requests from replicas in the new group until they receive f0 + 1 EPOCHSTARTED messages from the new replicas, where f0 is the threshold of the new group. At this point, the replica being replaced shuts down.
-   If a replica being replaced doesn’t receive the EPOCHSTARTED messages in a timely way, it sends STARTEPOCH messages to the new replicas (or the subset of those replicas it hasn’t already heard from). New replicas respond to these messages either by moving to the epoch or by sending the EPOCHSTARTED message to the old replica.

#### 7.2 Other Protocol Changes:

-   A replica does not accept messages for an epoch earlier than what it knows, and instead informs the sender about the new epoch.
-   In the view change protocol, the new primary checks the topmost request in the log; if it is a RECONFIGURATION request, it won’t accept any additional client requests. Furthermore, if the request is committed, it sends STARTEPOCH messages to the new replicas.
-   In the recovery protocol, an old replica that is attempting to recover while a reconfiguration is underway may be informed about the next epoch. If the replica isn’t a member of the new replica group, it shuts down; otherwise, it continues with recovery by communicating with replicas in the new group. RECONFIGURATION requests that are in the log but not in the topmost entry are ignored because the reconfiguration has already happened.

#### 7.3 Shutting down Old Replicas 
The protocol described above allows replicas to recognize when they are no longer needed so that they can shut down. However, we also provide a way for the administrator who requested the reconfiguration to learn when it has completed. This way machines being replaced can be shut down quickly, even when, for example, they are unable to communicate with other replicas because of a long-lasting network partition.

Receiving a reply to the RECONFIGURATION request doesn’t contain the necessary information since that only tells the administrator that the request has committed, whereas the administrator needs to know that enough new nodes have completed state transfer. To provide the needed information, we provide another operation, `CHECKEPOCH e, c, s`; the administrator calls this operation after getting the reply to the RECONFIGURATION request. Here `c` is the client machine being used by the administrator, `s` is `c`'s request-number, and `e` is the new epoch. The operation runs through the normal case protocol in the new group, and therefore when the administrator gets the reply this indicates the reconfiguration is complete.

It’s important that the administrator wait for the reconfiguration to complete before shutting down the nodes being replaced. The reason is that if one of these nodes were shut down prematurely, this can lead to more than `f` failures in the old group before the state has been transferred to the new group, and the new group would then be unable to process client requests.

#### 7.4 Locating the Group 
Since the group can move, a new client needs a way to find the current configuration. This requires an out-of-band mechanism, e.g., the current configuration can be obtained by communicating with a web site run by the administrator.

Old clients can also use this mechanism to find the new group. However, to make it easy for current clients to find the group, old replicas that receive a client request with an old epoch number inform the client about the reconfiguration by sending it a `NEWEPOCH e, v, newconfig` message.

#### 7.5 Discussion
Paper findings acknowledge the delay caused by the reconfiguration process and suggest warming up new nodes before the reconfiguration. During this time, the old group can continue to process client requests. The RECONFIGURATION request is only sent when the new nodes are almost up to date, resulting in a shorter delay until the new nodes can start handling client requests.

Additionally, It has been noted that the reconfiguration protocol can be used for several purposes beyond adding and removing replicas. For example, it can be used to change the replica set's parameters or even to switch between different replicated state machines. They also mention that the reconfiguration protocol can be used in conjunction with a Byzantine fault tolerance (BFT) algorithm to provide robustness against malicious nodes.

Finally, Paper also discuss some of the trade-offs of their approach, noting that while their protocol provides strong consistency guarantees and fault tolerance, it is not designed for high-performance applications. They suggest that the protocol is suitable for systems with moderate performance requirements, such as replicated file systems or replicated databases.

### Section 8: Correctness of View Changes and Recovery Protocol in VR

Section 8 of the VR paper provides an informal discussion of the correctness of the protocol. In this section, we'll discuss the correctness of the view change protocol ignoring node recovery and the correctness of the recovery protocol.

#### 8.1 Correctness of View Changes:

The safety correctness condition for view changes in VR is that every committed operation survives into all subsequent views in the same position in the serial order. This means that any request that had been executed retains its place in the order.

This condition holds in the first view. Assuming it holds in view v, the protocol will ensure that it also holds in the next view v0. The reasoning is that normal case processing ensures that any operation that committed in view v is known to at least f + 1 replicas, each of which also knows all operations ordered before o, including all operations committed in views before v. The view change protocol starts the new view with the most recent log received from f + 1 replicas. Since none of these replicas accepts PREPARE messages from the old primary after sending the DOVIEWCHANGE message, the most recent log contains the latest operation committed in view v (and all earlier operations). Therefore all operations committed in views before v0 are present in the log that starts view v0 in their previously assigned order.

It's crucial that replicas stop accepting PREPARE messages from earlier views once they start the view change protocol. Without this constraint, the system could get into a state in which there are two active primaries: the old one, which hasn't failed but is merely slow or not well connected to the network, and the new one. If a replica sent a PREPAREOK message to the old primary after sending its log to the new one, the old primary might commit an operation that the new primary doesn't learn about in the DOVIEWCHANGE messages.

The liveness of the protocol depends on properly setting the timeouts used to determine whether to start a view change so as to avoid unnecessary view changes and allow useful work to get done.

#### 8.2 Correctness of the Recovery Protocol:

The safety correctness condition for the recovery protocol in VR is that when a recovering replica changes its status to normal, it does so in a state at least as recent as what it knew when it failed.

When a replica recovers, it doesn't know what view it was in when it failed. However, when it receives f + 1 responses to its RECOVERY message, it is certain to learn of a view at least as recent as the one that existed when it sent its last PREPAREOK, DOVIEWCHANGE, or RECOVERYRESPONSE message. Furthermore, it gets its state from the primary of the latest view it hears about, which ensures it learns the latest state of that view. In effect, the protocol uses the volatile state at f +1 replicas as stable state.

The nonce is needed because otherwise, a recovering replica might combine responses to an earlier RECOVERY message with those to a later one; in this case, it would not necessarily learn about the latest state.

The key to correct recovery is the combination of the view change and recovery protocols. In particular, the view change protocol has two message exchanges (for the STARTVIEWCHANGE and DOVIEWCHANGE messages). These ensure that when a view change happens, at least f + 1 replicas already know that the view change is in progress. Therefore, if a view change was in progress when the replica failed, it is certain to recover in that view or a later one.

Having two message exchanges is necessary. If there were only one exchange, i.e., just an exchange of DOVIEWCHANGE messages, a scenario is possible where a recovering replica recovers in an incorrect view. The round of STARTVIEWCHANGE

#### 8.3 Correctness of the Reconfiguration

##### Safety

-   Reconfiguration is correct because it preserves all committed requests in the order selected for them.
-   The RECONFIGURATION request is the last committed request in the old epoch.
-   New replicas do not become active until they have completed state transfer, so they learn about all requests that committed in the previous epoch and their order is preserved.
-   It's possible for the primaries in both the old and new groups to be active simultaneously if the primary of the old group fails after the reconfiguration request commits, but processing in the old group cannot interfere with the ordering of requests that are handled in the new epoch.
-   The new epoch must start in view 0 to avoid having two primaries in the new group, which would be incorrect.

##### Liveness
-   The system is live because (1) the base protocol is live, ensuring that the RECONFIGURATION request will eventually be executed in the old group; (2) new replicas will eventually move to the next epoch; and (3) old replicas do not shut down until new replicas are ready to process client requests.
-   Old replicas wait for f0+1 EPOCHSTARTED messages before shutting down, ensuring that enough new replicas have their state to process client requests assuming no more than a threshold of failures in the new group.
-   Old replicas might shut down before some new replicas have finished state transfer, but this can happen only after at least f0+1 new replicas have their state, and the other new replicas can get up to date by doing state transfer from the new replicas.
-   Old replicas shut down by the administrator do not cause a problem if the administrator waits until after executing a CHECKEPOCH request in the new epoch, ensuring that at least f0+1 replicas in the new group have their state, and after this point the old replicas are no longer needed.

#### 9 Conclusion
-   This section provides a summary of the paper's contributions.
-   The paper presents an improved version of Viewstamped Replication, a protocol used to build replicated systems that can tolerate crash failures.
-   The protocol does not require any disk writes during normal processing or view changes, yet allows nodes to recover from failures and rejoin the group.
-   The paper also presents a reconfiguration protocol that allows for changes in the replica group's membership and failure threshold.
-   Reconfiguration is necessary for the protocol to be deployed in practice since the systems of interest are typically long-lived.
-   The paper describes various optimizations that make the protocol efficient, including using application state for state transfer, keeping the log small, storing service state at only f + 1 replicas, and reducing latency of reads and writes.
-   The hope is that this paper will be helpful to those developing the next generation of reliable distributed systems, as there is increasing use of replication protocols that handle crash failures in modern large-scale distributed systems.
