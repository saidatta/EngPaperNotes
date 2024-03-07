https://juejin.cn/post/7146133960865611806?searchId=2023071508383838FA16B4EC10E6962405
#### I. Storage View
- **Topics and Partitions**: Kafka uses topics and partitions as logical abstractions for users. Each partition has physical representations at the replica level.
- **Log and Log Segmentation**: 
  - **Concept**: Kafka's log is divided into multiple LogSegments to manage large logs effectively.
  - **Physical Storage**: Physically, a Log is stored as a directory, with each LogSegment corresponding to a log file and two index files on disk.
  - **Example**: For a topic "topic-log" with 4 partitions, Kafka storage will have corresponding folders named `topic-log-0`, `topic-log-1`, and so on.

#### II. Log Appending and Indexing
- **Appending Messages**: Messages are appended sequentially to the Log, with only the last LogSegment (activeSegment) being writable.
- **Index Files**: Each LogSegment has an offset index file and a timestamp index file for efficient message retrieval.
- **Offset Management**: LogSegments have a base offset (`baseOffset`) representing the offset of the first message in the LogSegment.
- **File Naming**: Files are named based on the base offset, padded to 20 digits.

#### III. Multiple Copies and Replication
- **Replica Mechanism**: Kafka employs a multi-replica mechanism for partitions, enhancing data redundancy and disaster recovery capabilities.
- **Leader and Follower Replicas**: Each partition has a leader replica (handling read/write requests) and follower replicas (responsible for syncing messages with the leader).
- **Replica Distribution**: Copies of the same partition are distributed across different brokers.

#### IV. Concepts of AR, ISR, and OSR
- **Assigned Replicas (AR)**: All replicas in a partition, including leader and follower.
- **In-Sync Replicas (ISR)**: Replicas maintaining sync with the leader replica. ISR is a subset of AR.
- **Out-of-Sync Replicas (OSR)**: Replicas lagging in synchronization, forming the difference between AR and ISR.
- **Leader Replica's Role**: Tracks lag status of follower replicas, managing ISR and OSR sets.

#### V. High Watermark (HW) and Log End Offset (LEO)
- **High Watermark (HW)**: Identifies the message offset up to which messages are visible to consumers.
- **Log End Offset (LEO)**: Represents the offset of the next message to be written. Determines the visible message range for consumers.
- **Replica Synchronization**: Explains the process of message writing and synchronization among replicas, impacting the values of HW and LEO.

#### VI. Kafka Replication Strategy
- **Synchronous vs Asynchronous Replication**: Kafka's replication is a balance between data reliability and performance, neither purely synchronous nor entirely asynchronous.
- **Replication Example**: Detailed example demonstrating how message writing and replication affect HW and LEO in different replicas.

---

#### I. Kafka Version Evolution
- **Version Milestones**:
  - **0.7 and Earlier**: Initial versions.
  - **0.8**: Introduction of a multi-replica mechanism, marking Kafka's advancement as a distributed messaging middleware.
  - **0.9**: Addition of basic security, authentication, and permission functions.
  - **0.10**: Introduction of Kafka Streams, marking Kafka's entry into distributed streaming.
  - **0.11**: Idempotence and transactions introduced, along with a major reconstruction of the underlying log format, which is still in use.

#### II. Kafka Log Format Evolution
- **v0 Format** (Pre-0.10.0.0):
  - **Structure**: Includes crc32, magic, attributes, keylength, key, value length, and value.
  - **LOG_OVERHEAD**: Comprises offset and message size, totaling 12 bytes.
  - **Message Set Concept**: Basic form for disk storage, network transmission, and compression unit.
![[Screenshot 2023-12-31 at 12.47.38 PM.png]]
- **v1 Format** (0.10.0 to 0.11.0):
  - **Additional Field**: Timestamp, indicating the message creation time.
  - **Magic Value**: Updated to 1.
  - **Timestamp Type**: Configurable via broker-side parameter.
![[Screenshot 2023-12-31 at 12.47.23 PM.png]]
- **v2 Format**:
  - **Record Batch**: Replaces the Message Set.
  - **Enhancements**: Includes length, timestamp delta, offset delta, headers, and other fields for more efficient storage and functionality.
  - **Compression**: Applied to the entire records field in Record Batch.
![[Screenshot 2023-12-31 at 12.47.04 PM.png]]
#### III. Message Compression
- **Inner and Outer Messages**: When compressed, the entire message set is treated as an inner message within a wrapper message.
- **Key Handling**: The key in the compressed outer message is typically null.

#### IV. Practical Application of Kafka
- **Partition Number Setting**:
  - **Factors**: Considerations include anticipated throughput, business scenarios, software and hardware conditions, and load.
  - **ProducerRecord**: Decides the partition number, influenced by key hashing.
  - **Partition Count Impact**: Affects parallelism for producers and consumers. Too many partitions can lead to performance degradation and increased file descriptor usage.
  - **Ordering Guarantees**: For ordered message processing, set the number of partitions to 1.

- **Limitations on Partition Modification**:
  - **Increasing Partitions**: Feasible but requires careful planning, especially for key-based topics.
  - **Decreasing Partitions**: Not supported due to complexity and potential data handling issues.

---

#### I. Pursuing Ultimate Performance in Kafka
1. **Batch Processing**:
   - **Traditional vs. Kafka**: Traditional message middleware focuses on individual messages, while Kafka aggregates and sends batches of messages.
   - **Efficiency**: Kafka reduces the number of required RPCs by batching, significantly improving throughput.

2. **Log Format and Encoding Improvements**:
   - **Evolution**: Kafka’s log format evolved through v0, v1, and v2, with each version increasingly optimized for batch processing.
   - **Variable Field Encoding**: Use of Varints and ZigZag encoding in the latest version to reduce additional field sizes, enhancing network and storage efficiency.

3. **Message Compression**:
   - **Supported Methods**: Kafka supports gzip, snappy, lz4 for message compression.
   - **Trade-off**: Compression reduces network I/O at the cost of increased latency.

4. **Index Creation for Quick Query**:
   - **Index Files**: Each log segment file has corresponding index files to speed up message lookup.
   - **Efficiency Gain**: This improves overall performance by reducing search time.

5. **Partitioning**:
   - **Role in Performance**: Partitioning is a key factor in performance enhancement, often overlooked but more impactful than log encoding or compression.

6. **Consistency Mechanism**:
   - **Unique Approach**: Kafka uses a method similar to PacificA, different from common consensus protocols like Paxos or Raft, enhancing sorting efficiency.

7. **Sequential Disk Writing**:
   - **Design**: Kafka employs a file append method, allowing only new messages to be appended to the end of log files.
   - **OS Optimization**: This approach benefits from the operating system's optimizations for linear read/write operations.

8. **Page Cache Utilization**:
   - **OS-Level Optimization**: Kafka leverages the page cache to reduce disk I/O operations.
   - **Dual Caching**: Avoids dual caching of data in the process and page cache, optimizing memory usage.
![[Screenshot 2023-12-31 at 12.46.45 PM.png]]
9. **Zero Copy Technology**:
   - **Consumption Efficiency**: Kafka uses Zero Copy technology, specifically the Linux `sendfile` system call, to enhance data transfer efficiency over the network.
![[Screenshot 2023-12-31 at 12.46.16 PM.png]]
#### II. Message Reliability Analysis in Kafka
1. **Replica Count**:
   - **Configurability**: The number of replicas can be configured or modified after topic creation.
   - **Trade-off**: More replicas enhance reliability but increase disk and network resource usage.

2. **Producer Client Parameter `acks`**:
   - **Reliability Maximization**: Setting `acks` to -1 (or `all`) maximizes message delivery reliability.
   - **Failure Scenarios**: Different configurations of `acks` affect how Kafka handles message failures and leader replica crashes.

3. **Message Sending Modes**:
   - **Options**: Send-and-forget, synchronous, asynchronous.
   - **Reliability Consideration**: Synchronous or asynchronous modes with notification on exceptions are more reliable.

4. **Retries and Exception Handling**:
   - **Configurable Retries**: Kafka allows setting retries for handling transient failures.
   - **Trade-offs**: Configuring retries can affect message order and latency.

5. **Minimizing Message Loss**:
   - **ISR Set Management**: Parameters like `min.insync.replicas` and `unclean.leader.election.enable` play a crucial role in maintaining data consistency and minimizing message loss.
   - **Broker-Side Parameters**: `log.flush.interval.messages` and `log.flush.interval.ms` control synchronous flushing for enhanced reliability.

6. **Consumer-Side Considerations**:
   - **Automatic Offset Committing**: Configuration of `enable.auto.commit` and manual offset commit strategies affect reliability on the consumer side.
   - **Retrospective Consumption**: Kafka’s capability for retrospective consumption can help recover missed messages.

---
