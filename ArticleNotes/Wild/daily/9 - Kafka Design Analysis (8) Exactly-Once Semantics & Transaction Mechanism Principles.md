> **Original**:  
> This article is forwarded from [Technology World](http://www.jasongj.com/kafka/transaction/), the original link is: [http://www.jasongj.com/kafka/transaction/](http://www.jasongj.com/kafka/transaction/)

## 1. Preface

- All Kafka principle descriptions refer to **Kafka 1.0.0** unless otherwise noted.
- Kafka’s transaction mechanism supports:
  1. **Exactly-Once Semantics** (EOS)
  2. **Operation atomicity**
  3. **Resiliency for stateful operations** (e.g., Kafka Streams)

Historically (prior to **0.11.0.0**), Kafka only offered **at-least-once** or **at-most-once** semantics. Now, from **0.11.0.0** onward, it can achieve **exactly-once** using **idempotent producers** + **transactional writes**.

---

## 2. Why Provide a Transaction Mechanism?

### 2.1 Exactly-Once Semantics

For many critical use cases (e.g., financial transactions, stateful streaming apps), **exactly-once** is essential. Previously, one solution was to rely on **idempotent** downstream systems, which:
- Requires the downstream system to handle duplicates.
- Has high implementation overhead and constraints.

Now, **Kafka** itself can guarantee exactly-once semantics if:
1. The **producer** is idempotent (so each partition sees each message once).
2. The **consumer offset** commits are included in the **same atomic transaction** as the produced messages.

### 2.2 Operation Atomicity

Atomic operations mean multiple writes or reads+writes are **all success or all fail**. This:
- Improves data consistency.
- Simplifies recovery from failures (just retry or skip the entire transaction).

### 2.3 Resilience for Stateful Operations

In **Kafka Streams**, for instance, an application:
1. Reads from input topics,
2. Processes/aggregates data,
3. Writes to output topics.

To recover from partial failures, combining read offsets and writes in a single transaction ensures consistent state.

---

## 3. Stages of Implementing Kafka’s Transaction Mechanism

### 3.1 Idempotent Send

**Idempotent Producer** ensures that re-sending messages (due to retries or duplicates) will not cause duplicates in the broker. Kafka introduces:

- **Producer ID (PID)**: a unique ID assigned to each producer upon initialization (not exposed to clients).
- **Sequence Number**: a monotonically increasing counter per `<PID, Topic, Partition>`.

**Broker** also tracks `<PID, Topic, Partition>` and the last committed sequence number. For each incoming message:
- If the sequence number is exactly **one more** than the broker’s stored sequence, it is accepted.
- If it is **less than or equal** to the broker’s stored sequence, it’s a **duplicate** and discarded.
- If it is **more** by more than one, it’s out-of-order → broker rejects it (`InvalidSequenceNumber`).

Thus, single-partition duplication and out-of-order issues are solved for a single producer session.

### 3.2 Transactional Guarantees

#### Why More Is Needed

- Idempotence alone doesn’t guarantee **atomicity** across multiple partitions or atomic offset commits.
- We want to atomically commit a batch of messages (and offset positions) so that they’re either **all visible** or **all invisible** to readers.

#### Transactional Producer

- **Transaction ID**: A user-supplied, stable name for the producer.  
- Internally, Kafka maps `(Transaction ID) -> (PID + epoch)`.  
- If a new producer with the **same Transaction ID** starts up, it increments the **epoch**, invalidating the old producer session.

**Outcome**:
1. **Cross-session** idempotence: The old producer can’t resume sending with the same `(PID, epoch)`.
2. **Transactional recovery**: If an app instance crashes, a new instance can recover and finalize the incomplete transaction (commit or abort) before continuing.

**Caveat**: On the consumer side, “exactly once” isn’t absolute for external readers because older log segments can be truncated or compacted. The main focus is consistency for normal streaming consumers with `isolation.level=read_committed`.

---

## 4. Transaction Mechanism Principles

### 4.1 Transactional Messaging (Atomic Writes)

Kafka 0.11 introduced a **Transaction Coordinator** server module that:
- Manages **transactional status** for producers.
- Maintains a **Transaction Log** (an internal Kafka topic storing transaction metadata).

A producer never directly reads/writes the Transaction Log but communicates via the Transaction Coordinator, which persists statuses to the log.

### 4.2 Committing Consumer Offsets in a Transaction

Many apps do:
1. **Consume** from topic A,
2. **Transform** data,
3. **Produce** to topic B,
4. **Commit offsets** for topic A.

To ensure atomicity:
- The offset commit must be in the **same transaction** as the messages produced.
- Otherwise, partial commits can lead to data loss or duplication.

Thus, Kafka provides:
```java
producer.sendOffsetsToTransaction(offsets, consumerGroupId);
```
Which includes the offset commits in the same transaction as the produced messages.

### 4.3 Control Messages for Transactional States

Kafka introduces **Control Messages** (with a special message type called **Transaction Marker**):
- Not user-visible data.
- Indicate whether the messages from a particular transaction are **COMMIT** or **ABORT**.

A consumer with `isolation.level=read_committed` uses these markers to skip reading aborted messages.

---

## 5. Transaction Process Example

```java
Producer<String, String> producer = new KafkaProducer<>(props);

// Initialize transactions
producer.initTransactions();

// Begin transaction
producer.beginTransaction();

ConsumerRecords<String, String> records = consumer.poll(100);

try {
    // Produce data
    producer.send(new ProducerRecord<>("Topic", "Key", "Value"));

    // Atomically send offsets
    producer.sendOffsetsToTransaction(offsets, "consumer-group-id");

    // Commit the transaction
    producer.commitTransaction();

} catch (ProducerFencedException | OutOfOrderSequenceException | AuthorizationException e) {
    // If error, abort
    producer.abortTransaction();
} finally {
    producer.close();
    consumer.close();
}
```

1. **initTransactions()**: obtains PID+epoch, and recovers any uncompleted transactions.
2. **beginTransaction()**: marks the start of a new transaction (locally).
3. **send(...)**: produce messages (tracked by PID, sequence).
4. **sendOffsetsToTransaction(...)**: include consumer offsets in the same transaction.
5. **commitTransaction()** or **abortTransaction()**: finalize.

---

## 6. Complete Transaction Flow

```mermaid
flowchart LR
    A[Producer] -->|FindCoordinator| B[Transaction Coordinator]
    B -->|InitPid| C[Transaction Log <br>(internal topic)]
    A -->|AddPartitionsToTxn| B
    A -->|ProduceRequest| D[Broker / Partition Leader]
    A -->|AddOffsetsToTxn| B
    B -->|TxnOffsetCommit| E[Consumer Coordinator (group offset)]
    A -->|EndTxn<br>(commit/abort)| B
    B -->|Transaction Marker (COMMIT / ABORT)| D
    B -->|Write final COMPLETE_COMMIT/ABORT| C
```

1. **FindCoordinator**: Producer locates the Transaction Coordinator for its Transaction ID.  
2. **InitPid**: Assign a (PID + epoch), stored in Transaction Log.  
3. **AddPartitionsToTxn**: Producer declares which `<Topic, Partition>` it will write.  
4. **ProduceRequest**: Actual messages are appended to partition logs with sequence numbers.  
5. **AddOffsetsToTxn**: For consumer offsets, similarly add these partitions to the transaction.  
6. **TxnOffsetCommit**: Commit offset data to `__consumer_offsets` with the same PID.  
7. **EndTxn**: Producer requests commit or abort.  
8. **Transaction Marker**: Coordinator writes COMMIT or ABORT markers into the partitions.  
9. **Transaction Log** final entry: Mark the transaction as complete.

---

## 7. Comparison with Other Transaction Mechanisms

### 7.1 PostgreSQL MVCC
- Both Kafka’s transaction approach and PostgreSQL’s **MVCC** rely on *marking* data as committed or aborted without physically removing data.  
- In rollback, Kafka’s approach discards the messages via an **ABORT marker**, while PostgreSQL sets an *xmax* or *abort* marker.

### 7.2 Two-Phase Commit (2PC)

**Kafka** differs from classical 2PC:
- In Kafka, `PREPARE_COMMIT` or `PREPARE_ABORT` is **local** (Transaction Log). No multi-resource participant voting.  
- If a transaction is declared **PREPARE_COMMIT**, the final result is definitely commit (no indefinite wait for participants).  
- Some partitions can be offline without blocking the entire transaction.  
- Kafka has a **transaction timeout** to prevent indefinite in-flight transactions.  
- There are multiple **Transaction Coordinator** instances for horizontal scalability.

### 7.3 ZooKeeper Atomic Broadcast
- ZK’s atomic broadcast is single-phase (all or nothing).  
- Kafka can either COMMIT or ABORT. ZK only “commits” the message if it has a majority ack. If not, it’s effectively aborted, but the concept is simpler.

---

## 8. Exception Handling and Transaction Expiration

### 8.1 Common Exceptions

- **InvalidProducerEpoch**: Fatal. Means a newer producer with the same Transaction ID has taken over.  
- **InvalidPidMapping**: The coordinator can’t find that PID mapping → Producer must re-init.  
- **NotCoordinatorForTransactionalId**: The node is no longer the coordinator → Producer re-fetches.  
- **CoordinatorNotAvailable**: Coordinator not initialized → Producer retries.  
- **DuplicateSequenceNumber**: The message is already processed. Producer can ignore.  
- **InvalidSequenceNumber**: Fatal, indicates out-of-order or truncated log. Producer stops.

### 8.2 Coordinator Failure Handling

- If coordinator fails before `PREPARE_COMMIT` is fully persisted, the new coordinator repeats or aborts the process.  
- If some partitions are offline at commit time, they remain invisible until they come back online, then the coordinator replays the transaction markers.

### 8.3 Transaction Timeout

- `transaction.timeout.ms` sets how long a transaction can remain open. If a producer crashes, the coordinator will **abort** the stale transaction.  
- Expired transactions are aborted to free memory and avoid blocking `READ_COMMITTED` consumers.

### 8.4 Transaction ID Expiration

- If a given **Transaction ID** is inactive for `transactional.id.expiration.ms` (default 7 days), the coordinator can remove it from memory and the Transaction Log.  
- Freed resources help scale.

---

## 9. Summary

- Kafka’s transaction mechanism extends **idempotent producer** semantics across multiple partitions and offset commits, enabling **exactly-once** for Kafka → Kafka pipelines.
- **Transaction markers** and **commit/abort** flags determine message visibility for `READ_COMMITTED` consumers.
- **Coordination** is done by the **Transaction Coordinator**, which uses a **Transaction Log** to store states.
- Kafka’s approach is conceptually akin to an internal 2PC-like approach, but specialized for **Kafka partitions** rather than arbitrary distributed resources.
- Producer code uses simple APIs:  
  - `initTransactions()`,  
  - `beginTransaction()`,  
  - `sendOffsetsToTransaction()`,  
  - `commitTransaction()`,  
  - `abortTransaction()`.  

**Kafka** does not provide cross-system distributed transactions (e.g., with an external DB). The transactions apply **only** to data written **to and read from** Kafka itself.

---

## 10. Further Reading

1. [Kafka Documentation on Transactions](https://kafka.apache.org/documentation/#transactions)  
2. [Idempotent Producer Configuration in Kafka Docs](https://kafka.apache.org/documentation/#producerconfigs_idempotence)  
3. [Kafka Streams Exactly-Once Semantics](https://kafka.apache.org/documentation/streams/)  
4. Articles by [Technology World](http://www.jasongj.com/) on Kafka design.  
5. [PostgreSQL MVCC vs. Kafka Transaction Mechanism](http://www.jasongj.com/sql/mvcc/)

```