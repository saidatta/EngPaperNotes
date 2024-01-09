https://juejin.cn/post/7122295644919693343
#### I. Overview of Kafka Transactions
- **Basic Principle**: Transactions in Kafka follow the classic database transaction concept – all operations within a transaction either succeed completely or fail without any effect.
- **Since Version 0.11.0.0**: Introduction of major features like idempotence and transactions.
#### II. Idempotence in Kafka
- **Purpose**: Ensures that even if a producer sends duplicate data to the broker, only one copy is persisted.
- **Configuration**: Enabled by default. Can be turned on/off via the `enable.idempotence` producer configuration.
- **Working Mechanism**:
  - **Primary Key of Messages**: `<PID, Partition, SeqNumber>`.
    - `PID`: ProducerID, a unique identifier for each producer.
    - `Partition`: The partition number to which the message is sent.
    - `SeqNumber`: An auto-incrementing ID assigned to each message by the producer.
#### III. Kafka Transactions
- **Based on Idempotence**: Transactions in Kafka are built upon the foundation of idempotence.
- **Atomic Writes Across Topics/Partitions**: Ensures that all messages in a transaction across multiple topics and partitions are written successfully or none at all.
- **Producer and Consumer Transactions**: While Kafka supports both, consumer transactions rely more on the consumer’s control.
#### IV. Starting and Using Transactions
1. **Starting a Transaction**:
   - Configuration: Set `transactional.id` in producer properties.
   - Example:
     ```java
     Properties properties = new Properties();
     // ... other configurations ...
     properties.setProperty(ProducerConfig.TRANSACTIONAL_ID_CONFIG, "transactional_id_1");
     KafkaProducer<String, String> producer = new KafkaProducer<>(properties);
     ```

2. **Sending Messages in a Transaction**:
   - Initialize and begin a transaction, send messages, and then commit or abort based on success or failure.
   - Example:
     ```java
     producer.initTransactions();
     producer.beginTransaction();
     try {
         // sending messages
         producer.commitTransaction();
     } catch (Exception e) {
         producer.abortTransaction();
     } finally {
         producer.close();
     }
     ```

#### V. Transaction Workflow
1. **Producer Start and Coordinator Allocation**:
   - **Transaction ID Assignment**: When starting, the producer is assigned a Transaction Coordinator based on its `transactional.id`.
   - **Coordinator Role**: The Transaction Coordinator, a specific broker instance, manages the transactions for the producer.
   - **Producer ID (PID) Allocation**: The coordinator assigns a unique PID to the producer, essential for the idempotence and transaction mechanisms.
2. **Pre-Transaction Preparations**:
   - **Topic-Partition Information**: Before sending transactional messages, the producer informs the coordinator about the partitions it intends to write to.
   - **Coordinator's Record-Keeping**: This information is recorded by the coordinator for managing the transaction.
3. **Transactional Message Sending**:
   - **Message Marking**: Messages sent within a transaction are marked differently to distinguish them from regular messages.
   - **Transactional State**: These messages carry information indicating their part in an ongoing transaction.
4. **Transaction Commit or Abort Initiation**:
   - **Producer Request**: Upon completing the message sending, the producer sends a Commit or Abort request to the coordinator.
   - **Coordinator's Role**: The coordinator waits to receive messages from all involved partitions before proceeding.
5. **Transaction Confirmation and Logging**:
   - **Start Logging**: At the start of the transaction, the coordinator logs the transaction's beginning in the `__transaction_state` topic.
   - **Completion Logging**: After receiving all responses, the coordinator logs the transaction's outcome (commit or abort) in the same topic.
6. **Coordinator-Partition Communication**:
   - **Success**: If the transaction is successful, partitions acknowledge the receipt of messages, and the coordinator confirms the transaction completion.
   - **Failure**: In case of failure or abort, partitions discard the messages, and the coordinator logs the transaction as aborted.
#### Additional Details
- **Transactional Messages Handling**: Transactional messages have a specific field indicating their participation in a transaction. This is crucial for the coordinator and partitions to process them correctly within the transactional context.
- **Transactional State Topic (`__transaction_state`)**: This special Kafka topic is critical for managing transaction states, recording starts, and completions of transactions.
- **Follower Synchronization for HW Updates**: HW (High Watermark) updates occur when follower replicas synchronize their data with the leader replica, affecting the transaction's visibility to consumers.
#### VI. Understanding Transaction Internals
- **High-Level Summary**: The article intends to provide a concise understanding of Kafka transactions without delving too deeply into technical details.
- **Further Exploration**: Interested readers can explore more about Kafka transaction internals, including the role of `__transaction_state` topic, handling of PID, and interaction between producers, coordinators, and partitions.
---