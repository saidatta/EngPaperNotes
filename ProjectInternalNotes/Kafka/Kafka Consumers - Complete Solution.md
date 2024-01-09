https://juejin.cn/post/7142516025458688013
---
## Preface
- **Context**: The article focuses on Kafka consumers, part of mainstream Message Queue (MQ) middleware based on the publish/subscribe model.
- **Key Concepts**: Kafka consumers, consumer groups, rebalancing scenarios, and solutions.
## Consumer Basics
- **Offset Management**: Consumers track read messages using the message's offset. Each message in a partition has a unique offset.
- **Offset Persistence**: The consumer saves the message offset in ZooKeeper (Zk) or Kafka, ensuring the reading state is preserved even after shutdowns or restarts.
## Multiple Consumers
- **Concurrent Consumption**: Kafka allows multiple consumers to read the same message stream without affecting each other, unlike other queue systems where a message, once read, is no longer available to other clients.
- **Consumer Groups**:
  - Consumers can form groups to share a message stream.
  - Each consumer in a group reads messages from a portion of the topic's partitions.
  - Avoid having more consumers than partitions as it leads to idle consumers.
## Consumer Group and Partition Rebalancing
- **Rebalancing**: Refers to the redistribution of partition ownership among consumers in a group. Triggered by adding or deleting consumers or changing partitions.
- **High Availability and Scalability**: Rebalancing improves these aspects but should not occur frequently due to temporary unavailability during the process.
- **Group Coordinator**: A broker responsible for maintaining consumer-group relations and partition ownership.
### Heartbeat Mechanism
- **Active Consumers**: Determined by regular heartbeat signals. Inactive consumers (no heartbeats) are considered dead, triggering rebalancing.
### Partition Allocation
- **JoinGroup Request**: Sent by a consumer joining a group. The first consumer becomes the group owner and allocates partitions to each consumer.
- **Ownership Knowledge**: Each consumer knows only its partition allocation; the group owner knows all allocations.
## Creating a Kafka Consumer
- **Consumer Object**: The first step in reading messages. Requires configurations like `bootstrap.servers`, `key.deserializer`, and `value.deserializer`.
- **Group ID**: Specified by `group.id`, indicating the consumer group membership.
## Subscribing to Topics
- **Topic Subscription**: Consumers can subscribe to specific topics or use regular expressions for dynamic topic matching.
## Polling Mechanism
- **Round-Robin Consumption**: Consumers retrieve messages and handle events like group coordination and offset committing through polling.
- **Thread Safety**: Running multiple consumers on a single thread can cause thread safety issues; typically, each consumer operates on its own thread.
## Consumer Configuration
- **Optional Configurations**: Despite being optional, they play a crucial role in consumer behavior.
  - `fetch.min.bytes`: Minimum data bytes fetched from the server.
  - `fetch.max.wait.ms`: Maximum wait time for data fetching.
  - `auto.offset.reset`: Determines behavior when no valid offset is found.
  - `enable.auto.commit`: Controls automatic offset committing.
## Commits and Offsets
- **Commit Operation**: Updating the current partition location.
- **Offset Tracking**: Consumers send offset information to the `_consumer_offset` topic, allowing them to resume from the last committed offset.
## Types of Offset Commits
- **Autocommit**: Automatic submission of offsets at regular intervals.
- **Manual Synchronous Commit**: Controlled offset submission after each `poll()` call.
- **Manual Asynchronous Commit**: Non-blocking offset submission without waiting for server response.
- **Synchronous and Asynchronous Combined**: A mix of both methods for balancing safety and throughput.

## Summary and Insights
- Kafka consumers provide a robust mechanism for reading messages from topics.
- Effective consumer group management and offset handling are crucial for ensuring data consistency and avoiding message duplication or loss.
- Choosing the right consumer configuration and commit strategy can significantly impact the performance and reliability of Kafka consumers.

---

Creating detailed Obsidian notes for the article on Kafka's specific offset commits and consumer management involves organizing the content into sections that cover advanced Kafka consumer behaviors, rebalancing mechanisms, and handling specific offset scenarios. Here's how the notes can be structured:

---
## Advanced Consumer Management and Offset Commit
## Committing Specific Offsets
- **Challenge**: Kafka's standard commit methods (`commitSync` and `commitAsync`) process the last offset in a batch. However, there may be a need to commit offsets during batch processing.
- **Solution**: Kafka allows for submitting specific partition and offset maps in `commitSync` or `commitAsync`.
- **Complexity**: This requires intricate code-level management of consumer read states.
## Rebalancing Listeners
- **ConsumerRebalanceListener**: Implement this listener to handle partition rebalancing events.
  - **onPartitionsRevoked**: Executed before rebalancing; used for committing offsets.
  - **onPartitionsAssigned**: Called after rebalancing; before the consumer starts reading messages.
## Seeking Specific Offsets
- **Use-Case**: Starting to read messages from a specific offset rather than from the latest or earliest.
- **Kafka API**: 
  - **`seekToBeginning`** and **`seekToEnd`**: Jump to the start or end of a partition.
  - **`seek`**: Allows setting a specific offset for a partition.
- **Binding Offsets to Messages**: Store offset information in an external database (e.g., MySQL, Hadoop) to preserve message reading states across rebalancing.
## Consumer Exits and Polling
- **Exiting Polling Loops**: Consumers are typically in infinite polling loops and require external triggers to exit gracefully.
- **Methods**:
  - **`consumer.wakeup`**: Safe to call from another thread, throws `WakeupException` to break the loop.
  - **`consumer.close`**: Notifies the group coordinator, triggering rebalancing and avoiding session timeout waits.
## Independent Consumers
- **Scenario**: Using a single consumer to read from all or specific partitions, bypassing consumer groups.
- **Implementation**:
  - Subscribe to a topic or assign specific partitions.
  - Use `consumer.partitionsFor` to dynamically check for new partitions.
- **Note**: Cannot handle new partitions automatically; requires periodic checks or application restarts.
## Custom Partitioner
- **Purpose**: Direct messages to specific partitions based on message content.
- **Implementation**:
  - Custom partitioner class that determines the partition based on message attributes.
  - Example:
    ```java
    public class MyPartitioner implements Partitioner {
        // Implementation details...
    }
    ```
## Producer and Consumer Examples
- **Producer Code**: Sending messages to specific partitions.
- **Consumer Code**: Consuming messages from specific partitions.
### Notes on Simultaneous Actions
- **Restrictions**: Certain actions, like subscribing to topics and manually assigning partitions, cannot be performed simultaneously.
### Handling New Partitions
- **Periodic Checks**: Regularly use `consumer.partitionsFor` to detect new partitions.
- **Post-Partition Addition**: Restart the application or reallocate partitions after adding new ones.
---
## Committing Specific Offsets
- **Custom Offset Commit**: Kafka enables committing specific offsets during batch processing.
  - **Code Example**:
    ```java
    // Custom map for partition and offset
    Map<TopicPartition, OffsetAndMetadata> currentOffsets = new HashMap<>();
    currentOffsets.put(new TopicPartition("yourTopic", partition), new OffsetAndMetadata(offset));
    consumer.commitSync(currentOffsets);
    ```
## Rebalancing Listeners
- **ConsumerRebalanceListener Implementation**:
  - **onPartitionsRevoked**:
    ```java
    public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
        consumer.commitSync(currentOffsets); // Commit offsets before rebalancing
    }
    ```
  - **onPartitionsAssigned**:
    ```java
    public void onPartitionsAssigned(Collection<TopicPartition> partitions) {
        // Code to execute after rebalancing and before reading messages
    }
    ```
## Seeking Specific Offsets
- **Using `seek` to Start from Specific Offsets**:
  - **Code Example**:
    ```java
    consumer.seek(new TopicPartition("yourTopic", partition), offset);
    ```
## Consumer Exits and Polling
- **Exiting Polling with WakeupException**:
  - **Code Example**:
    ```java
    try {
        while (true) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
            // Process records
        }
    } catch (WakeupException e) {
        // Handle exception for exiting the loop
    } finally {
        consumer.close(); // Close consumer
    }
    ```
## Independent Consumers
- **Direct Partition Assignment**:
  - **Code Example**:
    ```java
    List<TopicPartition> partitions = new ArrayList<>();
    partitions.add(new TopicPartition("yourTopic", 0)); // Assign specific partition
    consumer.assign(partitions);
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(1000));
        // Process records
    }
    ```
## Custom Partitioner
- **Custom Partitioner Class**:
  - **Code Example**:
    ```java
    public class MyPartitioner implements Partitioner {
        public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster) {
            // Custom logic to determine the partition
        }
        // Other necessary methods
    }
    ```
## Producer and Consumer Examples
- **Producer Sending to Specific Partition**:
  - **Code Example**:
    ```java
    KafkaProducer<String, String> producer = new KafkaProducer<>(properties);
    ProducerRecord<String, String> record = new ProducerRecord<>("yourTopic", "key", "value");
    producer.send(record); // Send record
    producer.close(); // Close producer
    ```
- **Consumer Consuming Specific Partition**:
  - **Code Example**:
    ```java
    KafkaConsumer<String, String> consumer = new KafkaConsumer<>(properties);
    consumer.subscribe(Collections.singletonList("yourTopic")); // Subscribe to a topic
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(1000));
        // Process records
    }
    consumer.close(); // Close consumer
    ```
---
