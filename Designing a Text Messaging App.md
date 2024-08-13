#### Overview
Designing a text messaging app for a large user base involves critical considerations in scalability, latency, fault tolerance, and privacy. This guide outlines the requirements and initial design approaches for creating a robust messaging system that can handle billions of short text messages without compromising performance or data integrity.
#### 14.1 Functional and Non-functional Requirements
- **Real-Time vs. Eventually-Consistent Delivery**: Analyze whether messages need to be delivered instantly or if eventual consistency is acceptable.
- **User Capacity in Chatrooms**: Support for 2 to 1,000 users per chatroom.
- **Message Size**: Limit messages to 1,000 UTF-8 characters, approximately 4 KB per message.
- **Privacy and Security**: Implement end-to-end encryption to ensure message confidentiality; messages should not be readable by the service provider.
- **Data Storage**: Each user can access up to 10 MB of their message history, requiring 10 PB of total storage for one billion users.
- **Scalability**: Handle 100K simultaneous users, assuming extensive connectivity among users and devices.
- **High Availability**: Target four nines availability (99.99% uptime).
- **Performance**: Achieve a P99 message delivery time of 10 seconds.
- **Consistency**: Manage the delivery of messages in a way that does not require strict ordering but ensures a logical sequence visible to users.
#### 14.2 Differences from Notification Services
- **Priority Handling**: Unlike notification systems with variable priority levels, all messages in the app have equal priority with a strict delivery timeline.
- **Service Complexity**: Messaging involves direct client-to-client interactions within a unified system without the need for multiple delivery channels or templates.
- **Encryption Requirements**: Stronger privacy and security measures due to the personal nature of text messages.
#### 14.3 Initial High-Level Design
- **Client Interaction**: Users select recipients, compose messages, and send these via client applications using the recipient's public key for encryption.
- **Service Architecture**: 
  - **Sender Service**: Manages sending messages to recipients and logging them.
  - **Message Service**: Stores messages and handles retrieval requests for sent and received messages.
  - **Connection Service**: Manages user connections, including contact lists and public keys.
![Figure 14.1: High-level service architecture](
#### Messaging Infrastructure
- **WebSocket Connections**: Maintain persistent connections to facilitate real-time message delivery.
- **Encryption at Transit**: Messages are encrypted using the recipient’s public key to ensure security during transmission.
#### 14.4 Connection Service
- **Endpoints**:
  - **GET /connection/user/{userId}**: Retrieve all connections for a user.
  - **POST /connection/user/{userId}/recipient/{recipientId}**: Send a new connection request.
  - **PUT /connection/user/{userId}/recipient/{recipientId}/request/{accept}**: Accept or reject a connection request.
  - **PUT /connection/user/{userId}/recipient/{recipientId}/block/{block}**: Block or unblock a user.
  - **DELETE /connection/user/{userId}/recipient/{recipientId}**: Delete a connection.
#### 14.5 Sender Service Architecture
- **New Message Handling**: Receives messages from senders, processes, and logs them into the system.
- **Message Delivery**: Implements efficient delivery mechanisms to handle high traffic and ensure message integrity and order.
- **Fault Tolerance**: Utilizes failover mechanisms and replication to handle potential system failures without losing messages.
![Figure 14.5: Sender service workflow](
#### Fault Tolerance and Scalability
- **Traffic Surge Management**: Capable of handling unpredictable spikes in message traffic without degradation of service quality.
- **Replication and Redundancy**: Uses replication across multiple servers to ensure data integrity and system availability.
#### Privacy and Security Measures
- **End-to-End Encryption**: Encrypts messages from the sender to the recipient, ensuring that intercepted messages cannot be read by third parties, including the service provider.
- **Authentication**: Requires robust user authentication mechanisms to prevent unauthorized access.
#### Performance Optimization
- **Load Balancing**: Implements load balancing to distribute traffic evenly across servers, minimizing latency and maximizing throughput.
- **Data Partitioning**: Utilizes data partitioning to manage large volumes of messages efficiently, ensuring quick access and retrieval.
#### Challenges and Considerations
- **Data Consistency**: Manages eventual consistency effectively to ensure that all users receive messages in a timely and reliable manner.
- **Service Integration**: Ensures seamless integration between different services (sender, message, and connection services) to provide a cohesive user experience.
#### 14.6 Future Enhancements and Considerations
- **Service Monitoring and Logging**: Implements comprehensive monitoring and logging to track system performance and identify potential issues early.
- **User Experience**: Continuously evaluates user feedback to enhance the application interface and functionality, focusing on ease of use and reliability.
-------
##### Functionality
- **Purpose**: Serves as a log of messages for users to retrieve past messages due to new device logins or undelivered messages during offline periods.
- **Data Security**: Implements end-to-end encryption to ensure messages are secured both in transit and at rest.
- **Storage Management**: Messages have a retention period of a few weeks after which they are deleted to conserve storage and enhance security.
##### End-to-End Encryption Process
1. **Key Generation**: Users generate a public-private key pair.
2. **Encryption**: Senders encrypt messages using the recipient’s public key.
3. **Decryption**: Recipients decrypt messages using their private key.
##### Handling Multiple Devices
- **Data Retention**: Retains messages in the undelivered message service with options for data retrieval across multiple devices.
- **Single Device Login**: Restricts users to log in from one phone at a time to simplify security and encryption protocols.
- **Data Backup**: Offers features for users to back up their message data to cloud services for ease of transfer to new devices.
##### Technology Considerations
- **Database Choice**: Cassandra is selected for its high write capabilities and low read traffic suitability, fitting the high write traffic expected.
- **Statelessness**: The backend service managing messages is stateless, interfacing with a shared Cassandra database.
#### 14.7 Message-Sending Service
##### Introduction
- **Challenges**: Handles the complexity of ensuring messages reach users who cannot act as servers due to security risks, increased traffic, and power consumption concerns.
##### Architecture Overview
- **WebSocket Use**: Maintains persistent WebSocket connections for real-time message delivery.
- **Host Management**: Employs a large cluster of hosts, each assigned to numerous users, managed via a distributed coordination service like ZooKeeper.
##### Failover and Redundancy
- **Host Failure**: Implements strategies for quick failover to standby hosts to minimize downtime.
- **Checkpointing**: Utilizes Redis for checkpointing to ensure message delivery continuity without duplication.

![Figure 14.8: High-level architecture of the message-sending service](
#### Challenges and Solutions
- **Persistent Connections**: Managing persistent connections requires significant memory and computational resources due to the stateful nature of WebSocket connections.
- **Failover Procedures**: Outlines failover steps including heartbeat emissions to devices, and rapid host replacement using a container orchestration system like Kubernetes.
- **Message Deduplication**: Discusses strategies like Redis checkpointing, resending messages with client-side deduplication, and using acknowledgment systems to ensure message integrity.
#### High-Level Architecture of the Message-Sending Service
- **Messaging Cluster**: Comprises a large number of hosts each managing connections with a subset of clients, requiring robust coordination and management to handle host failures and client reassignments efficiently.
- **Host Assigner Service**: A critical component that uses ZooKeeper to assign and reassign hosts to clients dynamically, aiding in quick recovery and load balancing across the messaging infrastructure.
- **Integration with Other Services**: Coordinates with the connection service and message service to ensure seamless messaging operations across different components of the system.
#### Operational Considerations
- **Load Balancing**: Employs strategies to distribute client connections evenly across hosts to prevent any single host from becoming overwhelmed, thus maintaining system performance and reliability.
- **Monitoring and Alerting**: Implements comprehensive monitoring of host status, message delivery metrics, and system health to quickly identify and respond to potential issues.
#### Security and Privacy
- **End-to-End Encryption**: Ensures that all messages are encrypted before being sent, providing a high level of security and privacy for user communications.
- **Public Key Management**: Manages public keys effectively to ensure that messages are always encrypted with the recipient’s current key, accommodating key changes without compromising message security.

#### Summary
Designing the message service and message-sending service for a text messaging app involves intricate system architecture considerations, robust failover mechanisms, and rigorous security measures to handle real-time communications for a large number of users. By leveraging technologies like Cassandra, Redis, and WebSocket along with a sound architectural framework, the system ensures efficient, secure, and reliable messaging capabilities. This detailed design note aids software engineers in understanding and implementing critical components of a scalable and secure text messaging application.