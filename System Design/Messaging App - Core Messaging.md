https://www.youtube.com/watch?v=lphB84Ol88A
#### Overview
- This session focuses on the system design of a messaging application like WhatsApp, Facebook Messenger, or Slack.
- Emphasis on higher-level components and workflows, specifically one-on-one messaging, group messaging, asset service, and presence service.
#### Core Requirements
1. **One-on-One Messaging**
2. **Group Messaging**
3. **Asset Service (Media Handling)**
4. **User Presence Service (Last Seen)**
#### One-on-One Messaging
- **Components**:
  - WebSocket Servers: Maintain connection lifecycle, no routing logic.
  - WebSocket Manager: Tracks user connections and facilitates message routing.
  - Chat Service: Handles message storage and delivery logic.
![[Screenshot 2024-01-26 at 11.15.01 PM.png]]
- **Workflow**:
  - Persistent connections are established to WebSocket servers.
  - Users update their state in the WebSocket Manager upon connection.
  - Messages flow through WebSocket connections to the Chat Service, stored in Cassandra, and then delivered.

- **Technical Implementation**:
  ```python
  class ChatService:
      def send_message(self, sender_id, recipient_id, message):
          # Store message in Cassandra
          # Determine recipient's connection state from WebSocket Manager
          # Deliver message accordingly (directly or through Kafka)
  ```

#### Group Messaging
- **Challenges**:
  - Delivering messages to a large number of users efficiently.
- **Solutions**:
  - Kafka cluster with a topic per group.
  - Message delivery based on user activity scores.
  - Active users receive messages in real-time; less active users receive messages with a delay.

- **Technical Implementation**:
  ```python
  class GroupChatService:
      def send_group_message(self, sender_id, group_id, message):
          # Store message in Cassandra
          # Publish message to group's Kafka topic
          # Consumers deliver messages based on user activity score
  ```

#### Asset Service (Media Handling)
- **Functionality**: Efficiently handles the storage and retrieval of media assets like images and videos.
- **Workflow**: Media uploaded by a user is stored in a cloud storage solution (like AWS S3), and a unique asset ID is generated for retrieval.

#### User Presence Service (Last Seen)
- **Functionality**: Tracks and displays when users are online or their last online time.
- **Workflow**: User activities are sent to a Kafka queue. The presence service consumes these messages to update the user's online status.

#### System Limitations and Site Reliability Testing
- **Approach**: Assess system resilience by simulating component failures and traffic spikes.
- **Key Considerations**:
  - Redundancy and fallback mechanisms (e.g., using Kafka for offline message delivery).
  - Scalability of components like WebSocket servers and databases.

#### Best Practices for Messaging App Design
- **Scalable Architecture**: Design components to handle increased loads and maintain performance.
- **Efficient Data Storage**: Use databases like Cassandra for efficient message storage and retrieval.
- **User Experience**: Ensure real-time message delivery for active users and manage resource usage for less active users.

#### Interview Preparation
- Be prepared to discuss the design of core messaging features, including technical challenges and potential improvements.
- Understand the role of different components like WebSocket servers, Kafka, and Cassandra in the system.

#### Further Reading
- Detailed system design concepts for messaging apps: [crashingtechinterview.com](https://crashingtechinterview.com)
- Explore Kafka, Cassandra, and WebSocket technologies in-depth.

#### Appendix
- **Glossary**:
  - **WebSocket**: A protocol providing full-duplex communication channels over a single TCP connection.
  - **Cassandra**: A distributed NoSQL database known for handling large amounts of data.
- **Diagrams and Slides**: Visual aids for understanding the messaging system architecture.

### Creating Obsidian Links and Tags
- Link to related topics like [[WebSocket Communication]], [[NoSQL Databases in Action]], [[Real-Time Messaging Systems]].
- Use tags like #MessagingAppDesign, #SystemArchitecture, #Kafka, #Cassandra for easy retrieval and cross-referencing.