https://www.youtube.com/watch?v=fBauP3tija4
#### Overview
- This video focuses on two specific features of a messaging application: Asset Management (for sending/receiving media) and Last Seen Service (user online status).
- It also discusses chaos engineering tests to evaluate system resilience.
#### Asset Management (Video Exchange)
1. **Workflow**:
    - User uploads a video, which is streamed from their phone to cloud storage.
    - The video passes through a load balancer, reaches the asset service, and is stored in an object storage system like S3.
    - It is then distributed across a Content Delivery Network (CDN) for lower latency access.
2. **Unique ID Generation**:
    - After storage, a unique video ID is returned to the user and attached to any text messages sent alongside the media.
    - The recipient's application downloads the content from the CDN to their device for future access.

#### Technical Implementation (Asset Management)
- **Client-Side Upload**:
    ```javascript
    // Pseudo-code for uploading a video
    function uploadVideo(videoFile) {
        const videoID = assetService.uploadToCloud(videoFile); // Uploads to cloud storage
        return videoID; // Returns a unique video ID
    }
    ```
- **Server-Side Processing**:
    ```javascript
    // Pseudo-code for server-side processing
    class AssetService {
        uploadToCloud(videoFile) {
            // Code to handle video streaming to object storage
            // Code to handle CDN distribution
            return generateUniqueVideoID(); // Generates a unique ID for the video
        }
    }
    ```

#### Last Seen Service
1. **Implementation**:
    - A simple service backed by a database that maps user IDs to their last online timestamps.
    - The database should be optimized for efficiency and quick search (e.g., a distributed key-value store).
2. **User Activity Tracking**:
    - User actions (message sending, settings updates) are sent to a Kafka queue.
    - The Last Seen Service consumes these messages to update the state.

#### Technical Implementation (Last Seen Service)
- **Updating Timestamps**:
    ```python
    # Python pseudo-code for updating last seen timestamps
    class LastSeenService:
        def update_last_seen(self, user_id):
            current_time = getCurrentTime()
            last_seen_db.update(user_id, current_time)
    ```
- **Serving Data to Users**:
    - When a user opens the app, a REST API call retrieves the last seen timestamps for all contacts.

#### Chaos Engineering Test: WebSocket Server Failure
- **WebSocket Server Failure**: If a WebSocket server goes down, clients will be disconnected.
- **Resilience Strategies**:
    - Implement retry logic on the client side.
    - Update the WebSocket Manager's Redis database to remove stale connections.
    - Use heartbeats to maintain active connections.

#### System Bottlenecks and Improvements
- **Identifying Bottlenecks**: Synchronous calls, like the one from the Chat Service to the Database, can create bottlenecks.
- **Improvement**: Introduce a queue between the Chat Service and the Cassandra database to make the process asynchronous.

#### Best Practices for Messaging App Design
- **Efficient Media Handling**: Ensure smooth media upload and distribution.
- **Real-Time Status Updates**: Optimize last seen service for quick and efficient updates.
- **Scalability and Resilience**: Design systems to handle component failures gracefully.

#### Interview Preparation
- Discuss the architecture for asset management and last seen features in messaging apps.
- Understand the importance of CDN in media distribution.
- Be prepared to suggest improvements and identify potential system bottlenecks.

#### Further Reading
- In-depth discussion on system design of messaging applications: [crashingtechinterview.com](https://crashingtechinterview.com)
- Technologies like Kafka, Redis, and cloud-based object storage.

#### Appendix
- **Glossary**:
  - **CDN (Content Delivery Network)**: A system of distributed servers that deliver content to users based on their geographic location.
  - **Kafka**: A distributed event streaming platform used for building real-time data pipelines.
- **Code Examples and Slides**: Visual and technical resources for understanding the app's features.

### Creating Obsidian Links and Tags
- Link to related topics like [[Real-Time Data Streaming with Kafka]], [[CDN Usage in Modern Apps]], [[Building Resilient Systems]].
- Use tags like #MessagingApp, #AssetManagement, #UserStatus, #ChaosEngineering for easy retrieval and cross-referencing.