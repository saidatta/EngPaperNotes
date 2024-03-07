https://shopify.engineering/server-sent-events-data-streaming
#### Introduction to Server Sent Events (SSE)

- **Context**: Real-time data applications necessitate an efficient method for server-to-client communication. Among various models, Server Sent Events (SSE) stands out for its simplicity and directness in delivering data from the server to the client over an HTTP connection.
- **Use Case**: Shopify's Black Friday Cyber Monday (BFCM) Live Map is a real-time data visualization tool that significantly benefited from migrating to SSE for data streaming, improving performance and reducing data latency.

#### Choosing the Right Real-time Communication Model

- **Communication Models**: 
  - **Push**: Maintains an open connection for real-time updates, best for immediate data push but requires managing a client registry for scalability.
  - **Polling**: Clients periodically request updates, leading to potential resource waste without new messages.
  - **Long Polling**: A hybrid approach that holds a connection open until new data is available, combining the benefits of push and polling with better efficiency.
- **Decision Factors**: The choice depends on specific use case requirements, with SSE being ideal for unidirectional data flow like Shopify’s BFCM Live Map.

#### Server Sent Events Versus WebSocket

- **WebSocket**: Offers a bidirectional communication channel, suitable for scenarios requiring mutual message exchange like chat applications.
- **SSE Advantages for Shopify**:
  - **Unidirectional Data Flow**: Only the server sends data updates, aligning with the data visualization nature of the BFCM Live Map.
  - **Simplicity and Familiarity**: Utilizes standard HTTP requests, simplifying implementation.
  - **Built-in Reconnection**: Automatically retries connection after a dropout, enhancing reliability.
  
#### Implementing SSE in Golang

- **Architecture Simplification**: Replacing the previous complex system with a Flink-based data pipeline and an SSE server for direct client updates.
- **Golang SSE Server Implementation**: Subscribes to Kafka topics, pushing updates to clients instantly. Simplifies the backend, enhances scalability, and significantly reduces latency.

##### Example Code Snippet for SSE Connection Registration in Golang

```go
func (sseServer *SSEServer) handleConnection(rw http.ResponseWriter, req *http.Request) {
    flusher, ok := rw.(http.Flusher)
    if !ok {
        http.Error(rw, "Streaming unsupported!", http.StatusInternalServerError)
        return
    }

    rw.Header().Set("Content-Type", "text/event-stream")
    rw.Header().Set("Cache-Control", "no-cache")
    // Additional headers for keeping the connection alive and CORS setup

    messageChan := make(chan []byte)
    sseServer.NewClientsChannel <- messageChan

    keepAliveTickler := time.NewTicker(15 * time.Second)
    notify := req.Context().Done()

    go func() {
        <-notify
        sseServer.ClosingClientsChannel <- messageChan
        keepAliveTickler.Stop()
    }()

    for {
        select {
        case kafkaEvent := <-messageChan:
            fmt.Fprintf(rw, "data: %s\n\n", kafkaEvent)
            flusher.Flush()
        case <-keepAliveTickler.C:
            // Keep-alive logic
            flusher.Flush()
        }
    }
}
```
- **Client Subscription**: Utilizes the `EventSource` interface for easy client-side subscription to the SSE endpoint.

##### Client-Side Subscription Example

```javascript
eventSource = new EventSource("http://localhost:8081/events");
eventSource.addEventListener('message', e => {
   var data = JSON.parse(e.data);
   console.log(data);
});
```

#### Ensuring SSE Scalability and Performance Under Load

- **Horizontal Scalability**: The SSE server is designed to be horizontally scalable, with load balancing managed by Nginx. This setup supports dynamic scaling to match the demand.
- **Load Testing**: Simulated a high number of connections to determine the server's capacity and ensure it can handle expected traffic volumes, crucial for maintaining uptime and performance during peak events like BFCM.

#### Conclusion and Reflections

- **Success Metrics**: The shift to SSE for the BFCM Live Map resulted in near-instantaneous data delivery to clients, a significant improvement over the previous 10-second minimum delay. The entire data flow, from ingestion to visualization, was streamlined to under 21 seconds.
- **Key Takeaway**: The transition to SSE exemplifies the importance of selecting a communication model that aligns closely with the application's requirements. For Shopify’s BFCM Live Map, SSE offered a simplified, scalable, and efficient solution for real-time data streaming, underscoring the principle of using the right tool for the right job in software engineering practices.