https://juejin.cn/post/7341232502824435762
### Introduction
Redis, traditionally known as a single-threaded in-memory database, introduced a multi-threaded model in version 6.0. This change has sparked discussions about whether Redis is single-threaded or multi-threaded. The essence of Redis remains single-threaded for most operations, but specific tasks have been offloaded to multiple threads to enhance performance, especially for network I/O.
### Single-Threaded Design Rationale
Redis's design philosophy centers around simplicity and performance. Here’s why Redis initially opted for a single-threaded model for its network I/O and data operations:
1. **Avoid Race Conditions**: By using a single-threaded model, Redis avoids complex concurrency control mechanisms and race conditions, ensuring data consistency and reliability.
2. **Avoid Lock Overhead**: Multi-threaded environments require locks to manage concurrent access to shared data. Redis's single-threaded design eliminates the overhead associated with locking mechanisms.
3. **CPU Cache Utilization**: A single-threaded model leverages the CPU cache more efficiently, reducing the performance degradation caused by frequent context switching in multi-threaded environments.
### Redis's Approach to Improving I/O Utilization
Redis employs **I/O multiplexing** to manage multiple network connections efficiently. This method allows Redis to handle thousands of client connections concurrently using a single thread, leveraging system calls like `select`, `poll`, or `epoll`.

**I/O Multiplexing Example:**
```c
int max_clients = 10000;
struct pollfd fds[max_clients];

// Initialize fds array...

while (1) {
    int ret = poll(fds, max_clients, -1);
    if (ret > 0) {
        for (int i = 0; i < max_clients; i++) {
            if (fds[i].revents & POLLIN) {
                // Handle incoming data...
            }
        }
    }
}
```
### Introduction of Multithreading in Redis 6.0
Despite the efficiency of I/O multiplexing, the increasing demand for higher QPS (queries per second) necessitated further performance improvements. Redis 6.0 introduced multithreading specifically for processing network I/O to address this need.
#### Multithreading in Redis 6.0
Redis 6.0 leverages multiple I/O threads to handle the parsing of network requests. These threads read data from client sockets and parse the requests concurrently, which are then processed by the main thread for actual memory operations.

**Multithreading Example in Redis:**
```c
// Pseudo-code to demonstrate multi-threading in Redis

void io_thread_function(void *arg) {
    while (1) {
        int client_fd = get_client_fd();
        char buffer[1024];
        int bytes_read = read(client_fd, buffer, sizeof(buffer));
        if (bytes_read > 0) {
            // Parse the request
            parse_request(buffer);
        }
    }
}

int main() {
    pthread_t io_threads[NUM_IO_THREADS];
    for (int i = 0; i < NUM_IO_THREADS; i++) {
        pthread_create(&io_threads[i], NULL, io_thread_function, NULL);
    }
    
    // Main thread handles data operations
    while (1) {
        process_data_operations();
    }
    
    return 0;
}
```

### Ensuring Thread Safety
Redis 6.0 mitigates concurrency issues by restricting multithreading to network I/O operations. The main thread handles all data read and write operations, ensuring thread safety and maintaining data consistency.
### Summary
The shift to a multi-threaded model for network I/O in Redis 6.0 was driven by the need to handle higher QPS and fully utilize multi-core CPUs. However, Redis retains its single-threaded approach for data operations to avoid complexity and ensure consistency.

### ASCII Visualization of Redis's Multi-threaded Network I/O

```plaintext
+--------------------------------------+
|                                      |
|          Main Thread (Single)        |
|                                      |
| +----------+   +----------+   +----------+ |
| | Network  |   | Network  |   | Network  | |
| | I/O      |   | I/O      |   | I/O      | |
| | Thread 1 |   | Thread 2 |   | Thread 3 | |
| +----------+   +----------+   +----------+ |
|                                      |
|  +----------------------------------+  |
|  |        Data Operations           |  |
|  +----------------------------------+  |
|                                      |
+--------------------------------------+
```

### Key Equations and Concepts

1. **CPU Cache Utilization**:
   \[
   \text{Cache Hit Rate} = \frac{\text{Cache Hits}}{\text{Total Memory Accesses}}
   \]

2. **I/O Multiplexing**:
   - Utilizes system calls like `select()`, `poll()`, or `epoll()` to handle multiple I/O operations.
   \[
   \text{Multiplexing Efficiency} = \frac{\text{Handled I/O Operations}}{\text{Total I/O Requests}}
   \]

### Conclusion

Redis's evolution to include multithreading for network I/O in version 6.0 highlights its commitment to performance and scalability. By carefully balancing the introduction of multithreading while maintaining a single-threaded model for critical operations, Redis ensures high efficiency and reliability in a distributed environment.