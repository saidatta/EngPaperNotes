https://juejin.cn/post/6844904200141438984

**Overview**

IO multiplexing is a mechanism that allows a single thread to monitor multiple IO file descriptors (e.g., sockets, files, etc.) simultaneously. Once any of these descriptors is ready for some kind of IO operation (like reading or writing), the application can handle the operation. If no descriptor is ready, the application will block, allowing the CPU to be freed for other tasks.

---
### 1. **What is IO Multiplexing?**
IO multiplexing is a synchronous IO model that enables a single thread to monitor multiple file descriptors and efficiently handle multiple IO events, such as reading from or writing to a network socket, without blocking on each individual descriptor. It is widely used in network applications to manage multiple connections within a single thread, improving scalability.

---
### 2. **Why is IO Multiplexing Necessary?**
Without IO multiplexing, two traditional models exist:
- **BIO (Blocking IO):** In synchronous blocking, the server uses multiple threads. Each thread blocks on IO operations (like `recv()`, `send()`, etc.), leading to inefficiency and heavy resource use when handling many connections.
  **Problems with BIO:**
  - Limited concurrency as threads block during IO.
  - Requires one thread per connection, leading to high overhead.
  - Significant resource waste with many idle connections.
- **NIO (Non-blocking IO):** In non-blocking IO, the server repeatedly polls each file descriptor to check for IO readiness. While it avoids blocking, this method consumes unnecessary CPU cycles when most file descriptors have no activity.
  **Problems with NIO:**
  - CPU waste due to busy looping over file descriptors, even if no data is ready.
  - Non-scalable, especially with high numbers of connections.
**Multiplexing (Current Practice):**  
IO multiplexing addresses these inefficiencies by enabling the server to monitor multiple file descriptors (e.g., sockets) and only act when an IO operation can be performed. Common multiplexing mechanisms include `select()`, `poll()`, and `epoll()`.

---
### 3. **Three Ways to Implement IO Multiplexing**

#### a. **select()**
  - Oldest method, available on almost every platform.
  - Monitors a fixed set of file descriptors and returns those ready for IO.

#### b. **poll()**
  - Similar to `select()`, but removes the fixed upper limit on file descriptors.
  - Uses an array of `pollfd` structures to monitor many file descriptors.

#### c. **epoll()** (Linux only)
  - Scalable and efficient. Ideal for applications handling thousands of connections.
  - Uses an event-driven approach where the kernel notifies about ready file descriptors.
---
### 4. **select() Function Interface**

```c
#include <sys/select.h>
#include <sys/time.h>

int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
```

- `nfds`: Maximum file descriptor number plus one.
- `readfds`, `writefds`, `exceptfds`: Sets of file descriptors to monitor for reading, writing, and exceptions.
- `timeout`: Maximum time to wait for an event, or `NULL` to block indefinitely.

**Helper Macros:**
- `FD_ZERO(fd_set *set)`: Clears the file descriptor set.
- `FD_SET(int fd, fd_set *set)`: Adds `fd` to the set.
- `FD_ISSET(int fd, fd_set *set)`: Tests if `fd` is part of the set.
- `FD_CLR(int fd, fd_set *set)`: Removes `fd` from the set.
---
### 5. **select() Usage Example**

```c
fd_set read_fds;
FD_ZERO(&read_fds);
FD_SET(listen_fd, &read_fds);  // Add listening socket to the set
int max_fd = listen_fd;

while (1) {
    fd_set temp_fds = read_fds;  // Preserve original set
    int ready_fds = select(max_fd + 1, &temp_fds, NULL, NULL, NULL);
    
    if (FD_ISSET(listen_fd, &temp_fds)) {
        int new_fd = accept(listen_fd, NULL, NULL);
        FD_SET(new_fd, &read_fds);  // Add new connection
        if (new_fd > max_fd) max_fd = new_fd;
    }

    // Handle data ready on existing connections
    for (int fd = 0; fd <= max_fd; fd++) {
        if (FD_ISSET(fd, &temp_fds)) {
            // Handle read event on `fd`
        }
    }
}
```
---
### 6. **Disadvantages of `select()`**
- **Limit on FDs:** Limited to 1024 file descriptors (FD_SETSIZE).
- **Linear Scanning:** Inefficient with large numbers of descriptors, as all must be scanned.
- **Overhead:** Each call requires copying the entire descriptor set from user space to kernel space.
---
### 7. **poll() Function Interface**
```c
#include <poll.h>

struct pollfd {
    int fd;         // File descriptor
    short events;   // Requested events (e.g., POLLIN, POLLOUT)
    short revents;  // Returned events
};

int poll(struct pollfd *fds, nfds_t nfds, int timeout);
```
- Removes the FD limit, allowing monitoring of any number of descriptors.
- Uses a pollfd array instead of bitmaps.
---
### 8. **poll() Usage Example**
```c
#define MAX_EVENTS 1024
struct pollfd fds[MAX_EVENTS];
fds[0].fd = listen_fd;
fds[0].events = POLLIN;

while (1) {
    int ready_fds = poll(fds, MAX_EVENTS, -1);
    
    if (fds[0].revents & POLLIN) {
        int new_fd = accept(listen_fd, NULL, NULL);
        // Add to `fds`
    }

    for (int i = 1; i < MAX_EVENTS; i++) {
        if (fds[i].revents & POLLIN) {
            // Handle read event
        }
    }
}
```
---
### 9. **Disadvantages of `poll()`**
- **Linear Scanning:** All descriptors still need to be polled.
- **FD Set Copy:** Each call to `poll()` requires copying the FD set between user and kernel space.
---
### 10. **epoll() Function Interface**
```c
#include <sys/epoll.h>

int epoll_create(int size); // Create an epoll instance
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event); // Add, modify, or remove a file descriptor
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout); // Wait for events
```
- Uses an efficient red-black tree to track file descriptors.
- Returns only the file descriptors with ready events, avoiding linear scanning.
---
### 11. **epoll() Usage Example**

```c
int epfd = epoll_create(1);
struct epoll_event ev, events[MAX_EVENTS];
ev.events = EPOLLIN;
ev.data.fd = listen_fd;
epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);

while (1) {
    int ready_fds = epoll_wait(epfd, events, MAX_EVENTS, -1);
    
    for (int i = 0; i < ready_fds; i++) {
        if (events[i].data.fd == listen_fd) {
            int new_fd = accept(listen_fd, NULL, NULL);
            ev.events = EPOLLIN;
            ev.data.fd = new_fd;
            epoll_ctl(epfd, EPOLL_CTL_ADD, new_fd, &ev);
        } else if (events[i].events & EPOLLIN) {
            // Handle read event
        }
    }
}
```

---

### 12. **Disadvantages of `epoll()`**
- Only available on Linux, limiting portability.
  
---

### 13. **Difference Between epoll LT and ET Modes**

- **LT (Level Triggered):** Default mode. Notifies the application whenever data is available until it is read.
- **ET (Edge Triggered):** Notifies the application only once when new data arrives. Requires careful management to avoid data loss.

---

### 14. **epoll Applications**
- Used in high-performance network servers, such as **Redis** and **Nginx**, which manage thousands of concurrent connections using `epoll()`.

---

### 15. **Comparison Between `select()`, `poll()`, and `epoll()`**

| Feature        | select          | poll            | epoll        |
|----------------|-----------------|-----------------|--------------|
| Max FDs        | 1024            | Unlimited       | Unlimited    |
| Data Structure | Bitmap           | Array           | Red-black tree |
| Efficiency     | O(n)             | O(n)            | O(1)         |
| FD Set Copying | Each call        | Each call       | Once per FD  |
| Available On   | Most platforms   | Most platforms  | Linux only   |

---

### 16. **Complete Code Example**

```c
#include <sys/epoll.h>
#include <netinet/in.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    int listen_fd, conn_fd, epfd, nfds;
    struct sockaddr_in server_addr;
    struct epoll_event ev, events[10];

    // Socket creation and binding
    listen_fd = socket(AF_INET, SOCK

_STREAM, 0);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    server_addr.sin_addr.s_addr = INADDR_ANY;
    bind(listen_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    listen(listen_fd, 10);

    // Epoll instance creation
    epfd = epoll_create(10);
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);

    while (1) {
        nfds = epoll_wait(epfd, events, 10, -1);
        for (int i = 0; i < nfds; i++) {
            if (events[i].data.fd == listen_fd) {
                conn_fd = accept(listen_fd, NULL, NULL);
                ev.events = EPOLLIN;
                ev.data.fd = conn_fd;
                epoll_ctl(epfd, EPOLL_CTL_ADD, conn_fd, &ev);
            } else if (events[i].events & EPOLLIN) {
                char buf[1024];
                read(events[i].data.fd, buf, sizeof(buf));
                // Handle data
            }
        }
    }
    return 0;
}
```

---

### 17. **Frequently Asked Interview Questions**
1. What is IO multiplexing, and why is it used?
2. What are the differences between `select()`, `poll()`, and `epoll()`?
3. Explain the difference between epoll's LT and ET modes.
4. What is the IO model used in applications like Nginx or Redis?
5. When would you choose `epoll()` over `select()` or `poll()`?