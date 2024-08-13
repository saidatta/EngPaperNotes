#### Overview
Redis is renowned for its speed due to several key factors:
- **In-memory storage:** All data is stored in memory, allowing for fast data read/write operations.
- **Simple data structures:** Redis offers efficient data structures (strings, lists, sets, hash tables, and sorted sets) with low time complexity.
- **Single-threaded model:** Avoids context switching and concurrency issues, ensuring atomicity and consistency.
- **I/O multiplexing:** Monitors multiple sockets simultaneously, notifying Redis when they are ready for I/O operations.
- **Optimized protocol:** Simple and compact, reducing network data transmission.
- **Additional features:** Persistence, publish/subscribe, transactions, data eviction strategies, master-slave replication, and sharding.
#### I/O Multiplexing in Redis
I/O multiplexing allows a single thread to monitor multiple file descriptors (typically sockets) and notify the program when they are ready for I/O operations. Common I/O multiplexing techniques include `select`, `poll`, `epoll` (Linux), and `kqueue` (BSD).

Redis uses the most appropriate I/O multiplexing API depending on the operating system:
- **Linux:** `epoll`
- **macOS:** `kqueue`

#### Epoll in Redis
Redis uses `epoll` on Linux systems due to its efficiency and scalability.

##### Initialization of Event Handler
In `ae.c`, Redis initializes the event handler and sets up the corresponding multiplexing library.
![[Screenshot 2024-06-24 at 4.36.21 PM.png]]

In `ae_epoll.c`, Redis implements the encapsulation of `epoll`.
![[Screenshot 2024-06-24 at 4.36.42 PM.png]]
```c
static int aeApiCreate(aeEventLoop *eventLoop) {
    aeApiState *state = zmalloc(sizeof(aeApiState));
    if (!state) return -1;

    state->epfd = epoll_create(1024); /* 1024 is just a hint for the kernel */
    if (state->epfd == -1) return -1;

    state->events = zmalloc(sizeof(struct epoll_event)*eventLoop->setsize);
    if (!state->events) return -1;
    ...
    eventLoop->apidata = state;
    return 0;
}
```

##### Registration of Events
![[Screenshot 2024-06-24 at 4.37.04 PM.png]]
```c
static int aeApiAddEvent(aeEventLoop *eventLoop, int fd, int mask) {
    struct epoll_event ee = {0}; /* avoid valgrind warning */
    int op = eventLoop->events[fd].mask == AE_NONE ?
             EPOLL_CTL_ADD : EPOLL_CTL_MOD;

    ee.events = 0;
    mask |= eventLoop->events[fd].mask; /* Merge old events */
    if (mask & AE_READABLE) ee.events |= EPOLLIN;
    if (mask & AE_WRITABLE) ee.events |= EPOLLOUT;
    ee.data.fd = fd;
    if (epoll_ctl(state->epfd, op, fd, &ee) == -1) return -1;
    return 0;
}
```

##### Event Loop
![[Screenshot 2024-06-24 at 4.37.13 PM.png]]
```c
void aeMain(aeEventLoop *eventLoop) {
    eventLoop->stop = 0;
    while (!eventLoop->stop) {
        if (eventLoop->beforesleep != NULL)
            eventLoop->beforesleep(eventLoop);
        aeProcessEvents(eventLoop, AE_ALL_EVENTS);
    }
}
```

##### Handling Epoll Events
![[Screenshot 2024-06-24 at 4.37.22 PM.png]]
```c
static int aeApiPoll(aeEventLoop *eventLoop, struct timeval *tvp) {
    aeApiState *state = eventLoop->apidata;
    int retval, numevents = 0;

    retval = epoll_wait(state->epfd, state->events, eventLoop->setsize,
        tvp ? (tvp->tv_sec*1000 + tvp->tv_usec/1000) : -1);
    if (retval > 0) {
        int j;

        numevents = retval;
        for (j = 0; j < numevents; j++) {
            int mask = 0;
            struct epoll_event *e = state->events+j;

            if (e->events & EPOLLIN) mask |= AE_READABLE;
            if (e->events & EPOLLOUT) mask |= AE_WRITABLE;
            if (e->events & EPOLLERR) mask |= AE_WRITABLE;
            if (e->events & EPOLLHUP) mask |= AE_WRITABLE;
            eventLoop->fired[j].fd = e->data.fd;
            eventLoop->fired[j].mask = mask;
        }
    }
    return numevents;
}
```
##### Handling File Events
![[Screenshot 2024-06-24 at 4.37.35 PM.png]]

#### Summary
Redis's I/O multiplexing technology is key to its ability to efficiently handle a large number of concurrent connections. By abstracting different I/O multiplexing APIs and encapsulating them into a unified event processing interface, Redis can run on different operating systems and maintain high performance.

### Ascii Visualizations

#### Workflow of Epoll Initialization and Event Loop

```
Initialization:
+--------------------------+
| aeCreateEventLoop        |
|  +---------------------+ |
|  | aeApiCreate         | |
|  |  +----------------+ | |
|  |  | epoll_create   | | |
|  |  +----------------+ | |
|  |  | zmalloc        | | |
|  |  +----------------+ | |
|  +---------------------+ |
+--------------------------+

Registration:
+--------------------------+
| aeApiAddEvent            |
|  +---------------------+ |
|  | epoll_ctl           | |
|  +---------------------+ |
+--------------------------+

Event Loop:
+--------------------------+
| aeMain                   |
|  +---------------------+ |
|  | aeProcessEvents     | |
|  |  +----------------+ | |
|  |  | aeApiPoll      | | |
|  |  |  +----------+ | | |
|  |  |  | epoll_wait| | | |
|  |  |  +----------+ | | |
|  |  |  | Process   | | | |
|  |  |  | Events    | | | |
|  |  +----------------+ | |
+--------------------------+

Handling Events:
+--------------------------+
| aeProcessEvents          |
|  +---------------------+ |
|  | Iterate Events      | |
|  |  +----------------+ | |
|  |  | rfileProc      | | |
|  |  | wfileProc      | | |
|  |  +----------------+ | |
+--------------------------+
```

### Key Takeaways
- **epoll** is used by Redis on Linux for I/O multiplexing, allowing efficient management of multiple connections.
- **Initialization:** Redis creates an `epoll` instance and allocates memory for event arrays.
- **Registration:** Events are registered with `epoll` using `epoll_ctl`.
- **Event Loop:** The event loop continuously processes events, waiting for I/O operations using `epoll_wait`.
- **Handling Events:** Ready file descriptors are processed by invoking the corresponding read/write event handlers.

This approach enables Redis to handle a large number of concurrent connections efficiently, contributing significantly to its high performance.