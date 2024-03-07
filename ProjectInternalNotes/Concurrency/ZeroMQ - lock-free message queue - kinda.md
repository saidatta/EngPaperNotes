https://www.codedump.info/post/20190209-zeromq-lockfree-queue/
## Overview
- **Version Analyzed**: 4.3.0
- **Key Features**:
  - Transition from TCP stream-oriented to message-oriented development.
  - Encapsulation of transmission modes for easy application service building.
  - Internal implementation of lock-free message queue for object communication, resembling the actor model.
## Basic Architecture

ZeroMQ operates with multiple IO threads, each having core components:
- **Poller**: Handles event polling using mechanisms like epoll and select.
- **Mailbox**: Receives messages, acting as a message queue for inter-object communication.

The architecture employs a per-thread-per-loop design, with thread communication via message mailboxes. Every object that needs message interaction inherits from `object_t`, utilizing a `command_t` structure for defining commands between IO objects.

## Lock-Free Message Queue Implementation

ZeroMQ models its internal architecture on the actor model, with each actor having a mailbox (`mailbox_t`) responsible for message sending and receiving. The lock-free queue is implemented using `ypipe_t`, which internally utilizes `yqueue_t` for queue functionality, optimizing memory allocation by batch element allocation.

### Key Components

- **yqueue_t**: Manages data chunks, each containing multiple elements, to reduce frequent memory allocations. It operates with pointers for chunk management (`begin_chunk`, `back_chunk`, `end_chunk`) and a `spare_chunk` for recycling chunks.
- **ypipe_t**: Constructs a single-write, single-read lock-free queue atop `yqueue_t`, with pointers `_w`, `_r`, `_f`, and `_c` for managing write, read, flush, and last-refreshed element operations, respectively.
- **mailbox_t**: Uses `ypipe_t` for sending and receiving messages. However, due to the necessity for multiple writers, it employs locking during write operations, contradicting the purely lock-free design for single-reader and single-writer scenarios.

### Operation Flow

1. **Initialization**: Begins with a dummy element to manage pointers correctly.
2. **Write Operations**: Managed using `_w` (write pointer) and `_f` (flush pointer), allowing batch data writing and selective visibility to the reading thread.
3. **Flush Mechanism**: Ensures that written data is available for reading, potentially waking up sleeping reader threads using atomic compare-and-swap (CAS) operations.
4. **Read Operations**: Checks for available data using `_r` (read pointer) and `_c` (last refreshed element pointer), with atomic CAS operations for thread-safe updates.

### Multi-Write Consideration

Although `ypipe_t` provides a lock-free mechanism for single-reader and single-writer scenarios, `mailbox_t` introduces locks for handling multiple writers, deviating from a purely lock-free design. A signaling mechanism (`_signaler`) is employed to wake sleeping reader threads, utilizing a pipe for inter-thread signaling.

## Conclusion

ZeroMQ's lock-free message queue offers an efficient mechanism for inter-object communication, closely aligning with the actor model. However, its lock-free nature is primarily applicable in single-reader and single-writer contexts, with locks introduced for multi-writer scenarios to maintain thread safety. This design choice reflects a pragmatic approach to achieving high-performance messaging while accommodating the realities of multi-threaded applications.