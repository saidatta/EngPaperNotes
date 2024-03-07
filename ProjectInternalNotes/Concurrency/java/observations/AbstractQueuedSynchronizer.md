AbstractQueuedSynchronizer (AQS) is a framework provided by Java for building locks and other synchronization mechanisms. It is the foundation for several concurrency utilities such as ReentrantLock, Semaphores, CountDownLatch, and others.

### Core Idea of AQS

The central concept of AQS is to manage a queue of threads trying to acquire a resource. When a thread requests a resource and it is available, AQS sets that thread as the active worker and locks the resource. If the resource is busy, AQS manages the threads by putting them into a waiting queue until the resource becomes available.

### Low-Level Implementation

AQS uses a volatile integer state to represent the synchronization state and relies on a built-in First-In-First-Out (FIFO) queue to manage the threads. It uses Compare-And-Set (CAS) operations to change the state safely without using traditional locking mechanisms.

### CLH (Craig, Landin, and Hagersten) Queue

AQS uses a variant of the CLH queue, which is a conceptual two-way queue, to implement its locking mechanism. Each thread that requests the shared resource is encapsulated into a node and managed in this queue.
![[Screenshot 2024-02-09 at 2.38.34 PM.png]]
### Node and Thread Modes
A node in AQS represents a thread in the CLH queue. Threads can be in two modes:

- **SHARED**: The thread is waiting for the lock in shared mode.
- **EXCLUSIVE**: The thread is waiting exclusively for the lock.

### waitStatus
The `waitStatus` of a node indicates the thread's state:
- **0**: Default value upon initialization.
- **CANCELLED (1)**: The thread's attempt to acquire the lock has been canceled.
- **CONDITION (-2)**: The node is in a waiting queue, and the thread is waiting for a signal to wake up.
- **PROPAGATE (-3)**: This status is used when the thread is in SHARED mode.
- **SIGNAL (-1)**: The thread is ready and waiting for the resource to be released.
![[Screenshot 2024-02-09 at 2.36.01 PM.png]]
### Sync State
The synchronization state is managed by a volatile integer `state`. AQS provides methods to get, set, and update this state atomically.
![[Screenshot 2024-02-09 at 2.36.35 PM.png]]
### ConditionObject
AQS also provides a `ConditionObject` which is a condition queue used for thread coordination. Threads can wait on a `ConditionObject`, and they can be signaled to wake up and attempt to acquire the resource again.
![[Screenshot 2024-02-09 at 2.51.38 PM.png]]
### Overridable Methods
When creating custom synchronizers using AQS, developers typically only need to override a small subset of methods provided by AQS to manage the state of the resource.
![[Screenshot 2024-02-09 at 2.51.01 PM.png]]
### AQS Implementations in JDK
AQS is used by several concurrency utilities in the JDK:
- **ReentrantLock**: Uses AQS to manage lock state.
- **Semaphore**: Uses AQS to manage permits.
- **CountDownLatch**: Uses AQS to manage a countdown.
- **ThreadPoolExecutor**: Uses AQS to manage worker threads.
- **CyclicBarrier**: Uses AQS to manage a barrier that threads must reach before proceeding.
- **ReentrantReadWriteLock**: Uses AQS to manage read and write access to a resource.

AQS is a powerful tool for implementing custom synchronization utilities, providing a robust framework for managing thread coordination and resource locking.

Citations:
[1] https://www.baeldung.com/java-volatile
[2] https://www.geeksforgeeks.org/volatile-keyword-in-java/
[3] https://inspectionsystems.co.nz/the-average-quantity-system-aqs-explained/
[4] https://en.wikipedia.org/wiki/Volatile_(computer_programming)
[5] https://www.epa.gov/aqs