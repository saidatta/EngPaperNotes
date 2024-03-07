https://juejin.cn/post/7243203130154172473
## Overview of Semaphore

Semaphore in Java is a synchronization aid that controls access to a resource by multiple threads. It manages a set of permits to allow threads to access resources. If a permit is available, a thread acquires it; if not, it may wait until one becomes available.
## Constructor and Structure
Semaphore allows the creation of fair and unfair locks, with unfair being the default.
```java
public Semaphore(int permits) {
    sync = new NonfairSync(permits);
}

public Semaphore(int permits, boolean fair) {
    sync = fair ? new FairSync(permits) : new NonfairSync(permits);
}
```
- **FairSync** and **NonfairSync** implement fair and unfair locking mechanisms, respectively. The main difference lies in the approach to acquiring shared resources.
## NonfairSync Implementation
NonfairSync tries to acquire a shared license in an unfair manner by attempting to reduce the state (number of permits) atomically.

```java
final int nonfairTryAcquireShared(int acquires) {
    for (;;) {
        int available = getState();
        int remaining = available - acquires;
        if (remaining < 0 || compareAndSetState(available, remaining))
            return remaining;
    }
}
```
## FairSync Implementation
FairSync extends the approach to include checking for queued predecessors, making it behave in a fair manner by respecting the order of threads waiting for permits.
```java
protected int tryAcquireShared(int acquires) {
    for (;;) {
        if (hasQueuedPredecessors())
            return -1;
        int available = getState();
        int remaining = available - acquires;
        if (remaining < 0 || compareAndSetState(available, remaining))
            return remaining;
    }
}
```
## Acquiring Permits
The primary method to acquire a permit is `acquire()`, which internally uses `sync.acquireSharedInterruptibly(1)`, blocking until a permit is available or the thread is interrupted.
```java
public void acquire() throws InterruptedException {
    sync.acquireSharedInterruptibly(1);
}
```
## Release of Permits
Releasing permits increases the number of available permits and potentially releases any waiting threads if permits become available.
```java
public void release() {
    sync.releaseShared(1);
}

public void release(int permits) {
    if (permits < 0) throw new IllegalArgumentException();
    sync.releaseShared(permits);
}
```
## Key Methods in AQS for Semaphore
- **tryAcquireShared(int arg)**: Tries to acquire the given number of permits.
- **tryReleaseShared(int arg)**: Releases the given number of permits, returning true if the release was successful.
- **doReleaseShared()**: Propagates the release to other threads in the queue.
- **addWaiter(Node mode)**: Adds a new waiting node in the AQS queue.
- **setHeadAndPropagate(Node node, int propagate)**: Sets the head of the queue and propagates conditions to other nodes as necessary.
## Semaphore in Practice
Semaphores are versatile tools used for controlling access to resources. They can be used for:
- **Limiting concurrent access** to a resource.
- **Implementing producer-consumer patterns** where the semaphore count reflects the number of available items or spaces.
- **Signal handling** between threads where a semaphore can act as a signal for the availability of data or the completion of tasks.
## Conclusion
The Semaphore class, backed by the AbstractQueuedSynchronizer (AQS), provides a flexible mechanism for rate-limiting and controlling access to shared resources. Understanding its implementation and usage patterns is crucial for developing concurrent applications in Java that require precise control over thread execution and resource access.