https://juejin.cn/post/7243393786469564473

ReentrantLock in Java is a reentrant mutex that leverages the AbstractQueuedSynchronizer (AQS) for implementing synchronization mechanisms. It supports fair and unfair locking mechanisms.
## UML Architecture Overview

The UML diagram illustrates the internal structure of ReentrantLock, showcasing its relationship with the AQS framework.
![[Screenshot 2024-02-09 at 3.38.26 PM.png]]
## Constructors
ReentrantLock provides constructors to specify the locking strategy:
```java
public ReentrantLock() {
    sync = new NonfairSync();
}

public ReentrantLock(boolean fair) {
    sync = fair ? new FairSync() : new NonfairSync();
}
```
- **FairSync**: Ensures that threads acquire the lock in a FIFO manner.
- **NonfairSync**: Does not guarantee any specific order in which threads acquire the lock.
## Lock Acquisition
### `lock()`
The `lock()` method is central to acquiring a lock:
```java
public void lock() {
    sync.lock();
}
```
#### Unfair Lock Implementation
Unfair lock quickly tries to set the lock state and set the current thread as the exclusive owner if the state is zero.

```java
final void lock() {
    if (compareAndSetState(0, 1))
        setExclusiveOwnerThread(Thread.currentThread());
    else
        acquire(1);
}
```
#### Fair Lock Implementation
Fair lock uses `acquire(1)` directly but differs in its `tryAcquire` method by checking for queued predecessors.
```java
final void lock() {
    acquire(1);
}
```
### Lock Acquisition Methods
- **`lockInterruptibly()`**: Acquires the lock unless the thread is interrupted.
- **`tryLock()`**: Non-blocking attempt to acquire the lock.
- **`tryLock(long timeout, TimeUnit unit)`**: Attempts to acquire the lock within the given waiting time and can be interrupted.
## Lock Release
Unlocking is performed by calling:
```java
public void unlock() {
    sync.release(1);
}
```
This involves setting the lock state to zero and potentially waking up queued threads.
## Condition Variables
ReentrantLock also supports condition variables through the `Condition` interface, providing methods for thread coordination.
### Waiting on Conditions
- **`await()`**: Causes the current thread to wait until it is signalled or interrupted.
- **`awaitUninterruptibly()`**: Waits without responding to interrupts.
- **`await(long time, TimeUnit unit)`**: Waits up to a specified waiting time for a signal.
- **`awaitUntil(Date deadline)`**: Waits until the specified deadline for a signal.
### Signalling Conditions
- **`signal()`**: Wakes up one waiting thread.
- **`signalAll()`**: Wakes up all waiting threads.
## Key Takeaways
- **ReentrantLock** provides a flexible locking mechanism with support for both fair and unfair locking strategies.
- **Condition variables** allow threads to coordinate access to shared resources more effectively.
- The implementation relies heavily on the **AbstractQueuedSynchronizer (AQS)** framework, which provides a foundation for developing synchronization primitives.
- Understanding **ReentrantLock** and its source code is crucial for developing concurrent applications that require complex synchronization among threads.