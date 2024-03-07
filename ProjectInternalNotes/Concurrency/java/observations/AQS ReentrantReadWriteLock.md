https://juejin.cn/post/7245206582002712634
https://juejin.cn/post/7245275769220104252
## Overview

The `ReentrantReadWriteLock` in Java provides a pair of read and write locks that enable multiple threads to safely access a shared resource. It is designed to offer higher concurrency in scenarios where reads outnumber write operations. The lock supports reentrancy, fairness policies, and lock downgrading.

## Constructors

`ReentrantReadWriteLock` can be instantiated with or without specifying fairness:

```java
public ReentrantReadWriteLock() {
    this(false);
}

public ReentrantReadWriteLock(boolean fair) {
    sync = fair ? new FairSync() : new NonfairSync();
    readerLock = new ReadLock(this);
    writerLock = new WriteLock(this);
}
```

- **Fairness**: Determines whether the longest-waiting thread is given preference in acquiring the lock.

## Internal Workings

### Sync Class

The `Sync` class, extending `AbstractQueuedSynchronizer (AQS)`, underlies the lock mechanism, dividing lock state into two halves:

- **High 16 bits**: Count of read locks held.
- **Low 16 bits**: Count of write locks held.

This division allows for effective read-write separation.

### ReadLock and WriteLock

`ReentrantReadWriteLock` provides two inner classes, `ReadLock` and `WriteLock`, to represent the read and write locks, respectively.

### Lock Acquisition

#### Read Lock

- **`lock()`**: Acquires the read lock if the write lock isn't held by another thread.
- **`lockInterruptibly()`**: Acquires the read lock unless the thread is interrupted.
- **`tryLock()`**: Non-blocking attempt to acquire the read lock.
- **`tryLock(long timeout, TimeUnit unit)`**: Attempts to acquire the read lock within the given waiting time.

#### Write Lock

- **`lock()`**: Acquires the write lock, blocking if necessary.
- **`lockInterruptibly()`**: Acquires the write lock unless the thread is interrupted.
- **`tryLock()`**: Non-blocking attempt to acquire the write lock.
- **`tryLock(long timeout, TimeUnit unit)`**: Attempts to acquire the write lock within the given waiting time.

### Unlocking

- **Read Lock**: `unlock()` method decrements the read lock hold count.
- **Write Lock**: `unlock()` method decrements the write lock hold count and may release the lock entirely if it's the final release.

### Conditions

- **Write Lock**: Supports creating `Condition` objects for coordination.

### Utilities

- **`isHeldByCurrentThread()`**: Checks if the current thread holds the write lock.
- **`getHoldCount()`**: Returns the number of recursive holds on the write lock by the current thread.

## Key Points

- **Reentrancy**: Both read and write locks support reentrancy.
- **Lock Downgrading**: Transitioning from a write lock to a read lock without fully releasing the write lock is supported. However, upgrading from a read lock to a write lock is not.
- **Fairness**: Optional fairness policy can be applied, influencing the order in which threads acquire locks.
- **Performance Considerations**: While `ReentrantReadWriteLock` can improve performance in read-heavy scenarios, the complexity of managing lock states can impact performance under high contention.

## Usage

`ReentrantReadWriteLock` is ideal for scenarios with high read frequency but low write frequency, enabling multiple readers while ensuring exclusive access for writers. Proper usage involves carefully managing lock acquisitions and releases to avoid deadlocks and ensure thread safety.

In conclusion, `ReentrantReadWriteLock` is a powerful tool for controlling access to shared resources in a multi-threaded environment, providing both flexibility and performance improvements when used appropriately.

---
# Detailed Notes on ReentrantReadWriteLock Source Code Interpretation (2)

## Overview

The second part of the `ReentrantReadWriteLock` source code interpretation focuses on the fair lock mechanism and additional utility methods provided by the lock. Fair locks ensure that threads acquire locks in the order they requested them, preventing starvation.

## FairSync Class

`FairSync` class extends `Sync` to implement fair locking policy for both read and write locks. The key to fair locking is the `hasQueuedPredecessors()` method, which checks if there are any threads waiting longer than the current thread.

```java
static final class FairSync extends Sync {
    final boolean writerShouldBlock() {
        return hasQueuedPredecessors();
    }
    final boolean readerShouldBlock() {
        return hasQueuedPredecessors();
    }
}
```

## Acquiring Locks with FairSync

- **Read Lock (`ReadLock`)**:
    - `lock()`, `lockInterruptibly()`, and `tryLock()` methods internally check if the current thread should block by calling `readerShouldBlock()`, which utilizes `hasQueuedPredecessors()` to enforce fairness.
    
- **Write Lock (`WriteLock`)**:
    - Similar to read lock, write lock acquisition methods (`lock()`, `lockInterruptibly()`, and `tryLock()`) use `writerShouldBlock()`, ensuring that write lock requests are granted in a fair manner.

## Utility Methods

### `getReadHoldCount()`

Returns the number of read locks held by the current thread. This method is specific to the current thread and ignores the read locks held by others.

```java
public int getReadHoldCount() {
    return sync.getReadHoldCount();
}
```

### `getReadLockCount()`

Returns the total number of read locks acquired on the lock object, providing a measure of read lock usage across all threads.

```java
public int getReadLockCount() {
    return sync.getReadLockCount();
}
```

### `getWriteHoldCount()`

Returns the number of write locks held by the current thread. Useful for checking if the current thread holds a write lock and how many times it has been re-entrantly acquired.

```java
public int getWriteHoldCount() {
    return sync.getWriteHoldCount();
}
```

## Practical Use Case: Read-Write Lock Cache

The following example demonstrates a simple cache mechanism protected by a `ReentrantReadWriteLock`, allowing multiple threads to read from the cache simultaneously, but ensuring exclusive access for writes.

```java
private final Map<String, String> cache = new HashMap<>();
private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock(false);

public void put(String key, String value) {
    lock.writeLock().lock();
    try {
        cache.put(key, value);
    } finally {
        lock.writeLock().unlock();
    }
}

public String get(String key) {
    lock.readLock().lock();
    try {
        return cache.get(key);
    } finally {
        lock.readLock().unlock();
    }
}
```

## Conclusion

The `ReentrantReadWriteLock` in Java offers a powerful mechanism for managing access to shared resources, with support for fairness policies and separate read and write locks. Understanding the internal workings of this lock, including its fair synchronization strategies and utility methods, can help developers effectively utilize this concurrency tool in their applications for improved performance and reliability.