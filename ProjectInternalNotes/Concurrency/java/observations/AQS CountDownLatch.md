https://juejin.cn/post/7243726862635253819
## Overview of CountDownLatch
CountDownLatch in Java is a synchronization aid that allows one or more threads to wait until a set of operations being performed in other threads completes.
## Structure and Constructor
CountDownLatch uses a single non-negative integer count as the synchronization control. The count is initialized in the constructor, and each call to `countDown()` decrements the count until it reaches zero, at which point all waiting threads are released to execute.
```java
public CountDownLatch(int count) {
    if (count < 0) throw new IllegalArgumentException("count < 0");
    this.sync = new Sync(count);
}
```
## Key Methods
### `countDown()`
Decrements the count of the latch, releasing all waiting threads if the count reaches zero.
```java
public void countDown() {
    sync.releaseShared(1);
}
```
### `await()`
Causes the current thread to wait until the latch has counted down to zero.
```java
public void await() throws InterruptedException {
    sync.acquireSharedInterruptibly(1);
}
```
### `await(long timeout, TimeUnit unit)`
Causes the current thread to wait until the latch has counted down to zero, unless the specified waiting time elapses.
```java
public boolean await(long timeout, TimeUnit unit) throws InterruptedException {
    return sync.tryAcquireSharedNanos(1, unit.toNanos(timeout));
}
```
## Internal Mechanics
### Synchronization Control
The synchronization is managed by an inner class `Sync` which extends AQS. The state represents the current count of the latch.
### `tryReleaseShared(int releases)`
Attempts to decrement the state (count) and returns true if the new state is zero.

```java
protected boolean tryReleaseShared(int releases) {
    for (;;) {
        int c = getState();
        if (c == 0) return false;
        int nextc = c - 1;
        if (compareAndSetState(c, nextc)) return nextc == 0;
    }
}
```
### `tryAcquireShared(int acquires)`
Attempts to acquire the latch. The method returns a positive number if the current state is zero, indicating that the latch is released.
```java
protected int tryAcquireShared(int acquires) {
    return (getState() == 0) ? 1 : -1;
}
```
### `doAcquireSharedNanos(int arg, long nanosTimeout)`
Manages timed waiting for the latch to be released.
## Usage Scenarios
CountDownLatch is particularly useful in scenarios where a task must wait for one or more parallel tasks to complete before proceeding.
## `getCount()`
Returns the current count of the latch.
```java
public long getCount() {
    return sync.getCount();
}
```
## Summary
CountDownLatch provides a versatile mechanism for controlling execution flow in concurrent applications, allowing for precise synchronization among threads based on the completion of tasks. Its implementation in Java's concurrency package leverages the powerful AbstractQueuedSynchronizer framework to efficiently manage thread coordination.