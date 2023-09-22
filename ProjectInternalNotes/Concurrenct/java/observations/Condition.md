The Condition class is an interface in java.util.concurrent.locks package that provides methods for managing threads' behavior when certain conditions are met or not met.

Here is an example of how it is used:

Let's say we have a simple producer-consumer problem with a buffer of fixed length. The producer puts items into the buffer and the consumer removes items from it. The producer should stop producing if the buffer is full and wait for the consumer to remove some items. Similarly, if the buffer is empty, the consumer should wait for the producer to put something into the buffer.

We can solve this problem using Condition:

```java
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Buffer {
    private final Lock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();

    private final Object[] items = new Object[100];
    private int putptr, takeptr, count;

    public void put(Object x) throws InterruptedException {
        lock.lock();
        try {
            while (count == items.length) {
                notFull.await(); // waits for the buffer to be not full
            }
            items[putptr] = x;
            if (++putptr == items.length) putptr = 0;
            ++count;
            notEmpty.signal(); // signals that the buffer is not empty
        } finally {
            lock.unlock();
        }
    }

    public Object take() throws InterruptedException {
        lock.lock();
        try {
            while (count == 0) {
                notEmpty.await(); // waits for the buffer to be not empty
            }
            Object x = items[takeptr];
            if (++takeptr == items.length) takeptr = 0;
            --count;
            notFull.signal(); // signals that the buffer is not full
            return x;
        } finally {
            lock.unlock();
        }
    }
}
```
In the `put` method, the producer checks if the buffer is full. If so, it waits for the `notFull` condition. Once the consumer removes an item, it signals the `notFull` condition and the producer can continue to put items. The `take` method is similar.

A Condition instance is intrinsically bound to a lock. To obtain a Condition instance for a particular Lock instance, use its `newCondition()` method.

Here are the key methods of the Condition interface:

- `await()`: Causes the current thread to wait until it is signaled or interrupted.
- `signal()`: Wakes up one waiting thread.
- `signalAll()`: Wakes up all waiting threads.

Note that calling `await()` releases the held lock and suspends the current thread, allowing other threads to acquire the lock and change the state. The thread remains suspended until it is signalled (with `signal()` or `signalAll()`) or is interrupted.

As per the spurious wakeups, these are wakeups that happen due to a thread being awoken from waiting state even when no thread has signalled. To prevent issues due to spurious wakeups, always use `await()` inside a loop that checks for the condition. 

```java
while (count == 0) {
    notEmpty.await();
}
```
This loop ensures that if a spurious wakeup occurs, the thread re-checks the condition and continues to wait if it is not met. 

Finally, remember that every call to `lock()` should be followed by a call to `unlock()`, typically in a `finally` block to ensure that the lock

 is released even if an exception is thrown. 

```java
lock.lock();
try {
    // perform operations on shared resource
} finally {
    lock.unlock();
}
```

This ensures that the lock is always released and prevents potential deadlock situations.