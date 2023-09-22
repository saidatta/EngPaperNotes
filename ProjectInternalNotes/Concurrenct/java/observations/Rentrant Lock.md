
# Java Concurrency: ReentrantLock - Fairness, TryLock and More

## Overview
This tutorial explains how Java's `ReentrantLock` works. The `ReentrantLock` class is a part of the java.util.concurrent.locks package and it's one of the central classes in the Java Concurrency Utilities framework. It provides a basic, yet very flexible locking mechanism.

## The Concept of Lock

Consider a scenario where we have an application that allows users to book a seat in a movie theater. The application is multi-threaded, allowing multiple users, each having a single thread, to access the application simultaneously. 

Now, if two threads try to access and book the same seat, we have a problem. One solution to this is to allow only one thread to book a seat at a time. For this, we introduce the concept of a `lock`.

Let's say we have four threads. All four of them are trying to book a seat simultaneously. When they all attempt to acquire the lock, only one thread will get the lock at a time, and only the owner of the lock can access the seat chart.

## ReentrantLock in Code
Here's an example of how a ReentrantLock can be used in Java:

```java
import java.util.concurrent.locks.ReentrantLock;

public class BookSeat {
    ReentrantLock lock = new ReentrantLock();

    public void book() {
        lock.lock();
        try {
            // Booking code here
        } finally {
            lock.unlock();
        }
    }
}
```

This code defines a `ReentrantLock` object. When a thread enters the `book` method, it acquires the lock with `lock.lock()`. After the booking code is executed, the thread releases the lock with `lock.unlock()`. If another thread tries to enter this method while the lock is held, it will be blocked until the lock is released.

## Reentrant Nature of ReentrantLock
The name `ReentrantLock` stems from its reentrant nature, which allows a thread to acquire the lock multiple times without getting blocked. This becomes especially handy when you have a recursive function that needs to acquire the lock.

## Fairness of ReentrantLock
When constructing a `ReentrantLock`, you can pass a boolean value to decide whether the lock should be "fair". If set to `true`, the lock is "fair" and when multiple threads are waiting for the lock, the thread that's been waiting longest will acquire the lock first. If set to `false`, the lock is "unfair" and there's no access order.

## TryLock
The `ReentrantLock` class also provides the `tryLock` method, which attempts to acquire the lock and returns a boolean indicating whether or not the acquisition was successful. This can be useful when you want a thread to do something else if it can't acquire the lock.

```java
public void book() {
    if (lock.tryLock()) {
        try {
            // Booking code here
        } finally {
            lock.unlock();
        }
    } else {
        // Do something else
    }
}
```

There's also a variant of `tryLock` that accepts a timeout parameter, defining how long the thread should try to acquire the lock before giving up.

## Conclusion
`ReentrantLock` offers a more flexible alternative to the intrinsic synchronization of Java objects. It provides additional capabilities such as fairness, ability to check the lock status, and ability to try to acquire the lock without being blocked. It is an essential tool for writing concurrent applications in Java. 

**Remember:** Always unlock in a `finally` block, because if the code guarded by the lock throws an exception, the `finally` block ensures that the lock is