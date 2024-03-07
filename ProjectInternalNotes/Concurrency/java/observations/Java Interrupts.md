In this video, we learned about Java interrupts, a cooperative mechanism used to indicate to a running thread that it should stop its current operation. This mechanism allows one thread to request another to stop what it's doing. However, it doesn't forcefully stop the thread; it's up to the target thread to respond to the interrupt.

## What is Java Interrupt?

Imagine we have a main thread (Thread 1) that starts a task thread (Thread 2) to perform a long-running operation. At some point, Thread 1 decides it no longer needs Thread 2 to complete its task. Unlike some other programming languages, Java's Thread class doesn't have a `cancel()` method. This is where the interrupt mechanism comes in.

Thread 1 can call the `interrupt()` method on Thread 2. The `interrupt()` method sets an interrupt flag in Thread 2, essentially asking it to stop executing. How Thread 2 responds to this request depends on how it's coded. It can ignore the interrupt entirely, or it can check its interrupt status periodically and stop executing when it's interrupted.

```java
// Main thread
Thread taskThread = new Thread(new Task());
taskThread.start();

// ... sometime later ...
taskThread.interrupt();
```

On the task thread side:

```java
// Task thread
for (int i = 0; i < 10000; i++) {
    process(i);

    if (Thread.currentThread().isInterrupted()) {
        // The interrupt flag was set, stop execution.
        break;
    }
}
```

If Thread 2 is performing an operation in a loop and checks for an interrupt flag at each iteration, it can stop the operation early, saving system resources.

## Using Thread.interrupted()

The `Thread.interrupted()` method both returns the value of the interrupt flag and resets the flag to false. This method can be used to ensure the flag is reset when the thread is interrupted.

```java
// Task thread
for (int i = 0; i < 10000; i++) {
    process(i);

    if (Thread.interrupted()) {
        // The interrupt flag was set, stop execution.
        break;
    }
}
```

## Interrupts and Exceptions

Sometimes you may want to throw an `InterruptedException` in response to an interrupt. This checked exception signals that the thread has stopped its operation because of an interrupt. It helps differentiate between a thread that has stopped normally and a thread that has been interrupted.

However, `Runnable` doesn't allow checked exceptions to be thrown. If you need to throw an `InterruptedException`, consider using a `Callable` instead.

```java
// Using Callable to throw InterruptedException
public class Task implements Callable<Void> {
    @Override
    public Void call() throws InterruptedException {
        for (int i = 0; i < 10000; i++) {
            process(i);

            if (Thread.interrupted()) {
                throw new InterruptedException();
            }
        }

        return null;
    }
}
```

## JVM-Provided Interrupt Checks

The Java Virtual Machine (JVM) checks the interrupt flag in some cases and throws an `InterruptedException` if the flag is set. This occurs when calling:

- `Object.wait()`
- `Thread.sleep()`
- `Thread.join()`

When calling these methods, you need to handle the possible `InterruptedException` that can be thrown.

```java
try {
    Thread.sleep(2000);
} catch (InterruptedException e) {
    System.out.println("Interrupted while sleeping!");
    return;
}
```

## Checking the Interrupted Status

In Java, there are two methods to check the interrupted status:

1. `Thread.isInterrupted()`: Returns the

## Java Concurrency: What are Java Interrupts

Java interrupts provide a mechanism for one thread to signal another thread to terminate its execution, such as when a task or operation is no longer required. It's crucial to understand that interrupts in Java are a cooperative mechanism. One thread cannot force another to stop what it is doing. Instead, the interrupt mechanism signals the other thread, and it's up to the receiving thread to handle the interrupt signal as it sees fit. 

### Java Thread Example

To illustrate, let's imagine you have a main thread, "Thread 1", that starts another thread, "Thread 2", to perform a long-running operation. However, for some reason, Thread 1 decides it no longer needs Thread 2 to complete the operation.

```java
Runnable task = new Runnable() {
    @Override
    public void run() {
        for (int i = 0; i < 10000; i++) {
            process(i);
            if (Thread.currentThread().isInterrupted()) {
                // Clean up and terminate
                return;
            }
        }
    }

    private void process(int i) {
        // long-running operation
    }
};

Thread taskThread = new Thread(task);
taskThread.start();
// Later...
taskThread.interrupt();
```

In this code example, Thread 1 creates a new task and then starts a new thread with this task. If Thread 1 later decides to interrupt Thread 2, it would call `taskThread.interrupt()`. 

On Thread 2's side, it has a responsibility to periodically check whether an interrupt has been issued using the `Thread.currentThread().isInterrupted()` method. If this method returns `true`, it means an interrupt signal has been sent, and the thread should clean up and terminate its operation. 

### `InterruptedException`

Java also provides a mechanism to throw an `InterruptedException` to indicate that a thread that is in the waiting, sleeping, or otherwise occupied state has been interrupted. 

Here's an example of how you might see it used:

```java
try {
    Thread.sleep(2000);
} catch (InterruptedException e) {
    System.out.println("Interrupt triggered while I was asleep");
    return;
}
```

In this example, the current thread is put to sleep for two seconds using `Thread.sleep(2000)`. If an interrupt is issued during this sleep period, the thread will catch the `InterruptedException` and then handle it accordingly. In this case, it prints a message and returns, effectively ending the thread's operation.

### `Thread.isInterrupted()` vs `Thread.interrupted()`

There are two methods you can use to check the interrupt status:

1. `Thread.isInterrupted()`: This method returns a `boolean` indicating whether the current thread has been interrupted.

2. `Thread.interrupted()`: This method checks if the current thread has been interrupted and then clears the interrupt status.

Use `Thread.interrupted()` when you want to check the interrupt status and also reset the status in one go. It's generally recommended to use this when throwing an `InterruptedException`:

```java
if (Thread.interrupted()) {
    throw new InterruptedException();
}
```

In this example, the thread checks for an interrupt and resets the interrupt status. If an interrupt is detected, it throws an `InterruptedException`.

### Conclusion

Interrupts in Java provide a mechanism to indicate that a thread should stop its operation. They are cooperative, meaning the thread receiving the interrupt signal has the discretion to handle it. Interrupts are typically used to prevent a thread from running unnecessary operations, hence optimizing resource usage.

