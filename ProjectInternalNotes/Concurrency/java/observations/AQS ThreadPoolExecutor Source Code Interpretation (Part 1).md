https://juejin.cn/post/7244466016000000055
https://juejin.cn/post/7244810003591659578
# Detailed Obsidian Notes on 

## Overview and Core Concepts of ThreadPoolExecutor

ThreadPoolExecutor in Java's concurrent package manages a pool of threads to execute tasks asynchronously. It optimizes thread utilization, balancing the workload among threads to improve application performance.

### Core Parameters Explained

- **corePoolSize**: Specifies the minimum number of threads retained in the pool, even if they are idle. Threads beyond this count are terminated if they remain idle for more than the keepAliveTime.

- **maximumPoolSize**: The ceiling on the number of threads in the pool. The ThreadPoolExecutor does not exceed this number of threads in the pool, ensuring a boundary on resource consumption.

- **keepAliveTime** and **unit**: Determines the time limit for which threads exceeding the core pool size will remain idle before being terminated. The keepAliveTime is coupled with its TimeUnit to provide flexibility in defining this duration.

- **workQueue**: The BlockingQueue used for holding tasks before they are executed by a worker thread. The choice of queue (e.g., LinkedBlockingQueue, SynchronousQueue) influences task handling, especially in terms of task ordering and thread starvation.

- **threadFactory**: An optional factory for creating new threads. By supplying a custom ThreadFactory, you can customize the properties of new threads (e.g., naming, daemon status).

- **handler**: Defines the policy for handling tasks that cannot be executed immediately. The RejectedExecutionHandler can be customized to implement different strategies for tasks that cannot be accommodated by the thread pool or work queue.

### Important Member Variables

- **ctl**: An AtomicInteger that encodes both the thread pool's run state and the worker count within its 32 bits. This compact representation allows atomic updates to the state and count, crucial for thread pool control.

- **allowCoreThreadTimeOut**: A boolean value that, when true, permits core threads to time out after the keepAliveTime, making the thread pool more adaptive to the workload.

- **largestPoolSize**: Records the peak historical size of the pool, offering insights into the maximum concurrency level reached by the application.

- **completedTaskCount**: Accumulates the number of tasks that have been completed, providing a metric for work done by the thread pool.

## Key Methods and Their Workings

### `execute(Runnable command)`

This method is the gateway for submitting tasks to the thread pool. It employs a strategic approach to task execution:

1. **Direct Handoff to Worker Thread**: If the current number of threads is below `corePoolSize`, it attempts to start a new worker thread with the given task. This is the fastest path for task execution when the pool is underutilized.

2. **Queueing the Task**: If adding a new worker thread is not feasible, it tries to queue the task. The success of this operation depends on the capacity and current load of the workQueue.

3. **Expanding the Pool**: If the task cannot be queued (due to a full queue), it attempts to create a new worker thread for the task, provided the number of threads is below `maximumPoolSize`.

4. **Task Rejection**: Failing to accommodate the task in either the queue or by creating a new thread leads to task rejection. The rejection policy defined by the `handler` is then invoked.

### Task Submission Overloads

- **`submit(Runnable task, T result)`**: Submits a task with a predefined result, returning a Future to fetch the result post-execution. It wraps the task in a FutureTask for execution management.

- **`submit(Callable<T> task)`**: Similar to the Runnable submission but leverages a Callable, allowing the task to return a result. The FutureTask encapsulates the Callable, bridging it with the Future mechanism.

### Worker Thread Lifecycle

The `Worker` class plays a pivotal role, acting as a bridge between the ThreadPoolExecutor's management layer and the actual execution of tasks. Each Worker is responsible for:

- **Executing Tasks**: Fetching tasks from the workQueue and running them. It enters a loop, continuously pulling tasks and executing them until no more tasks are available or the pool is shutting down.

- **Thread Management**: Managing its lifecycle, including starting, running, and terminating based on the pool's state and task availability.

- **Handling Interruptions and Shutdowns**: Properly responding to interrupts and pool shutdown commands, ensuring tasks are either completed or appropriately halted.

## Summary

The ThreadPoolExecutor's design encapsulates the complexities of thread management, task scheduling, and execution into a coherent framework. By offering a high degree of configurability and leveraging Java's concurrent utilities, it provides a robust solution for developing concurrent applications that require efficient and manageable execution of asynchronous tasks.

---
# AQS ThreadPoolExecutor Source Code Interpretation (Part 2)
This document continues the interpretation of the `ThreadPoolExecutor` source code, focusing on shutdown mechanisms, task invocation methods, and common thread pools provided by the `Executors` class.
## Shutdown Mechanisms
### `shutdown()`
- **Purpose**: Initiates an orderly shutdown in which previously submitted tasks are executed, but no new tasks will be accepted.
```java
public void shutdown() {
    final ReentrantLock mainLock = this.mainLock;
    mainLock.lock();
    try {
        checkShutdownAccess();
        advanceRunState(SHUTDOWN);
        interruptIdleWorkers();
        onShutdown(); // Hook for ScheduledThreadPoolExecutor
    } finally {
        mainLock.unlock();
    }
    tryTerminate();
}
```
- **Key Operations**:
    - **Locking**: Ensures exclusive access to modify pool state.
    - **State Transition**: Sets the pool state to `SHUTDOWN`.
    - **Worker Interruption**: Interrupts all idle worker threads.
    - **Termination Attempt**: Tries to transition the pool to termination if conditions are met.
### `shutdownNow()`
- **Purpose**: Attempts to stop all actively executing tasks, halts the processing of waiting tasks, and returns a list of the tasks that were awaiting execution.

```java
public List<Runnable> shutdownNow() {
    List<Runnable> tasks;
    final ReentrantLock mainLock = this.mainLock;
    mainLock.lock();
    try {
        checkShutdownAccess();
        advanceRunState(STOP);
        interruptWorkers();
        tasks = drainQueue();
    } finally {
        mainLock.unlock();
    }
    tryTerminate();
    return tasks;
}
```

- **Key Operations**:
    - **Immediate Stop**: Sets the pool state to `STOP`, preventing further task execution.
    - **Worker Interruption**: Interrupts all active workers to stop task execution.
    - **Draining Tasks**: Retrieves and returns the tasks that were queued but not executed.

## Task Invocation Methods

### `invokeAny(Collection<? extends Callable<T>> tasks)`

Executes the given tasks, returning the result of one, if successful, and cancels all other tasks upon success or failure.

```java
public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException {
    try {
        return doInvokeAny(tasks, false, 0);
    } catch (TimeoutException cannotHappen) {
        assert false;
        return null;
    }
}
```

### `invokeAll(Collection<? extends Callable<T>> tasks)`

Executes all given tasks and returns a list of `Future` objects representing the tasks, in the same order they were in the input collection.

```java
public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException {
    if (tasks == null) throw new NullPointerException();
    ArrayList<Future<T>> futures = new ArrayList<>(tasks.size());
    boolean done = false;
    try {
        for (Callable<T> t : tasks) {
            RunnableFuture<T> f = newTaskFor(t);
            futures.add(f);
            execute(f);
        }
        // Wait for completion of all tasks
        for (Future<T> f : futures) {
            if (!f.isDone()) {
                try { f.get(); }
                catch (CancellationException | ExecutionException ignore) { }
            }
        }
        done = true;
        return futures;
    } finally {
        if (!done) futures.forEach(f -> f.cancel(true));
    }
}
```

## Common Thread Pools via Executors Class

### Fixed Thread Pool

Creates a thread pool with a fixed number of threads. Tasks are executed sequentially or concurrently depending on the number of threads.

```java
public static ExecutorService newFixedThreadPool(int nThreads) {
    return new ThreadPoolExecutor(nThreads, nThreads, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>());
}
```

### Cached Thread Pool

Creates a thread pool that creates new threads as needed but reuses previously constructed threads when available.

```java
public static ExecutorService newCachedThreadPool() {
    return new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, TimeUnit.SECONDS, new SynchronousQueue<Runnable>());
}
```

### Single Thread Executor

Ensures that tasks are executed sequentially in the order they are submitted.

```java
public static ExecutorService newSingleThreadExecutor() {
    return new FinalizableDelegatedExecutorService(new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>()));
}
```

### Scheduled Thread Pool

Allows for delayed execution or periodic execution of tasks.

```java
public static ScheduledExecutorService newScheduledThreadPool(int corePoolSize) {
    return new ScheduledThreadPoolExecutor(corePoolSize);
}
```

### Work Stealing Pool

Utilizes multiple threads to process a collection of tasks, potentially in parallel

, by employing a work-stealing algorithm.

```java
public static ExecutorService newWorkStealingPool(int parallelism) {
    return new ForkJoinPool(parallelism, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
}
```

## Considerations and Best Practices

- **Customization over Convenience**: While `Executors` provides convenient methods to create thread pools, customizing `ThreadPoolExecutor` directly offers greater control over configurations, addressing specific performance and resource management needs.

- **Resource Management**: Understanding the task characteristics (CPU-bound vs. IO-bound) and system resources is crucial in determining the optimal size and type of thread pool to prevent resource exhaustion or underutilization.

- **Graceful Shutdown**: Proper shutdown of the thread pool is essential to ensure that all tasks are completed and resources are released appropriately.

This document provides a comprehensive overview of ThreadPoolExecutor's key functionalities, including task execution strategies, shutdown mechanisms, and common thread pool configurations, enabling software engineers to leverage concurrent programming effectively in Java applications.