https://juejin.cn/post/7155132111949660190?searchId=202404260537024D3853D0600815493CB9

### Introduction
This note provides a detailed analysis of the evolution from individual threads to thread pools. It includes code examples in Java and explores the underlying principles and benefits of using thread pools.

### From Thread to Thread Pool

#### What is a Thread Doing?
A thread is responsible for executing a function and then terminating. Let's start with a basic example in Java to illustrate this concept:

```java
public class Demo01 {

    public static void main(String[] args) {
        var thread = new Thread(() -> {
            System.out.println("Hello world from a Java thread");
        });
        thread.start();
    }
}
```

This code snippet creates and starts a thread that prints a message. The same can be achieved using an anonymous inner class:

```java
public class Demo01 {

    public static void main(String[] args) {
        var thread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Hello world from a Java thread");
            }
        });
        thread.start();
    }
}
```

In both cases, the thread executes the `run` method of a `Runnable` object. The Java thread constructor is as follows:

```java
public Thread(Runnable target) {
    this(null, target, "Thread-" + nextThreadNum(), 0);
}
```

When `start()` is called, it triggers the native `start0` method, which in turn calls the operating system's API to create a thread that executes the `run` method's bytecode and then exits.

![[Screenshot 2024-05-08 at 1.32.06 PM.png]]
![[Screenshot 2024-05-08 at 1.32.15 PM.png]]
#### The Lifecycle of a Thread
- **Creation**: A thread is created using the system's underlying functions.
- **Execution**: The thread executes the function provided to it.
- **Termination**: After executing the function, the thread exits and is recycled by the system.
In essence, a thread's role is to execute a function and then terminate.
### Why Do We Need a Thread Pool?

Creating and destroying threads frequently can be resource-intensive due to the need for system resources and context switching between user mode and kernel mode. This overhead can significantly impact performance, especially under high concurrency.

A thread pool addresses this issue by reusing a fixed number of threads to execute multiple tasks. This avoids the overhead of creating and destroying threads repeatedly.
### Thread Pool Implementation Principle
#### Key Points
1. **A thread executes a function**.
2. **Threads in the thread pool can execute many functions but will not exit**.

These principles can be achieved by using a loop within the thread to continuously fetch and execute tasks from a task queue until instructed to stop.
![[Screenshot 2024-05-08 at 1.33.39 PM.png]]
![[Screenshot 2024-05-08 at 1.34.34 PM.png]]
#### Sample Code
The following code snippet demonstrates the basic structure of a thread pool thread:

```java
public class ThreadPoolThread extends Thread {
    private BlockingQueue<Runnable> taskQueue;
    private boolean isStopped = false;

    public ThreadPoolThread(BlockingQueue<Runnable> queue) {
        taskQueue = queue;
    }

    public void run() {
        while (!isStopped) {
            try {
                Runnable task = taskQueue.take();
                task.run();
            } catch (InterruptedException e) {
                // do nothing
            }
        }
    }

    public synchronized void stopThread() {
        isStopped = true;
        this.interrupt(); // Break the thread out of the blocking queue wait.
    }
}
```

This thread continuously takes tasks from a blocking queue and executes them. It stops only when `isStopped` is set to true.

### Details of Thread Pool Implementation

#### Concurrent and Safe Blocking Queue
A thread pool requires a thread-safe queue to hold the tasks. The `BlockingQueue` interface in Java provides a suitable structure.

#### Ensuring All Threads Exit Normally
To ensure that threads exit normally, a flag (e.g., `isStopped`) can be used to indicate when the thread should stop. The thread checks this flag regularly and exits the loop when the flag is set.

### Summary
This note has introduced the evolution from individual threads to thread pools, focusing on the principles behind thread pools. Future notes will delve into the implementation details and further optimizations.

---

### Additional Resources
- Java Concurrency in Practice by Brian Goetz
- The Art of Multiprocessor Programming by Maurice Herlihy and Nir Shavit
- [Java Thread Pool](https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/Executors.html)

### Code Repository
For more examples and detailed implementations, visit the [GitHub Repository](https://github.com/username/repository).

------
![[Screenshot 2024-05-08 at 1.37.36 PM.png]]
## Thread Pool Usage and Analysis

### Preface
In Java, thread pools are a widely used concurrent framework. They are essential for executing tasks asynchronously or concurrently. Proper use of thread pools can provide several benefits:

1. **Reduce Resource Consumption**: By reusing threads, thread creation and destruction costs are minimized.
2. **Improve Response Speed**: Tasks can be executed immediately without waiting for thread creation.
3. **Improve Thread Manageability**: Threads are scarce resources; uncontrolled creation can deplete resources and reduce system stability. A thread pool allows for unified allocation, tuning, and monitoring.

### 1. Implementation Principle of Thread Pool

The processing flow of a thread pool is illustrated as follows:

#### Thread Pool States
The `ctl` field in the thread pool represents its current state. It is an `AtomicInteger` encapsulating two fields: `workerCount` (number of valid threads) and `runState` (whether it is running, shutting down, etc.). The state transitions are:
- **RUNNING**: Accepts new tasks and processes queued tasks.
- **SHUTDOWN**: Does not accept new tasks but processes queued tasks.
- **STOP**: Does not accept new tasks, does not process queued tasks, and interrupts ongoing tasks.
- **TIDYING**: All tasks have terminated, `workerCount` is zero, and the thread transitions to the `terminate()` method.
- **TERMINATED**: `terminate()` method execution is completed.
![[Screenshot 2024-05-08 at 2.08.52 PM.png]]
The state transitions:
- `RUNNING` -> `SHUTDOWN` when `shutdown()` is called.
- `(RUNNING or SHUTDOWN)` -> `STOP` when `shutdownNow()` is called.
- `SHUTDOWN` -> `TIDYING` when both the queue and thread pool are empty.
- `STOP` -> `TIDYING` when the thread pool is empty.
- `TIDYING` -> `TERMINATED` when the `terminate()` method completes.
![[Screenshot 2024-05-08 at 2.08.29 PM.png]]
#### Execution Process
There are four scenarios when using the `execute()` method to submit tasks to the thread pool:

1. **Thread Count < corePoolSize**: A new thread is created to execute the task.
2. **Thread Count ≥ corePoolSize**: The task is added to the blocking queue.
3. **Queue Full**: A new thread is created to process the task (requiring a global lock).
4. **Thread Count Exceeds maximumPoolSize**: The thread pool rejects the task, calling the `RejectedExecutionHandler.rejectedExecution()` method.

#### Source Code Analysis
```java
public void execute(Runnable command) {
    if (command == null)
        throw new NullPointerException();
    int c = ctl.get();
    if (workerCountOf(c) < corePoolSize) {
        if (addWorker(command, true))
            return;
        c = ctl.get();
    }
    if (isRunning(c) && workQueue.offer(command)) {
        int recheck = ctl.get();
        if (!isRunning(recheck) && remove(command))
            reject(command);
        else if (workerCountOf(recheck) == 0)
            addWorker(null, false);
    } else if (!addWorker(command, false))
        reject(command);
}
```

### 2. Creation and Use of Thread Pool

#### Core Parameters
1. **corePoolSize**: Minimum number of threads kept in the pool, even if they are idle.
2. **runnableTaskQueue**: A blocking queue to hold tasks before they are executed.
   - **ArrayBlockingQueue**: Fixed size, FIFO order.
   - **LinkedBlockingQueue**: Variable size, FIFO order.
   - **SynchronousQueue**: Direct handoff, no queue.
   - **PriorityBlockingQueue**: Priority order.
3. **maximumPoolSize**: Maximum number of threads allowed in the pool.
4. **RejectedExecutionHandler**: Strategy for handling tasks that cannot be executed immediately.
   - **AbortPolicy**: Throws an exception.
   - **CallerRunsPolicy**: Executes the task in the caller's thread.
   - **DiscardOldestPolicy**: Discards the oldest task.
   - **DiscardPolicy**: Discards the task silently.
5. **keepAliveTime**: Time for which threads can remain idle before being terminated.
6. **TimeUnit**: Unit of time for keepAliveTime.
7. **ThreadFactory**: Factory to create new threads.

#### Creating a Thread Pool

1. **Using Executors Factory Class**
   - **Single-Threaded Pool**:
     ```java
     public static ExecutorService newSingleThreadExecutor() {
         return new FinalizableDelegatedExecutorService(
             new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>())
         );
     }
     ```
   - **Fixed-Thread Pool**:
     ```java
     public static ExecutorService newFixedThreadPool(int nThreads) {
         return new ThreadPoolExecutor(nThreads, nThreads, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>());
     }
     ```
   - **Cacheable Thread Pool**:
     ```java
     public static ExecutorService newCachedThreadPool() {
         return new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, TimeUnit.SECONDS, new SynchronousQueue<Runnable>());
     }
     ```

2. **Custom Thread Pool Creation**
   ```java
   public ThreadPoolExecutor(int corePoolSize, int maximumPoolSize, long keepAliveTime, TimeUnit unit, BlockingQueue<Runnable> workQueue) {
       this(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, Executors.defaultThreadFactory(), defaultHandler);
   }
   ```

#### Submitting Tasks to the Thread Pool

1. **Using `execute()` Method**: Submits tasks that do not return a value.
   ```java
   public static void main(String[] args) {
       ...
       threadPool.execute(new Runnable {
           public void run() {
               // do something...
           }
       });
       ...
   }
   ```

2. **Using `submit()` Method**: Submits tasks that return a value.
   ```java
   public static void main(String[] args) {
       ...
       Future<Object> future = threadPool.submit(handleTask);
       try {
           Object res = future.get();
       } catch (InterruptedException e) {
           // handle interruption
       } catch (ExecutionException e) {
           // handle execution exception
       } finally {
           threadPool.shutdown();
       }
       ...
   }
   ```

#### Shutting Down the Thread Pool
- **shutdown()**: Initiates an orderly shutdown.
  ```java
  public void shutdown() {
      final ReentrantLock mainLock = this.mainLock;
      mainLock.lock();
      try {
          checkShutdownAccess();
          advanceRunState(SHUTDOWN);
          interruptIdleWorkers();
          onShutdown();
      } finally {
          mainLock.unlock();
      }
      tryTerminate();
  }
  ```

- **shutdownNow()**: Attempts to stop all actively executing tasks and halts the processing of waiting tasks.
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

### 3. Recommended Thread Pool Parameter Settings

#### Task-Based Reference

- **CPU-Bound Tasks**: Number of threads = Number of CPU cores + 1.
- **IO-Bound Tasks**: Number of threads = 2 * Number of CPU cores + 1.

#### Ideal State Evaluation Based on Number of Tasks

1. **Default Values**:
   ```java
   corePoolSize = 1
   queueCapacity = Integer.MAX_VALUE
   maxPoolSize = Integer.MAX_VALUE
   keepAliveTime = 60s
   allowCoreThreadTimeout = false
   rejectedExecutionHandler = AbortPolicy()
   ```

2. **Task-Based Calculation**:
   - **corePoolSize**: Based on task frequency and duration.
     ```java
     corePoolSize = tasksPerSecond * taskDuration
     ```
   - **queueCapacity**:
     ```java
     queueCapacity = (corePoolSize / taskDuration) * maxResponseTime
     ```
   - **maxPoolSize**:
     ```java
     maxPoolSize = (maxTasks - queueCapacity) / taskCapacity
     ```

### 4. Recommended Thread Pool Usage Scenarios

#### Scenario 1: Asynchronous Processing
- **Create Thread Pool**:
  ```java
  @Bean
  public ThreadPoolTaskExecutor asyncExecutorPool() {
      ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
      executor.setCorePoolSize(20);
      executor.setMaxPoolSize(100);
      executor.setQueueCapacity(500);
      executor.setWaitForTasksToCompleteOnShutdown(true);
      executor.setAwaitTerminationSeconds(60);
      executor.setThreadNamePrefix("test-async-thread-");
      executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
      executor.initialize();
      return executor;
  }
  ```

- **Use @Async Annotation**:
  ```java
  @Async("asyncExecutorPool")
  public void processTask1() {
      // do something...
  }
  ```

#### Scenario 2: Ordered Execution
- **Single-Threaded Pool**:
  ```java
  public class SingleExecutorTest {
      private HashMap<Long, ThreadPoolExecutor> executorMap = new HashMap<>();

      public void init

() {
          for (int i = 0; i < 5; i++) {
              ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.SECONDS, new ArrayBlockingQueue<>(1000));
              threadPoolExecutor.setRejectedExecutionHandler(new ThreadPoolExecutor.DiscardPolicy());
              ThreadFactory threadFactory = new CustomizableThreadFactory("testSingle-" + i + "-");
              threadPoolExecutor.setThreadFactory(threadFactory);
              executorMap.put(Long.valueOf(i), threadPoolExecutor);
          }
      }

      public void processTask() {
          ThreadPoolExecutor executor = executorMap.get(Long.valueOf(id % 5));
          executor.submit(() -> {
              // do something...
          });
      }
  }
  ```

### Summary
This document explains the principles of thread pool execution, creation methods, recommended parameter settings, and general usage scenarios. Proper creation and usage of thread pools can reduce resource consumption and improve response speed in Java applications.