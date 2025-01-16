**High-Level Purpose:**
The `AsyncSemaphore` class is a concurrency control mechanism that ensures no more than a specified number of tasks (`maxPermits`) are running at once. When a task is submitted, it doesn’t start immediately if the system is already at its concurrency limit. Instead, it queues the task and starts it only when a "permit" is available (i.e., when other tasks complete). The tasks themselves are asynchronous operations represented by `ListenableFuture`s, so the semaphore integrates smoothly with asynchronous workflows.

**Logical Explanation (For a PhD Engineer):**

1. **Concurrency Limiting (The Core Idea)**:  
   Imagine you have a long list of tasks that need to be executed concurrently, such as fetching data for multiple users from a remote server. If you try to run them all at once, you may overwhelm the server or your system’s resources. The `AsyncSemaphore` ensures that at most `maxPermits` tasks are running at the same time, preventing resource contention or overload.

2. **Queueing Mechanism**:  
   When you submit a task:
   - It is wrapped in a `QueuedTask` and placed into an internal queue (`queuedTasks`).
   - If the current number of active tasks is less than `maxPermits`, the task is dequeued and started immediately.
   - If the maximum concurrency is already reached, the task waits in the queue until another task finishes.

   **Key Point**: The submission does not block; it returns a `ListenableFuture` that completes when the task itself completes. Internally, the semaphore just postpones the actual start of the work until it is allowed to run.

3. **Permits and Counters**:  
   The class uses an `AtomicInteger counter` to keep track of how many tasks are currently running or in the process of starting:
   - **Acquire a Permit**: When you submit a task, the counter is incremented. If the new count is within the `maxPermits`, the task is launched right away.
   - **Release a Permit**: When a task completes, the counter is decremented. If there are more tasks waiting, the completion triggers the start of another pending task.

   This logic ensures that no more than `maxPermits` tasks run at once.

4. **Asynchronous Execution**:  
   Instead of directly running tasks, the `AsyncSemaphore` uses a `submitter` function (`Function<T, ListenableFuture<R>>`) and an `Executor`. This design separates the concern of how tasks are executed from how they are scheduled. The `submitter` is responsible for:
   - Submitting the task to an asynchronous environment (e.g., making a remote call asynchronously).
   - Returning a `ListenableFuture` that will complete once the task finishes, fails, or is cancelled.

   By doing so, the semaphore integrates seamlessly with asynchronous paradigms common in large-scale or distributed systems.

5. **Failure and Cancellation Handling**:  
   - If a returned future fails or is cancelled, the semaphore’s logic can propagate that failure or cancellation to any downstream futures.
   - The `processAll` and `processAllToCompletion` methods offer two modes:
     - **`processAll`**: If any task fails, it cancels the rest. This is useful when a failure in one part makes the rest pointless.
     - **`processAllToCompletion`**: Even if some tasks fail, the others continue running. This is useful when partial results are still valuable.

6. **Usage Example**:  
   Suppose you have a method `getUserInfoById(int userId)` that returns a `ListenableFuture<UserInfo>`.

   ```java
   List<Integer> userIds = Arrays.asList(1, 2, 3, 4, 5);
   int maxConcurrency = 2; // Only run 2 tasks at once
   Executor executor = ... // Your favorite executor, maybe a thread pool
   
   // Process all userIds, but only two at a time
   ListenableFuture<List<UserInfo>> future = AsyncSemaphore.processAll(
       userIds,   // tasks
       client::getUserInfoById, // submitter
       maxConcurrency,
       executor
   );
   
   // future will complete once all userInfos are fetched or a failure occurs
   future.addListener(() -> {
       try {
           List<UserInfo> userInfos = future.get();
           System.out.println("Fetched user infos: " + userInfos);
       } catch (Exception e) {
           e.printStackTrace(); // Handle failure
       }
   }, executor);
   ```

   Internally, `AsyncSemaphore` queues up these five tasks. It starts with tasks #1 and #2 immediately (because `maxConcurrency`=2). When task #1 finishes, it starts task #3. When task #2 finishes, it starts task #4, and so on, never running more than two tasks simultaneously.

7. **Why This is Useful in Complex Systems**:  
   For a PhD-level engineer, consider a scenario like a distributed system where you must fetch data from multiple microservices or a large dataset. Overloading a single node or service with too many concurrent requests can lead to performance degradation or even service outages. The `AsyncSemaphore` provides a straightforward, asynchronous, and non-blocking way to throttle concurrency. It allows full pipeline parallelism up to a controlled limit, leading to efficient resource usage and stable performance.

**In Summary**:  
The `AsyncSemaphore` is a concurrency throttle for asynchronous tasks. It:
- Limits how many tasks run at once.
- Queues tasks until capacity (permits) is available.
- Uses `ListenableFuture` to integrate with asynchronous, non-blocking code.
- Handles failures and cancellations gracefully.

This pattern is common in large-scale systems, where controlling concurrency is essential for maintaining performance and reliability.