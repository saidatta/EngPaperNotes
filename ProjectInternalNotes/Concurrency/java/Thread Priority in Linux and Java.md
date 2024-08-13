https://medium.com/@vikas.singh_67409/deep-dive-into-thread-priority-in-java-be1a5da30a34
### Key Takeaways
- **Linux Thread Model**: Threads in Linux are essentially processes, which has profound implications for how thread priorities work.
- **Java Thread Priority**: Java thread priorities interact with Linux’s scheduling mechanisms, particularly the Completely Fair Scheduler (CFS), which handles most of the scheduling duties on a Linux system.
- **System-Level Interaction**: Java thread priority management in Linux requires understanding the underlying kernel-level process scheduling and the limitations posed by the Linux threading model.
---
## In-Depth Analysis of Linux Threads and Process Scheduling

### Linux Threading Model
- **Threads as Processes**:
  - **Thread Creation**: In Linux, the `clone()` system call is used to create a new process. By passing specific flags (e.g., `CLONE_VM`, `CLONE_FS`, `CLONE_FILES`, `CLONE_SIGHAND`, `CLONE_THREAD`), Linux can create a process that shares its memory space, file descriptors, signal handlers, and other execution contexts with the parent process, effectively creating what higher-level programming languages like Java recognize as a "thread".
  - **Shared Resources**: The key Linux mechanisms that enable threads to share resources include:
    - **Memory**: `CLONE_VM` ensures that the new thread shares the virtual memory space of the parent.
    - **File Descriptors**: `CLONE_FILES` shares the file descriptor table.
    - **Signal Handlers**: `CLONE_SIGHAND` shares the signal handler table.
  - **Thread-Local Storage (TLS)**: Although threads share many resources, Linux still allows for thread-local storage to ensure that certain data remains thread-specific.

### Detailed Example: Java Thread Creation as Linux Processes
```java
import java.util.concurrent.TimeUnit;

public static void main(String[] args) throws Exception {
   Thread.currentThread().setName("MainThread");

   Thread newThread = new Thread(() -> {
      Thread.currentThread().setName("ForkedThread");
      try {
         TimeUnit.DAYS.sleep(1);
      } catch (InterruptedException ex) {
         // Handle exception
      }
   });
   newThread.start();
   newThread.join();
}
```
- **System-Level View**:
  - When running the above code, tools like `top` will show each Java thread as a separate process with its own Process ID (PID). This illustrates how Linux treats threads as processes, with the same scheduling rules applied to both.
  - **Implication**: Each thread in Java is a full-fledged process under the Linux kernel's perspective, which affects how CPU time is allocated to these threads.

### Advanced Process Scheduling in Linux
- **Static Priority and Scheduling Policies**:
  - **Static Priority Range**: In Linux, static priority ranges from 0 (default for normal processes) to 99 (for real-time processes).
  - **Scheduling Policies**:
    - **SCHED_FIFO**: First-In-First-Out real-time scheduling. Processes with the same priority are executed in the order they were added to the queue.
    - **SCHED_RR**: Round-Robin real-time scheduling. Processes with the same priority share CPU time in a round-robin fashion.
    - **SCHED_OTHER (CFS)**: The default time-sharing scheduler used for normal processes with a static priority of 0. This is implemented by the CFS.
  
- **Completely Fair Scheduler (CFS)**:
  - **Target Latency and Minimum Granularity**:
    - **Target Latency**: The total time within which each runnable task should get at least one opportunity to run.
    - **Minimum Granularity**: The smallest time slice that can be allocated to a task to prevent excessive context switching overhead.
  - **Scheduler Implementation**:
    - **Time Slices**: Each runnable task gets a time slice proportional to `1/N` of the target latency, where `N` is the number of runnable tasks.
    - **Nice Values**:
      - Nice values influence the allocation of time slices by weighting the `1/N` slice according to the process’s nice value.
      - **Formula**: The time slice is adjusted by a weight determined by the nice value:
        \[
        \text{Weight} = \text{sched\_prio\_to\_weight}[20 + \text{nice\_value}]
        \]
        - Lower nice values (higher priority) result in larger time slices.

### Nice Value Impact on Process Scheduling
- **Nice Value Calculation**:
  - **Kernel Mapping**: Linux uses a specific mapping of nice values to weights, which determines the proportion of CPU time a process receives relative to others.
  - **Weight Example**:
    - A process with a nice value of 0 (default) has a weight of 1024.
    - A process with a nice value of 4 has a weight of 423, meaning it receives approximately 41.3% of the CPU time compared to a process with a nice value of 0.
  - **Context Switching Considerations**: Higher nice values lead to smaller time slices, which can increase context switching overhead if many low-priority tasks are running.

### Real-World Impact: Example with Target Latency
- **Example Calculation**:
  - If the target latency is 24 ms and there are 8 runnable tasks:
    - **Without Nice Values**: Each task would receive `24 ms / 8 = 3 ms`.
    - **With Nice Values**: The actual time slice each task receives will be adjusted based on its nice value.
      - For instance, a task with a nice value of 4 may only get `3 ms * (423/1024) ≈ 1.24 ms`.

---

## Detailed Analysis of Java Thread Priority on Linux

### How Java Thread Priorities Work on Linux
- **Java Thread Priority API**:
  - Java provides the `Thread.setPriority()` method to adjust the priority of threads.
  - The JVM attempts to map Java thread priorities to native OS thread priorities.

### JVM Internals: Setting Thread Priority
- **JVM Code Path**:
  - The `Thread.setPriority()` method eventually calls native code specific to the operating system:
    - **JVM Method Call Flow**:
      1. **Java Method**: `Thread.setPriority()` in Java.
      2. **Native Method**: Calls `Thread.start0()` in `Thread.c`.
      3. **JVM Method**: Calls `JVM_StartThread` in `jvm.cpp`.
      4. **OS Specific Code**: For Linux, this calls `pthread_create()` via `os::create_thread()` in `os_linux.cpp`.

- **Thread Priority Mapping**:
  - On Linux, Java thread priorities map to Linux nice values, but this mapping requires enabling specific JVM flags (`-XX:ThreadPriorityPolicy=1` and `-XX:+UseThreadPriorities`).

### Example: Impact of JVM Flags on Thread Priority
```bash
$ sudo java -XX:ThreadPriorityPolicy=1 -XX:+UseThreadPriorities ThreadPriorityTest &
```
- **Observation**: After setting the JVM flags, the `nice` values reflect the Java thread priorities:
  - High priority thread (`Thread.MAX_PRIORITY`) -> Lower nice value (e.g., -3).
  - Low priority thread (`Thread.MIN_PRIORITY`) -> Higher nice value (e.g., 3).

- **Linux Priority Mapping**:
  - **Mapping Array in JVM**:
    - Java priorities are mapped to Linux nice values as shown below:
    ```cpp
    int os::java_to_os_priority[CriticalPriority + 1] = {
      19, // 0: Unused
       4, // 1: Thread.MIN_PRIORITY (nice 4)
       3, // 2
       2, // 3
       1, // 4
       0, // 5: Thread.NORM_PRIORITY (nice 0)
      -1, // 6
      -2, // 7
      -3, // 8
      -4, // 9: NearMaxPriority (nice -4)
      -5, // 10: Thread.MAX_PRIORITY (nice -5)
      -5  // 11: CriticalPriority (nice -5)
    };
    ```

### Practical Considerations: Limitations and Caveats
- **Impact on CPU Time Allocation**:
  - **CFS Behavior**: Even with modified thread priorities, CFS guarantees that all threads will get some CPU time, which can limit the effectiveness of priority adjustments.
  - **Too Many Low-Priority Threads**: If an application spawns many low-priority threads, it may still starve high-priority threads due to the proportional allocation of CPU time.
  
- **System-Level Constraints**:
  - **No Distinction Between Threads and Processes**: In Linux, there’s no distinction between threads and processes at the OS level, meaning other processes on the system can also impact the performance of high-priority threads in a Java application.
  - **Global Impact**: High-priority threads can still be affected by system load, and the Linux scheduler will enforce fair CPU usage across all tasks.

### Advanced CPU Management Techniques
- **Autogroups**:
  - **Autogrouping in Linux**: Processes can be grouped into autogroups, which the CFS scheduler treats as a single entity. This allows applying nice values to the entire group, ensuring consistent scheduling behavior within that group.
  
- **Control Groups (cgroups)**:
  - **cgroups for CPU Limitation**: cgroups

 allow fine-grained control over CPU usage by a set of processes, making them useful for containerized applications where resource management is crucial.
  - **Integration with Docker**: Since most applications are now deployed in containers, understanding how cgroups interact with thread priorities is essential for optimizing performance.

### Example: cgroups and Thread Priorities in Containers
- **Configuring cgroups**:
  - Set up a cgroup for a specific application or container:
    ```bash
    sudo cgcreate -g cpu:/mygroup
    sudo cgset -r cpu.shares=512 /mygroup
    sudo cgclassify -g cpu:/mygroup <PID>
    ```
  - Adjust the nice value of threads within this group to fine-tune their CPU allocation.

### Conclusion: Thread Priority in the Modern Linux Landscape
- **Challenges in Modern Systems**: In environments with heavy use of containers and virtualization, setting thread priorities in Java has limited impact. Instead, system-level resource management tools like cgroups and careful control of thread spawning are more effective.
- **Understanding and Optimization**: While understanding the interaction between Java thread priorities and Linux scheduling is critical, in most modern applications, performance tuning involves system-wide considerations that go beyond simple priority settings.