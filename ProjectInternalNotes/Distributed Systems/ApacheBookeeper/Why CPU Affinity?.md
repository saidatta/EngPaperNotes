https://github.com/apache/bookkeeper/pull/1641/files
Apache BookKeeper's implementation of CPU affinity seeks to reduce latency in cases where latency is a critical factor. Here's a breakdown of what this issue entails:

1. **Motivation**: The main goal of introducing CPU affinity is to reduce latency. It's not designed to be the default setting because focusing on reducing latency can compromise other aspects, like throughput. However, in scenarios where latency is a key concern, these changes could be beneficial.

2. **Mechanism**: By pinning a specific thread to a particular CPU, it ensures that no other processes will run on that CPU, thereby reducing the amount of context switching. Context switching can cause variations in latency. In Apache BookKeeper's implementation, a thread that wants to be pinned to a specific CPU needs to call `CpuAffinity.acquireCore()`. This not only assigns the thread to a CPU but also disables hyper-threading on that CPU to further optimize performance.

3. **Discovery of Isolated CPUs**: The solution also automatically discovers available isolated CPUs (those not being used by the system for other tasks), which can be utilized for the threads requesting affinity.

4. **Independent CPU Acquisition**: By using file-based locks, the solution allows multiple processes on the same machine to independently acquire CPUs. This means even if there are multiple instances of Apache BookKeeper (or other applications) running on the same machine, they can all utilize CPU affinity without interfering with each other's CPU assignments.

These changes aim to optimize Apache BookKeeper's performance in low-latency environments by using CPU affinity to pin threads to specific CPUs, reducing context switching, disabling hyper-threading, and enabling isolated CPUs to be utilized independently by different processes.