## Overview
- Thermal throttling, cache misses, and I/O operations are additional obstacles that can affect the performance of parallel programming.
- These elements are further complicated by the increasing speed of CPUs and the requirements of maintaining data consistency across multiple cores in multi-threaded systems.

## Thermal Throttling
- Thermal throttling is a result of CPU micro-optimizations that, while reducing clock cycles, may increase power consumption and heat dissipation.
- Exceeding the cooling system's capacity can lead to thermal throttling of the CPU, which might involve reducing its clock frequency to avoid overheating.
- If performance is crucial and you cannot modify the cooling system (for example, when using a rented server from a cloud provider), you may need to apply algorithmic optimizations or parallelize your code over multiple cores to distribute the work and heat.

![CPU Encounters Thermal Throttling](image_link)
Figure 3.8: CPU Encounters Thermal Throttling

## Cache Misses
- Cache misses pose a significant obstacle in multi-threading environments where variables are frequently shared among CPUs.
- When a CPU wants to modify a variable recently modified by another CPU, it incurs an expensive cache miss.
- This is because the variable resides in the other CPU’s cache but not in the accessing CPU’s cache.
- Despite large caches in modern CPUs aimed at reducing performance penalties due to high memory latencies, cache misses can reduce performance.

![CPU Meets a Cache Miss](image_link)
Figure 3.9: CPU Meets a Cache Miss

## I/O Operations
- I/O operations involving networking, mass storage, or human beings pose greater obstacles than internal operations, such as cache misses.
- I/O operations could be considered as a type of CPU-to-CPU I/O operation, making them among the cheapest.
- Differences exist between shared-memory and distributed-system parallelism: Shared-memory parallel programs usually deal with obstacles no worse than a cache miss, while distributed parallel programs typically incur larger network communication latencies.
- The goal of parallel hardware design is to reduce the ratio of communication overhead to the actual work being done to achieve performance and scalability goals. Similarly, parallel software design aims to reduce the frequency of expensive operations like communication cache misses.

![CPU Waits for I/O Completion](image_link)
Figure 3.10: CPU Waits for I/O Completion

## Quick Quiz Answer
- CPU designers have indeed worked on reducing the overhead of cache misses. They use techniques such as prefetching data and instructions, out-of-order execution to fill the pipeline, and hyper-threading to keep the CPU busy while waiting for a cache miss to resolve. However, cache misses still pose a significant challenge due to the nature of parallel programming, and the effectiveness of these solutions varies widely depending on the specifics of the program being executed.

## Additional Notes
- While identifying potential performance obstacles is important, it is also necessary to determine whether these obstacles significantly impact the performance of your particular code.
- Detailed analysis and profiling are required to uncover the most critical obstacles in your code. This will be discussed in subsequent sections.