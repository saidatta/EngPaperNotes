https://www.uber.com/blog/uber-gc-tuning-for-improved-presto-reliability/
#### Context: Uber's Use of Presto
- **Presto at Uber**: Utilized for querying a myriad of data sources, Presto is pivotal for data-driven decisions at Uber. Operating across 20 clusters with over 10,000 nodes, it handles 500,000 daily queries, engaging with around 12,000 users weekly.
- **Infrastructure**: The infrastructure spans big machines (with >300 GB heap memory) for heavy-duty tasks and smaller machines (<200 GB heap memory) for lighter operations, tailored to both interactive and batch workloads.
#### The Challenge: Memory Fragmentation and Garbage Collection

- **Problem Statement**: Despite ongoing optimizations, Uber’s Presto clusters faced issues with full Garbage Collections (GCs) and occasional Out-of-Memory (OOM) errors, impacting performance and reliability.

#### Garbage Collection in Java: G1GC
- **G1GC Overview**: The Garbage-First Garbage Collector (G1GC) aims to balance throughput and latency, managing memory in a generational manner and dividing the heap into regions for efficient garbage collection.
- **Heap Division**: The heap is segmented into regions that can be categorized as young (further divided into Eden and survivor spaces), old, or free, facilitating a structured approach to memory management.
#### G1GC Tuning at Uber
- **JDK 8 Tuning**: Initially, Uber’s focus was on adjusting the `-XX:InitiatingHeapOccupancyPercent` flag to optimize the threshold for starting concurrent mark-and-sweep cycles.
- **JDK 11 Adjustments**: With the introduction of dynamic IHOP in JDK 11, Uber adopted a systematic approach to fine-tune G1GC, incorporating additional GC metrics and adjusting heap space configurations.
#### Tuning Strategy and Results
- **Key Adjustments**: 
  - Decreased the maximum young generation size from 60% to 20%.
  - Increased the free space threshold from 10% to 40% and adjusted the heap waste percentage.
- **Impact**: These adjustments led to a reduction in full GC occurrences and enhanced the overall performance of Presto clusters.
#### Final Tuning Flags and Outcomes
- **Adopted Flags**: 
  - `-XX:+UnlockExperimentalVMOptions`
  - `-XX:G1MaxNewSizePercent=20`
  - `-XX:G1ReservePercent=40`
  - `-XX:G1HeapWastePercent=2`
- **Cluster Performance**: Post-tuning, Uber witnessed a significant decrease in internal OOM errors and full GCs, thereby improving the reliability and efficiency of their Presto clusters.
#### Forward Path
- **Future Focus**: Uber intends to extend its GC tuning efforts to other storage applications, anticipating unique challenges due to the variance in heap sizes and operational demands.
- **Community Sharing**: Insights gained from the Presto GC tuning experience will contribute to developing best practices and guidelines for broader application across Uber’s storage solutions, aiming for enhanced performance and reliability.

### Technical Insights and Best Practices

- **Generational GC Understanding**: Recognizing the impact of generational garbage collection on application performance is crucial for systems operating at scale.
- **Heap Configuration**: Adjusting heap configurations based on workload characteristics and monitoring outcomes is key to optimizing GC performance.
- **Dynamic IHOP Tuning**: Adapting to changes in JVM versions (such as the dynamic IHOP in JDK 11) requires a flexible approach to GC tuning, leveraging detailed GC logs and metrics for data-driven adjustments.
- **Application-Specific Tuning**: GC tuning is highly application-specific; strategies successful for one system (like Presto at Uber) may not directly apply to others, emphasizing the need for tailored tuning practices.

### Conclusion

Uber’s journey in tuning G1GC for Presto showcases the critical role of garbage collection optimization in enhancing system reliability and performance. Through meticulous tuning and monitoring, Uber achieved significant improvements, underscoring the importance of continuous optimization in managing large-scale, data-intensive applications.