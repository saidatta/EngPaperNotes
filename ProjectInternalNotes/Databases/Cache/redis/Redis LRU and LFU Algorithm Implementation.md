
https://juejin.cn/post/7252357813649981495?searchId=20231006050418F9BD631730263F3EC54B
## Introduction
Redis, a high-performance, in-memory NoSQL database, extensively uses caching to achieve remarkable speeds, handling thousands of operations per second. As memory is generally more limited and costly compared to disk storage, Redis employs complex cache eviction strategies to manage memory usage efficiently. This guide delves into Redis's implementation of two pivotal algorithms: Least Recently Used (LRU) and Least Frequently Used (LFU), detailing their workings, advantages, and challenges.

## Redis LRU Implementation
### LRU Algorithm Basics
The LRU cache eviction algorithm is based on the premise that data not accessed recently is less likely to be needed shortly. It is a popular strategy in systems like Linux's page table exchange and MySQL's Buffer Pool. 
### Redis’s Approximate LRU Algorithm
Redis implements an approximate LRU algorithm, which, for performance efficiency, randomly selects a subset of keys for eviction consideration instead of examining all data. This approach sacrifices accuracy for speed and resource conservation.

Redis uses a simplified approach compared to the classic LRU algorithm's hash table + doubly linked list structure. This decision is motivated by the desire to avoid performance penalties associated with frequent data reordering and the additional memory overhead of maintaining a separate linked list.

### Implementation Details
In Redis, each object (`redisObject`) contains an `lru` field that stores the last access time, updated on every access or modification. This mechanism allows Redis to approximate LRU behavior without maintaining a full ordered list of all objects.

```c
#define LRU_BITS 24

typedef struct redisObject {
    unsigned type:4;
    unsigned encoding:4;
    unsigned lru:LRU_BITS;
    int refcount;
    void *ptr;
} robj;
```

Redis updates the LRU clock (`server.lruclock`) based on access patterns, using a resolution that defaults to seconds but can be adjusted. The LRU algorithm's effectiveness is partly dependent on the `server.hz` configuration, which affects the clock update frequency.

### Challenges and Limitations

Redis's LRU implementation has a limitation due to the 24-bit space allocated for the LRU clock, leading to a wrap-around issue after approximately 194 days. This can potentially make the eviction policy partly ineffective after this period.

## Redis LFU Implementation
### LFU Algorithm Basics
The LFU eviction algorithm assumes that data frequently accessed in the recent past is likely to be accessed again soon. Unlike LRU, LFU incorporates an element of time decay to adjust the significance of access frequency over time, acknowledging that recent accesses are more indicative of future access patterns.
### Redis’s LFU Algorithm
Redis's LFU implementation also uses the `lru` field in the `redisObject` structure but repurposes it to store both a frequency counter and a timestamp, effectively combining access frequency and recency into a single metric.

```c
uint8_t LFULogIncr(uint8_t counter) {
    if (counter == 255) return 255;
    double r = (double)rand()/RAND_MAX;
    double baseval = counter - LFU_INIT_VAL;
    if (baseval < 0) baseval = 0;
    double p = 1.0/(baseval*server.lfu_log_factor+1);
    if (r < p) counter++;
    return counter;
}
```
Redis introduces two configurable parameters to tune the LFU behavior: `lfu-log-factor` and `lfu-decay-time`, allowing fine-tuning of how access frequency and time decay impact eviction decisions.
### Analysis and Insights
An analysis of Redis's LFU implementation reveals a counterintuitive aspect: as the access frequency increases, the probability of the counter incrementing decreases, reflecting a logarithmic relationship. This design aims to balance capturing genuinely hot data against the memory and computational cost of maintaining precise access counts.
## Conclusion
Redis's LRU and LFU implementations showcase the database's sophisticated approach to memory management, balancing accuracy, performance, and resource utilization. While each algorithm has its advantages and ideal use cases, Redis provides the flexibility to choose the most suitable eviction strategy based on specific application requirements and data access patterns. Understanding the intricacies of these algorithms enables developers and system administrators to optimize Redis configurations for maximum efficiency and effectiveness.