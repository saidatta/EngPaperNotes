## 3.2.1 Hardware System Architecture

### Overview
- Eight-core system with each die having a pair of CPU cores each with its own cache. The pair of CPUs can communicate with each other through an interconnect, and the whole system interconnect allows the four dies to communicate with each other and with the main memory.

- Data moves through this system in units of “cache lines” usually ranging from 32 to 256 bytes in size.

- When a CPU loads a variable from memory to one of its registers, it must first load the cache line containing that variable into its cache.

- When a CPU stores a value from one of its registers into memory, it must also load the cache line containing that variable into its cache, but must also ensure that no other CPU has a copy of that cache line.

### Example 

Example of how data gets written in the case when CPU 0 writes to a variable whose cache line resides in CPU 7's cache:

1. CPU 0 checks its local cache, and does not find the cache line. It therefore records the write in its store buffer.

2. A request for this cache line is forwarded to CPU 0’s and 1’s interconnect, which checks CPU 1’s local cache, and does not find the cache line.

3. This request is forwarded to the system interconnect, which checks with the other three dies, learning that the cache line is held by the die containing CPU 6 and 7.

4. This request is forwarded to CPU 6’s and 7’s interconnect, which checks both CPUs’ caches, finding the value in CPU 7’s cache.

5. CPU 7 forwards the cache line to its interconnect, and also flushes the cache line from its cache.

6. CPU 6’s and 7’s interconnect forwards the cache line to the system interconnect.

7. The system interconnect forwards the cache line to CPU 0’s and 1’s interconnect.

8. CPU 0’s and 1’s interconnect forwards the cache line to CPU 0’s cache.

9. CPU 0 can now complete the write, updating the relevant portions of the newly arrived cache line from the value previously recorded in the store buffer.

### Points for discussion

- The sequence of events here represents a simplified version of a discipline called cache-coherency protocols. These protocols can cause considerable traffic which can significantly degrade your parallel program’s performance.

- However, if a variable is being frequently read during a time interval during which it is never updated, that variable can be replicated across all CPUs’ caches, which allows for faster access.

## 3.2.2 Costs of Operations

### Overview

- Shows the overheads of some common operations important to parallel programs. 

- The operations' costs are normalized to a clock period in the third column, labeled "Ratio". 

- CAS (compare-and-swap) is an atomic operation which compares the contents of the specified memory location to a specified "old" value, and if they compare equal, stores a specified "new" value. The operation is atomic in that the hardware guarantees that the memory location will not be changed between the compare and the store.

- The “same-CPU” prefix means that the CPU now performing the CAS operation on a given variable was also the last CPU to access this variable, so that the corresponding cache line is already held in that CPU’s cache.

- Blind CAS is a case where the software specifies the old value without looking at the memory location. This is usually used when attempting to acquire a lock. 

- The table also shows costs for CAS operations for different combinations of CPUs in the same socket, different sockets and across inter