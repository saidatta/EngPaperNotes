## Overview
- **ConcurrentBag** is a Lock-free data structure used primarily for database connection storage.
- It forms the core of HikariCP.
- Designed for high-speed operations without compromising thread safety.
## Core Data Structures
1. **sharedList**
   - Type: `CopyOnWriteArrayList<T>`
   - Purpose: Cache all connections.
2. **threadList**
   - Type: `ThreadLocal<List<Object>>`
   - Purpose: Cache all connections used by a specific thread (acts as a quick reference).
3. **waiters**
   - Type: `AtomicInteger`
   - Purpose: Tracks the number of threads currently trying to acquire connections.
4. **handoffQueue**
   - Type: `SynchronousQueue<T>`
   - Purpose: Acts as a fast delivery queue with zero capacity.

## BagEntry State Management
- The elements in `ConcurrentBag` utilize certain variables to manage their state without locks.
- The abstract interface `IConcurrentBagEntry` lists possible states:
   ```java
   int STATE_NOT_IN_USE = 0;
   int STATE_IN_USE = 1;
   int STATE_REMOVED = -1;
   int STATE_RESERVED = -2;
   ```
## Acquiring Connections (`borrow` Method)
- Fast threads that consume many connections can retrieve connection objects quickly through `ThreadLocal` without accessing the broader pool.
- The ThreadLocal cache (`threadList`) can hold up to 50 connection object references.
- If no reusable object is found in `threadList`, the broader pool (`sharedList`) is accessed.
- If a connection cannot be retrieved from the pool, threads wait for other threads to release resources.
## Returning Connections (`requite` Method)
- Connections returned to the pool are first marked as usable.
- The system enters a loop, waiting for a consumer to take the connection.
- If the connection is not picked up by other threads upon return, it is stored in the corresponding `ThreadLocal`.
## Key Takeaways and Knowledge Points:
- **ThreadLocal**: Helps cache local resource references and uses thread-sealed resources to minimize lock conflicts.
- **CopyOnWriteArrayList**: A thread-safe structure optimal for read-heavy operations. Its concurrent nature minimally impacts read efficiency.
- **AtomicInteger**: Used to compute the number of waiting threads; its CAS (Compare-And-Swap) lock-free operation enhances speed.
- **SynchronousQueue**: A zero-capacity exchange queue facilitating rapid object handoffs.
- **CAS primitives**: Employed for efficient state changes.
- **Loop controls**: Methods like `park` and `yield` are utilized to prevent infinite loops from hogging the CPU.
- **Methods in concurrent data structures**: It's vital to understand and aptly use methods like `offer`, `poll`, `peek`, `put`, `take`, `add`, and `remove`.
- **`volatile` keyword**: Important for understanding how state changes are managed efficiently in multithreaded scenarios.
- **WeakReference**: Understanding its performance during garbage collection is crucial.
## Conclusion
By thoroughly understanding the intricacies of the `ConcurrentBag` in Hikari, one can master multithreaded programming. It's a compact yet comprehensive example that can significantly enhance one's multithreading skills.