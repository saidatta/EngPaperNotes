**Author**: antirez (Salvatore Sanfilippo)  
**Date**: 3382 days ago (~9+ years)  
**Views**: 345,747 (at time of posting)  

Redis is famously **single-threaded**, although it has a few helper threads for slow disk I/O. This article describes an evolution in Redis to handle **lazy frees**—non-blocking delete operations—and how broader architectural changes (removing shared `robj` usage in data structures) pave the way for faster Redis and potential future **threaded** features.

---
## 1. Background: Blocking `DEL` Command
### 1.1 The Problem
- **`DEL mykey`** in Redis can block the server if `mykey` is huge (e.g., 50 million objects). 
- This blocking behavior can last **seconds**, during which Redis serves **no** other commands.
- Historically, Redis tolerates blocking on large O(N) operations, but real use cases show it can be limiting when you need to remove large data sets quickly.
### 1.2 Initial Attempt: Incremental Free
1. **Incrementally** freeing: Instead of freeing all objects at once, do partial frees in repeated small steps (e.g., free 1,000 elements every millisecond).  
2. **Adaptive** approach: Increase or decrease how aggressively you free memory depending on memory usage trends.  
**Pros**  
- Lowers latency spikes (no big synchronous free).  
**Cons**  
- If new data is added faster than you free, memory can **grow unbounded**.  
- Requires complicated heuristics and adaptive logic to keep track of memory usage trends vs. free rates.  
- The single-threaded approach can still degrade performance (e.g., up to ~65% throughput in stress scenarios).

---
## 2. The Threaded Approach to Lazy Free
### 2.1 Motivations for Threaded Freeing.;÷
1. **Simplicity**: Freeing large data in a separate thread is more direct, rather than incremental steps in the main event loop.  
2. **Performance**: A dedicated free thread can release memory quickly without stalling the main loop’s command processing or incurring overhead of frequent partial frees.  

### 2.2 Challenge: Redis Architecture

- Redis has historically used **reference-counted objects** (`robj`), with heavy object **sharing** across internal operations (e.g., `SUNIONSTORE` might reuse existing objects in the target set).  
- **Client output buffers** might also store references to shared objects.  

**Why is that a problem?**  
- If objects are shared in many places, we can’t simply free them in another thread without carefully ensuring no one else references them.  
- Also, shared references sometimes *don’t* bring huge memory benefits in practice, and can cause unpredictable memory usage spikes after reloading data or unsharing.

---

## 3. Removing Shared `robj` in Data Structures

### 3.1 Flattening Out Indirection

Instead of:

```
key -> value_obj -> hash table -> robj -> sds_string
```

Redis internally moves to a design where aggregated data types (Sets, Hashes, Sorted Sets, etc.) contain:

```
hash table -> SDS strings
```

- That is, remove the `robj` layer within data structures, using raw SDS strings directly.

**Outcome**:
- Fewer cache misses (less pointer chasing).
- Slightly more copying at times (since you can’t simply “point” to a shared `robj`), but modern Redis workloads often spend more time on network I/O and cache misses than on copying small strings.
- Freed up the possibility for **threaded** lazy freeing.

### 3.2 Impact on Client Output Buffers

- Similarly, **client output buffers** no longer store references to shared `robj`s. Instead, they copy data into straightforward dynamic strings.  
- This removal of indirect references also simplified code and boosted performance.
---
## 4. Threaded Lazy Free Implementation

### 4.1 How It Works

1. When you want to delete a key with a large data set, Redis can schedule that data for **background freeing** in a separate thread.  
2. The main thread returns quickly, so the server doesn’t block.  
3. The background thread does all the freeing (invoking the memory allocator’s `free()` calls, etc.).
### 4.2 New Command: `UNLINK`
- The existing `DEL` remains **blocking**.  
- The new `UNLINK` command performs the same logical “delete” but in a **lazy** manner.  
- **Smart** approach:  
  - If the object is small, `UNLINK` frees it immediately (like `DEL`).  
  - If it’s huge, it offloads the freeing to a background task.  

#### Example
```
UNLINK mykey
```
If `mykey` is large, the command returns quickly, and the actual memory release happens asynchronously.

### 4.3 Non-Blocking `FLUSHALL` / `FLUSHDB`

- Now `FLUSHALL` and `FLUSHDB` can also have a **lazy** variant (`LAZY` option), removing all keys from the DB without blocking the main thread.  
- Under the hood, these calls queue all key frees in the background thread.
---
## 5. Performance Gains
1. **Negligible performance drop** when deleting massive keys. The main loop continues serving other requests.  
2. Removing `robj` overhead yields **speedups** in normal (non-delete) commands as well. Fewer pointer dereferences → better CPU cache usage.
3. Freed resources are reclaimed quickly **in practice** because a dedicated free thread can outpace typical creation rates.

---
## 6. Future Directions: Toward Threaded Redis?

### 6.1 Potential for Multi-Threaded I/O

- Now that data structures aren’t sharing many `robj`s, Redis could more safely attempt to **serve clients in multiple threads**.  
- The main DB still can live in a single thread (with shared-nothing semantics or partial locks), but reading/writing sockets, parsing commands, or other non-DB-bound tasks can run in parallel.  
- This approach is reminiscent of **memcached**’s multi-threaded model.

### 6.2 Threaded Slow Operations

- Certain slow commands on big data structures (e.g., `SMEMBERS` on a giant set) could run in a background thread, blocking only that key.  
- Other clients remain unblocked, and only the client that asked for the huge operation might wait for the result.

### 6.3 Unblocking Only Some Keys

- A lock or queue can mark a key as “busy” for background processing.  
- Other keys remain fully accessible in the main event loop.

---

## 7. API Note & Summary

- **`DEL`**: Remains **synchronous**.  
- **`UNLINK`**: New command for **non-blocking** delete (lazy free).  
- **`FLUSHALL`/`FLUSHDB LAZY`**: For non-blocking flush.  

**This approach**:
- Preserves backward compatibility (existing code using `DEL` sees old behavior).
- Offers new options to offload slow frees.  
- Simplifies Redis internals by removing shared `robj` usage in data structures, resulting in a net performance gain **even** for normal commands.

---

## 8. Code Excerpt & Visual

### 8.1 Example Snippet (Old Incremental Approach)
```c
/* From the removed incremental lazyfree code */
if (prev_mem < mem) mem_trend = 1;
mem_trend *= 0.9;
int mem_is_raising = mem_trend > .1;

size_t workdone = lazyfreeStep(LAZYFREE_STEP_SLOW);

/* Adjust timer_period to adapt frequency */
if (workdone) {
    if (timer_period == 1000) timer_period = 20;
    if (mem_is_raising && timer_period > 3)
        timer_period--;
    else if (!mem_is_raising && timer_period < 20)
        timer_period++;
} else {
    timer_period = 1000; /* 1Hz */
}
```
**Note**: This incremental approach was replaced with a **threaded** system.

### 8.2 Diagram

```mermaid
flowchart LR
    A[DEL mykey<br>(huge data)] -->|Blocking| B[Single-thread loop blocked<br>Old approach]
    A2[UNLINK mykey<br>(huge data)] -->|Non-blocking| C[Main loop continues<br>Background free thread]

    style A fill:#ffdede,stroke:#f00,stroke-width:2px
    style A2 fill:#deffde,stroke:#0f0,stroke-width:2px
    style B fill:#fff,stroke:#000,stroke-width:1px
    style C fill:#fff,stroke:#000,stroke-width:1px

    note right of C: Freed in background
```

- `DEL` still blocks, `UNLINK` queues the job for the background lazyfree thread.

---

## 9. Conclusion & Impact

**Key Lessons**  
1. **Lazy free**: Non-blocking deletion is critical for large data sets.  
2. **Threaded Freed**: Avoids the overhead of incremental frees in the main event loop.  
3. **`robj`less Data Structures**: Less pointer chasing → better performance for all commands.  
4. **Road to Multi-Threading**: Freed from heavy reference-counting burdens, Redis can now consider threading more deeply (e.g., multi-threaded I/O or per-key background operations).

**Overall**: This redesign allows Redis to remain extremely fast in typical commands **and** handle large-deletions without big latency hits. It also lays the foundation for future multi-threaded or partial concurrency improvements in Redis.

```  
────────────────────────────────────────────────────────────
 "A taboo is gone. Redis can remove big objects without 
 blocking, thanks to threaded lazy free. Redis is now 
 simpler internally and faster overall." 
────────────────────────────────────────────────────────────
```
```