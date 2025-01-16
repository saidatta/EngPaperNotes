
aliases: [Zero-Copy in Go, Linux splice, Pipe Pool, Go Concurrency Optimization]
tags: [Go, Zero-Copy, Splice, Pipe Pool, Concurrency]
creation date: 2021-06-05
https://strikefreedom.top/archives/pipe-pool-for-splice-in-go

> **Author**: Pan Shao  
> **Posted**: 06-05-2021  
> **Last updated**: 06-06-2024  
> **Views**: 31740  
> **Word Count**: 5695 (~9.5 minutes reading time)  

## 1. Introduction
If you’ve used **Go** to build a proxy server or do low-level I/O, you’ve likely encountered methods such as `io.Copy()`, `io.CopyN()`, `io.CopyBuffer()`, `io.ReaderFrom()`. Under the hood, for Linux-based systems, these interfaces can leverage zero-copy techniques like **`sendfile`** and **`splice`**.  

This article shares an optimization for **zero-copy** in Go that uses a **pipe pool** to reduce overhead in frequent `splice()` calls, thereby improving performance for high-throughput scenarios. We’ll also touch upon concurrency and synchronization design principles that emerged from this optimization work.

> **Disclaimer**: The author notes any mistakes or omissions are unintentional, and feedback is appreciated.

---

## 2. `splice()` in Linux

### 2.1 Overview
`splice()` is a Linux system call that implements zero-copy by transferring data between file descriptors **without** routing data through user space. It typically requires using a **pipe** as an intermediate.  

```c
#include <fcntl.h>
#include <unistd.h>

int pipe(int pipefd[2]);
int pipe2(int pipefd[2], int flags);
ssize_t splice(int fd_in, loff_t *off_in,
               int fd_out, loff_t *off_out,
               size_t len, unsigned int flags);
```

- **`fd_in`, `fd_out`**: input and output FDs (one must be a pipe).  
- **`off_in`, `off_out`**: optional offsets.  
- **`len`**: bytes to transfer.  
- **`flags`**: bitmask to tweak behavior (e.g. `SPLICE_F_NONBLOCK`).

### 2.2 Usage Example

```c
int pfd[2];
pipe(pfd);

// Move up to 4096 bytes from file_fd -> pipe write-end
ssize_t n = splice(file_fd, NULL, pfd[1], NULL, 4096, SPLICE_F_MOVE);

// Then move data from pipe read-end -> socket_fd
n = splice(pfd[0], NULL, socket_fd, NULL, n, SPLICE_F_MOVE | SPLICE_F_MORE);
```

#### Data Flow

```
file -> [kernel] -> pipe -> [kernel] -> socket -> NIC
```

1. **pipe()** is called once, creating a unidirectional pipe (two FDs).
2. **splice()** from `file_fd` into the pipe.  
3. **splice()** from pipe to `socket_fd`.
4. Data is transferred in the kernel via page references (pages are “moved,” not copied).

---

## 3. Pipe Pool for `splice()`

### 3.1 Why a Pipe Pool?
By design, `splice()` requires a **pipe** as an intermediate.  
- If you only do a single large transfer, creating/destroying a pipe is negligible.  
- But if your system frequently uses `splice()` (like a proxy forwarding data for many sockets), the overhead of creating/destroying pipes for each transaction is huge.

**Solution**: **reuse** pipes via a **pipe pool**.

#### 3.1.1 Pipe Pool in HAProxy
[HAProxy](http://www.haproxy.org/) is a well-known high-performance TCP/HTTP load balancer. For traffic forwarding, it uses `splice()` heavily. To reduce system overhead, HAProxy **reuses** pipe buffers.  

Key design points in HAProxy’s pipe pool:
1. A **global** resource pool of available pipe buffers (linked list + spin lock).  
2. **Thread-local** resource cache: Each thread tries its local list first (no locking). Only if empty does it acquire the global lock and fetch from the global pool.  
3. When returning a pipe, the thread puts it in local storage or, if local is “too big,” returns it to the global pool.

**Benefits**:
- Minimizes global lock contention by using local caches, a standard concurrency pattern.

---

### 3.2 Pipe Pool in Go

**Go** 1.17 introduced an internal optimization for `splice()` called a **pipe pool** (proposed by the author and others) so that frequent `splice()` calls can reuse the same pipe FDs, significantly reducing overhead.  

However, Go has no built-in thread-local storage (TLS) like C, because Go uses a **GMP** (Goroutine, M-threads, Processor) scheduler, abstracting away OS threads. Instead, it provides:
- `sync.Pool`: A concurrency-friendly pool for reusable objects.
- `runtime.SetFinalizer`: Let’s you define a finalizer callback to clean up an object before GC.

#### 3.2.1 sync.Pool
`sync.Pool` is effectively a local + global caching mechanism for short-lived objects. Internally, it uses:
- A **private** cache per P (logical processor in Go runtime).
- A **shared** lock-free linked structure for cross-P borrowing.

**Objects** in `sync.Pool` can be reclaimed after 2 GC cycles (the “victim cache” model).  
**Hence** we can store pipe FDs as objects in a `sync.Pool` for reuse.

#### 3.2.2 Using `runtime.SetFinalizer`
When an FD in `sync.Pool` is no longer referenced (except by the pool), the GC can reclaim it. But we need to also close the **OS-level** pipe.  
- A finalizer can be attached to the FD object, which calls `close()` on the OS pipe FD.  
- Once the object is truly unreachable, the finalizer runs asynchronously, closing the pipe resources.

### 3.3 Performance Impact
After implementing the pipe pool in Go:

```
name                   old time/op   new time/op   delta
SplicePipe-8           1.36µs       0.02µs       -98.57%
SplicePipeParallel-8    747ns       4ns          -99.41%
```

A >99% reduction in overhead for repeatedly creating/destroying pipe FDs. Memory allocs drop from 1 to 0. This is an *idealized benchmark* but still shows a big improvement for frequent zero-copy calls.

**As of Go 1.17**, this pipe pool optimization is included. If you do heavy `splice()` usage, you’ll see big gains.

---

## 4. Additional Concurrency Optimization Insights

From the design of pipe pools in HAProxy and Go, we see several concurrency lessons:

1. **Resource Reuse**: The simplest and most direct way to improve concurrency performance—avoid repeated creation/destruction of expensive resources.  
2. **Data Structure Choice**: 
   - Arrays are good for random access, contiguous memory → better CPU cache usage, but might not be flexible for expansions/shrinking.  
   - Linked lists handle dynamic insertion/removal well. For pools where all items are “equally suitable,” a linked list often suffices.  
3. **Local + Global**: 
   - Common approach to reduce lock contention: each thread (or CPU) keeps a local cache, only accessing the global pool if needed.  
4. **User-Mode Locking**: 
   - For extremely short critical sections, spin locks or atomic operations can be faster than kernel-based mutexes.  
5. **Leveraging the Runtime**: 
   - Languages like Go/Java can incorporate runtime-level solutions (GC-based finalization) to free resources at appropriate times.  
   - C-based solutions (like HAProxy) require manual resource management.

---

## 5. Summary

By implementing a **pipe pool** for `splice()` in the **Go** standard library, we achieve:
- Substantial speedup for repeated zero-copy operations.
- Freed from overhead of creating/destroying pipes each time.
- Automatic cleanup via `sync.Pool` + finalizers.

**Key Takeaways**:
1. **Zero-copy** with `splice()` is powerful for proxies and high-volume forwarding.  
2. **Pipe pool** is essential for repeated usage of `splice()` to avoid heavy system call overhead.  
3. **sync.Pool** in Go effectively acts like TLS + a global shared region.  
4. **Finalizers** let us release OS resources automatically at safe times.  
5. This feature landed in **Go 1.17**—if your system does frequent zero-copy forwarding, you can benefit from significant speedups.

---

## 6. References & Further Reading

1. **`sync.Pool` Documentation**: [Go Docs](https://pkg.go.dev/sync#Pool)
2. [Pipe Pool in HAProxy (GitHub code)](https://github.com/haproxy/haproxy)
3. [internal/poll: implement a pipe pool for splice() call (Go source commit)](https://go.dev/cl/330869)
4. [internal/poll: fix the intermittent build failures with pipe pool](https://go.dev/cl/334749)
5. [Use Thread-local Storage to Reduce Synchronization](https://dl.acm.org/doi/10.1145/nnnnnnn.nnnnnnn)
6. [ELF Handling For Thread-Local Storage](https://akkadia.org/drepper/tls.pdf)

```