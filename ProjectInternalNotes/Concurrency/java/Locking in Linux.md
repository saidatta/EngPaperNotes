https://medium.com/@vikas.singh_67409/lock-on-linux-a-deep-dive-8269305d15dd
### Key Takeaways
- **Locking Mechanism**: Locks (mutexes) are synchronization primitives that ensure mutual exclusion, allowing only one thread to hold the lock at any given time.
- **Futex (Fast Userspace Mutex)**: A highly efficient Linux-specific mechanism designed to handle uncontested locks in user space, reducing the need for expensive kernel space transitions.
- **System Calls and Kernel Interaction**: A detailed examination of the `futex` system call and its internal handling within the Linux kernel, including atomic operations and wait queue management.

---
## Introduction to Locking in Linux
### What is a Lock (Mutex)?
- **Definition**: A lock (or mutex) is a synchronization primitive that prevents multiple threads from accessing a shared resource simultaneously by allowing only one thread to acquire the lock at a time.
- **Variants**: 
  - **Recursive Lock**: Allows the thread that currently holds the lock to reacquire it multiple times without blocking itself.
  - **Non-Recursive Lock**: The thread will block even if it attempts to acquire the lock it already holds, preventing deadlock in non-reentrant code.

### Example: Simple Lock Implementation in Python
```python
#!/usr/bin/env python
import threading

lock = threading.Lock()
# First lock acquisition succeeds
lock.acquire()
print("Successfully acquired lock")

# Second lock acquisition blocks (non-recursive lock)
lock.acquire()
print("This should not get printed")
```
- **Expected Output**:
  - The program will hang on the second `lock.acquire()` call because the lock is non-recursive. The process will need to be terminated manually using `SIGTERM`.

### System Call Tracing with `strace`
- **Command**:
  ```bash
  $ strace -e execve,futex ./lock.py
  ```
  - **`execve`**: System call used to execute a program.
  - **`futex`**: System call used to manage fast userspace mutexes.
- **Output**:
  - The program starts with `execve` and blocks on the `futex` system call after the second lock acquisition attempt, indicating that the thread is waiting for the lock to be released.

---

## Deep Dive into Futex

### What is Futex?
- **Definition**: Futex (Fast Userspace Mutex) is a Linux kernel feature that minimizes the overhead of locking mechanisms by handling uncontested locks entirely in user space, using atomic operations. When contention occurs, it efficiently transitions the locking mechanism to kernel space.
  
- **Atomic Operations**:
  - **CMPXCHG (Compare and Swap)**: A crucial atomic instruction that ensures the integrity of lock acquisition by comparing the current value of a memory location with an expected value and, if they match, swapping it with a new value.
  - **Example**:
    ```cpp
    bool compare_and_swap(int *accum, int *dest, int newval) {
      if (*accum == *dest) {
        *dest = newval;
        return true;
      } else {
        *accum = *dest;
        return false;
      }
    }
    ```
    - **accum**: Pointer to the current value of the lock.
    - **dest**: The expected value (e.g., `0` if the lock is free).
    - **newval**: The value to set (e.g., `1` to indicate the lock is acquired).
    - **Outcome**: If `accum` equals `dest`, the swap occurs, and the lock is acquired.
- **Futex Mechanism**:
  - **User-Space Operation**:
    - The thread initially tries to acquire the lock using `CMPXCHG`. If the lock is uncontested (value is `0`), the operation succeeds entirely in user space.
  - **Kernel-Space Transition**:
    - If the lock is already held (`CMPXCHG` fails), the thread invokes the `futex` system call to wait for the lock, transitioning the operation to kernel space.
### Efficiency of Futex
- **Avoiding Kernel Calls**: 
  - When the lock is uncontested, the entire operation remains in user space, avoiding costly context switches and kernel interventions, leading to significant performance improvements in multi-threaded applications.
- **Atomicity**: 
  - The atomic nature of `CMPXCHG` ensures that no other thread can interfere between the compare and swap operations, maintaining consistency and correctness even in highly concurrent environments.
### Futex System Call Workflow
1. **Initial Attempt**: The thread uses `CMPXCHG` to acquire the lock atomically.
2. **Futex Wait**: If the lock is held, the thread makes a `futex` system call with the `FUTEX_WAIT` flag, indicating that it should wait until the lock is released.
3. **Kernel Handling**: The kernel adds the thread to the wait queue associated with the futex.
4. **Wake-Up**: When another thread releases the lock, it calls `futex` with the `FUTEX_WAKE` flag, waking up the threads waiting for the lock.
---
## System Call Mechanism in Linux

### What is a System Call?
- **Definition**: A system call is a mechanism that allows user-space programs to request services from the kernel, such as I/O operations, process control, or inter-process communication.
- **Execution Flow**:
  - **User-Space to Kernel-Space Transition**:
    - System calls are invoked by user-space programs and result in a transition to kernel space, where the operating system executes the requested operation. This transition is triggered by a software interrupt using privileged instructions such as `syscall` on x86_64 or `int 0x80` on x86.
### System Call Execution: Example with `futex`
- **Lock Initialization in Python**:
  - **Threading Module**: Python's `threading.Lock()` internally maps to POSIX-compliant `pthread` functions on Linux, enabling Python to utilize low-level thread synchronization primitives.
  - **Memory Allocation**: Lock objects in Python are backed by binary semaphores (`sem_t`), allocated using `PyMem_RawMalloc`.
### Python to C and C to Kernel
- **Lock Creation**:
  - **Python Code**: `threading.Lock()` is mapped to `PyThread_allocate_lock`.
  - **C Code**: In CPython, this maps to `pthread` functions like `pthread_mutex_lock`, which are defined in the POSIX thread library (`pthread`).
- **Futex Wait**:
  - **Flow**: The `sem_wait` call in Python eventually leads to the `futex` system call via a series of glibc function calls.
  - **Key Transition**: The call to `futex_abstimed_wait_cancelable` is where the user-space code transitions to the kernel to manage the wait queue.
### Assembly and Privileged Instructions
- **Assembly Instruction**: 
  - The `syscall` instruction on x86_64 architectures triggers a software interrupt, causing the CPU to switch from user mode to kernel mode. This instruction is critical in invoking system calls like `futex`.
- **Register Usage in Syscall**:
  - **Registers**: 
    - `rax`: Holds the system call number (e.g., `202` for `futex`).
    - `rdi`, `rsi`, `rdx`: Hold the first three arguments of the system call.
    - **Example Macro**:
    ```cpp
    #define internal_syscall2(number, err, arg1, arg2) \
    ({ \
      unsigned long int resultvar; \
      register unsigned long int _a1 asm ("rdi") = (unsigned long int)(arg1); \
      register unsigned long int _a2 asm ("rsi") = (unsigned long int)(arg2); \
      asm volatile ( \
        "syscall\n\t" \
        : "=a" (resultvar) \
        : "0" (number), "r" (_a1), "r" (_a2) \
        : "memory", "rcx", "r11" \
      ); \
      (long int) resultvar; \
    })
    ```
### Syscall Latency and Optimization
- **Avoiding System Calls**: System calls are expensive due to the overhead of context switching, saving/restoring registers, and flushing the TLB (Translation Lookaside Buffer). Futex is optimized to minimize the need for system calls unless contention occurs, significantly reducing syscall-related latency.
---
## Kernel Handling of Futex System Call
### Futex Key and Wait Queue
- **Futex Key**: A unique identifier derived from the memory address of the futex, used to locate the appropriate wait queue in the kernel.
- **Wait Queue**: The kernel maintains a hash table (`futex_queue`) where each key maps to a linked list of threads waiting on the futex. The linked list nodes are instances of the `futex_q` structure.
### Detailed Process Flow
1. **Atomicity of Queue Operations**: 
   - The kernel uses a spinlock to ensure that the process of adding a thread to the futex wait queue is atomic. This prevents race conditions where the futex state could change (e.g., being released) during the queueing process.
2. **Futex Availability Check**:
   - Before blocking the thread, the kernel re-checks whether the futex is still held. This check is crucial to avoid scenarios where a thread is incorrectly put to sleep despite the lock being available.
   - If the futex is not held, the kernel returns the `EWOULDBLOCK` error, indicating that the thread should not wait.
3. **Thread Blocking**:
   - If the futex is still held, the thread is added to the wait queue. The thread is then put to sleep until another thread releases the lock and calls `futex` with the `FUTEX_WAKE` flag.
### Kernel Structures Involved
- **`futex_q`**: Represents each thread waiting on a futex. Contains references to the thread's task structure and the futex key.
- **`futex_hash_bucket`**: Part of the hash table (`futex_queues`) that holds lists of `futex_q` structures. Each bucket corresponds to a set of futex keys.
- **`wait_queue_t`**: The data structure used by the Linux kernel to manage wait queues, ensuring that threads are efficiently managed while they are blocked on futexes.
### Performance Considerations
- **Lock Contention**: When multiple threads contend for a lock, futex efficiently manages the transition from user space to kernel space, minimizing the overhead and maintaining performance under high contention.
- **Scalability**: The futex mechanism scales well with the number of threads, especially in multi-core environments, due to its ability to minimize kernel space interactions when possible.
### Potential Issues and Race Conditions
- **Race Condition Mitigation**:
  - The kernel's re-check of the futex state before blocking a thread is critical in preventing race conditions that could lead to deadlocks or missed wake-ups.
- **Spinlocks and Atomicity**:
  - Spinlocks are used within the kernel to ensure that futex operations are atomic, especially when manipulating shared data structures like the wait queues.
---
## Conclusion
### Overview
- **Futex Efficiency**: Futexes provide a highly efficient mechanism for locking in Linux by minimizing kernel-space interactions and using atomic operations in user space.
- **Traditional Locks**: Despite the advantages of futexes, traditional locking mechanisms (e.g., spinlocks, mutexes) are still necessary, particularly in scenarios involving low-level system code or complex synchronization needs across multiple cores.
### Future Exploration
- **Advanced Locking Mechanisms**:
  - Further study of advanced synchronization primitives such as spinlocks, barriers, and condition variables, particularly in the context of multi-core systems and real-time applications.

---

Below is a Go implementation simulating the kernel handling of the futex system call, including the futex key, wait queue, atomicity of queue operations, futex availability check, and thread blocking:
### Go Implementation of Futex System

```go
package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// FutexKey represents the unique identifier for a futex, derived from its memory address.
type FutexKey uintptr

// FutexQ represents a node in the futex wait queue, which corresponds to a thread waiting on a futex.
type FutexQ struct {
	taskChan chan struct{} // Channel to simulate thread blocking
	key      FutexKey      // The futex key
}

// FutexQueue is a structure to manage futex wait queues.
type FutexQueue struct {
	waitQueue map[FutexKey][]*FutexQ // Hash table where each key maps to a list of FutexQ (linked list of threads)
	mu        sync.Mutex             // Spinlock to ensure atomic operations on the queue
}

// NewFutexQueue initializes a new FutexQueue.
func NewFutexQueue() *FutexQueue {
	return &FutexQueue{
		waitQueue: make(map[FutexKey][]*FutexQ),
	}
}

// FutexWait adds the thread to the futex wait queue and blocks it if the futex is still held.
func (fq *FutexQueue) FutexWait(addr *uintptr, futexHeld *atomic.Bool) error {
	key := FutexKey(*addr)

	fq.mu.Lock()

	// Re-check whether the futex is still held
	if !futexHeld.Load() {
		fq.mu.Unlock()
		return fmt.Errorf("EWOULDBLOCK")
	}

	// Add the thread to the wait queue
	task := &FutexQ{
		taskChan: make(chan struct{}),
		key:      key,
	}
	fq.waitQueue[key] = append(fq.waitQueue[key], task)

	fq.mu.Unlock()

	// Block the thread until it is woken up
	<-task.taskChan
	return nil
}

// FutexWake wakes up one thread waiting on the futex.
func (fq *FutexQueue) FutexWake(addr *uintptr) {
	key := FutexKey(*addr)

	fq.mu.Lock()

	if waiters, exists := fq.waitQueue[key]; exists && len(waiters) > 0 {
		// Wake up the first thread in the wait queue
		task := waiters[0]
		close(task.taskChan) // Unblock the thread

		// Remove the woken thread from the wait queue
		fq.waitQueue[key] = waiters[1:]

		// Clean up the key if no more waiters
		if len(fq.waitQueue[key]) == 0 {
			delete(fq.waitQueue, key)
		}
	}

	fq.mu.Unlock()
}

func main() {
	futexQueue := NewFutexQueue()
	futexHeld := atomic.Bool{}
	futexHeld.Store(true) // The futex is initially held

	addr := uintptr(0x1000) // Simulate the address of the futex

	// Simulate a thread waiting on the futex
	go func() {
		if err := futexQueue.FutexWait(&addr, &futexHeld); err != nil {
			fmt.Println("Thread could not wait:", err)
		} else {
			fmt.Println("Thread successfully woke up")
		}
	}()

	// Simulate the futex being released after some time
	time.Sleep(1 * time.Second)
	futexHeld.Store(false) // The futex is no longer held
	futexQueue.FutexWake(&addr)

	// Give enough time for the waiting thread to wake up
	time.Sleep(1 * time.Second)
}
```

### Explanation:

1. **FutexKey**: Represents the unique identifier for a futex, derived from its memory address (`uintptr`).

2. **FutexQ**: Represents each thread waiting on a futex. It includes a `taskChan` to simulate the blocking and waking of a thread and the futex key it is waiting on.

3. **FutexQueue**: Manages the futex wait queues. It uses a hash table (`waitQueue`) where each futex key maps to a list (linked list) of `FutexQ` structures representing threads waiting on the futex. The `mu` mutex ensures atomic operations on the queue.

4. **FutexWait**:
   - The method re-checks whether the futex is still held after acquiring the mutex to prevent race conditions.
   - If the futex is still held, it adds the thread to the wait queue and then blocks the thread by reading from the `taskChan`.
   - If the futex is not held, it returns an `EWOULDBLOCK` error.

5. **FutexWake**:
   - Wakes up one thread waiting on the futex by closing its `taskChan`, which unblocks the thread.
   - Removes the woken thread from the wait queue. If no threads remain, it removes the key from the hash table.

6. **Simulation**:
   - The main function simulates a thread waiting on the futex and then releases the futex after a delay. The futex wake function is called to unblock the waiting thread.

### Key Points:
- **Atomicity**: Ensured by locking mechanisms (`sync.Mutex`).
- **Wait Queue Management**: Implemented using a hash table where each key maps to a list of waiting threads.
- **Thread Blocking**: Simulated using Go channels, which block until they receive a signal to wake up.

---
#### Performance Considerations

1. **Lock Contention**:
    
    - The `FutexQueue` uses a mutex (`mu`) to ensure atomic operations on the wait queue. In high contention scenarios (many threads contending for the same futex), this design minimizes the overhead by limiting the scope of locking, ensuring that only critical sections (like queue manipulation) are protected.
    - The futex mechanism efficiently transitions between user space and kernel space, reducing the number of expensive kernel calls when threads can handle contention in user space.
2. **Scalability**:
    
    - The futex mechanism scales well because it reduces the need for kernel space operations when the futex can be acquired immediately. In multi-core environments, this design minimizes bottlenecks, allowing threads to operate in parallel without significant overhead from kernel interactions.

#### Potential Issues and Race Conditions

1. **Race Condition Mitigation**:
    
    - A critical part of the futex design is the re-check of the futex state before blocking the thread. This check ensures that a thread does not go to sleep if the futex has been released by another thread between the time the thread checked the futex state and the time it attempts to block. This mitigates race conditions that could lead to deadlocks or missed wake-ups.
2. **Spinlocks and Atomicity**:
    
    - While the Go implementation uses a mutex for simplicity, the concept of spinlocks in the kernel ensures that futex operations are atomic. Spinlocks are preferred in low-level kernel code because they are more performant for protecting short critical sections. In Go, the `sync.Mutex` achieves atomicity, ensuring that shared data structures like the wait queues are manipulated safely without race conditions.