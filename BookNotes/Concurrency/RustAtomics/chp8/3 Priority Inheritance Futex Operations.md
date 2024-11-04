## Introduction

Priority inversion is a common problem in concurrent programming where a high-priority thread is blocked waiting for a resource held by a lower-priority thread. To mitigate this issue, priority inheritance mechanisms can temporarily elevate the priority of the lower-priority thread holding the resource. In Linux, priority inheritance is supported by a set of futex operations that extend the basic futex functionality.

### Overview

The priority inheritance futex operations include six specific syscalls designed to implement priority inheriting locks. These operations ensure that a low-priority thread holding a lock can inherit the highest priority of the waiting threads, reducing priority inversion issues.

### Data Representation

For priority inheritance futex operations, the 32-bit atomic variable's contents are standardized:
- The highest bit indicates whether any threads are waiting.
- The lowest 30 bits contain the thread ID (Linux TID) of the thread holding the lock or zero if the lock is available.
- The second highest bit is set if the thread holding the lock terminates unexpectedly, making the mutex robust.

### Priority Inheritance Futex Operations

1. **FUTEX_LOCK_PI**: Locks the mutex with priority inheritance.
2. **FUTEX_UNLOCK_PI**: Unlocks the mutex with priority inheritance.
3. **FUTEX_TRYLOCK_PI**: Attempts to lock the mutex without blocking.
4. **FUTEX_CMP_REQUEUE_PI**: Requeues waiting threads from one futex to another with priority inheritance.
5. **FUTEX_WAIT_REQUEUE_PI**: Waits on a futex and requeues to another futex with priority inheritance.

#### FUTEX_LOCK_PI
- **Description**: Locks a mutex and inherits the priority of waiting threads.
- **Arguments**:
  - Pointer to the futex variable.
  - Operation type: `FUTEX_LOCK_PI`.
  - Optional timeout for the lock attempt.
  
```rust
pub fn futex_lock_pi(futex: &AtomicU32, timeout: Option<&libc::timespec>) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_LOCK_PI,
            timeout.map_or(ptr::null(), |t| t as *const libc::timespec)
        ) as i32
    }
}
```

#### FUTEX_UNLOCK_PI
- **Description**: Unlocks a mutex and handles priority inheritance.
- **Arguments**:
  - Pointer to the futex variable.
  - Operation type: `FUTEX_UNLOCK_PI`.

```rust
pub fn futex_unlock_pi(futex: &AtomicU32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_UNLOCK_PI
        ) as i32
    }
}
```

#### FUTEX_TRYLOCK_PI
- **Description**: Attempts to lock a mutex without blocking.
- **Arguments**:
  - Pointer to the futex variable.
  - Operation type: `FUTEX_TRYLOCK_PI`.

```rust
pub fn futex_trylock_pi(futex: &AtomicU32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_TRYLOCK_PI
        ) as i32
    }
}
```

#### FUTEX_CMP_REQUEUE_PI
- **Description**: Requeues waiting threads from one futex to another with priority inheritance.
- **Arguments**:
  - Pointer to the primary futex variable.
  - Operation type: `FUTEX_CMP_REQUEUE_PI`.
  - Number of threads to requeue.
  - Pointer to the secondary futex variable.
  - Expected value of the primary futex variable.

```rust
pub fn futex_cmp_requeue_pi(
    futex: &AtomicU32,
    num_requeue: i32,
    secondary_futex: &AtomicU32,
    expected: u32
) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_CMP_REQUEUE_PI,
            num_requeue,
            secondary_futex as *const AtomicU32,
            expected
        ) as i32
    }
}
```

#### FUTEX_WAIT_REQUEUE_PI
- **Description**: Waits on a futex and requeues to another futex with priority inheritance.
- **Arguments**:
  - Pointer to the futex variable.
  - Operation type: `FUTEX_WAIT_REQUEUE_PI`.
  - Expected value of the futex variable.
  - Pointer to a `timespec` struct for the maximum wait duration (nullable for no timeout).
  - Pointer to the secondary futex variable.

```rust
pub fn futex_wait_requeue_pi(
    futex: &AtomicU32,
    expected: u32,
    timeout: Option<&libc::timespec>,
    secondary_futex: &AtomicU32
) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_WAIT_REQUEUE_PI,
            expected,
            timeout.map_or(ptr::null(), |t| t as *const libc::timespec),
            secondary_futex as *const AtomicU32
        ) as i32
    }
}
```

## Practical Example

Let's implement a simple priority inheriting mutex using these futex operations.

### Priority Inheriting Mutex

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

pub struct PriorityInheritingMutex {
    futex: AtomicU32,
}

impl PriorityInheritingMutex {
    pub const fn new() -> Self {
        Self {
            futex: AtomicU32::new(0),
        }
    }

    pub fn lock(&self) {
        while futex_lock_pi(&self.futex, None) != 0 {
            // Retry on failure
        }
    }

    pub fn unlock(&self) {
        while futex_unlock_pi(&self.futex) != 0 {
            // Retry on failure
        }
    }
}

fn futex_lock_pi(futex: &AtomicU32, timeout: Option<&libc::timespec>) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_LOCK_PI,
            timeout.map_or(ptr::null(), |t| t as *const libc::timespec)
        ) as i32
    }
}

fn futex_unlock_pi(futex: &AtomicU32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_UNLOCK_PI
        ) as i32
    }
}

fn main() {
    let mutex = PriorityInheritingMutex::new();

    let handle = thread::spawn(move || {
        mutex.lock();
        println!("Thread 1 acquired the lock.");
        thread::sleep(Duration::from_secs(2));
        mutex.unlock();
        println!("Thread 1 released the lock.");
    });

    thread::sleep(Duration::from_secs(1));

    let handle2 = thread::spawn(move || {
        mutex.lock();
        println!("Thread 2 acquired the lock.");
        mutex.unlock();
        println!("Thread 2 released the lock.");
    });

    handle.join().unwrap();
    handle2.join().unwrap();
}
```

## Conclusion

Priority inheritance futex operations provide an effective way to mitigate priority inversion problems in concurrent programming. By understanding and utilizing these operations, we can build high-performance synchronization primitives that ensure efficient thread management and resource allocation. These operations are especially critical in real-time systems where meeting priority constraints is essential.

---- ATTEMPT 2

----

# Priority Inheritance Futex Operations

Priority inversion is a common problem in real-time systems where a high-priority thread is blocked waiting for a lock held by a low-priority thread. This scenario effectively inverts the priorities because the high-priority thread is forced to wait for the low-priority thread to release the lock. 

Priority inheritance is a mechanism designed to solve this problem. It temporarily raises the priority of the thread holding the lock to the highest priority of any thread waiting for it. This ensures that the low-priority thread will execute and release the lock sooner, allowing the high-priority thread to proceed.

## Priority Inheritance Futex Operations

Linux provides several futex operations designed specifically for implementing priority inheritance locks. These operations require a specific format for the 32-bit atomic variable to allow the kernel to understand the lock's state.

### Atomic Variable Format

- **Highest bit**: Indicates if there are threads waiting to lock the mutex.
- **Second highest bit**: Indicates if the thread holding the lock has terminated unexpectedly (set only if there are waiters).
- **Lowest 30 bits**: Contain the thread ID (Linux TID) of the thread holding the lock, or zero if the lock is not held.

### Priority Inheriting Futex Operations

1. **FUTEX_LOCK_PI**: Lock the mutex with priority inheritance.
2. **FUTEX_UNLOCK_PI**: Unlock the mutex with priority inheritance.
3. **FUTEX_TRYLOCK_PI**: Try to lock the mutex without blocking.
4. **FUTEX_CMP_REQUEUE_PI**: Requeue waiters with priority inheritance.
5. **FUTEX_WAIT_REQUEUE_PI**: Wait for requeue with priority inheritance.

These operations provide the necessary functionality to implement priority inheriting mutexes, allowing for efficient and correct handling of priority inversion.

## macOS

macOS offers various low-level concurrency-related syscalls through system libraries. The kernel interface is not stable and should not be used directly. Instead, interactions with the kernel should be through system-provided libraries, such as libc, libc++, and others.

### Pthread Primitives

macOS includes a full pthread implementation, offering synchronization primitives like mutexes, reader-writer locks, and condition variables. However, pthread locks on macOS are relatively slow due to their fairness properties, which ensure that threads are served in order of arrival.

### os_unfair_lock

Introduced in macOS 10.12, `os_unfair_lock` is a lightweight, platform-specific mutex that does not ensure fairness. It is 32 bits in size, initialized with `OS_UNFAIR_LOCK_INIT`, and does not require destruction. It can be locked (`os_unfair_lock_lock`) and unlocked (`os_unfair_lock_unlock`) efficiently but lacks condition variables and reader-writer variants.

## Windows

Windows provides synchronization primitives through the Win32 API, accessible via libraries such as kernel32.dll. Rust programs can use these through Microsoft's `windows` and `windows-sys` crates.

### Heavyweight Kernel Objects

These include Mutex, Event, and WaitableTimer, managed fully by the kernel. They are suitable for cross-process synchronization and support fine-grained permissions. Creating these objects results in a `HANDLE`, which can be managed with various wait functions.

### Lighter-Weight Objects

- **CRITICAL_SECTION**: A recursive mutex referred to as a critical section, which a single thread can lock multiple times. It is initialized with `InitializeCriticalSection` and must not be moved. It is locked with `EnterCriticalSection` and unlocked with `LeaveCriticalSection`.
- **Slim Reader-Writer Locks (SRW Lock)**: Introduced in Windows Vista, these are lightweight and efficient. They can be initialized statically with `SRWLOCK_INIT` and do not require destruction. SRW locks provide exclusive and shared locking without prioritizing either. They are commonly used with condition variables (`SleepConditionVariableSRW`).

### Address-Based Waiting

Windows 8 introduced `WaitOnAddress`, similar to Linux's `FUTEX_WAIT` and `FUTEX_WAKE`. It allows waiting on an atomic variable and waking threads based on the variable's address. This is used to build efficient synchronization primitives.

### Example of `WaitOnAddress`

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Duration;
use windows::Win32::System::Threading::{
    WaitOnAddress, WakeByAddressSingle, 
    WAKE_BY_ADDRESS_ALL
};

static DATA: AtomicU32 = AtomicU32::new(0);

fn main() {
    let a = thread::spawn(|| {
        thread::sleep(Duration::from_secs(1));
        DATA.store(1, Ordering::Relaxed);
        unsafe { WakeByAddressSingle(&DATA as *const _ as *const _) };
    });

    let b = thread::spawn(|| {
        unsafe { 
            WaitOnAddress(&DATA as *const _ as *const _, &0, std::mem::size_of::<u32>(), WAKE_BY_ADDRESS_ALL); 
        }
        println!("Data: {}", DATA.load(Ordering::Relaxed));
    });

    a.join().unwrap();
    b.join().unwrap();
}
```

This example shows how `WaitOnAddress` and `WakeByAddressSingle` can be used to implement a simple wait-wake mechanism in Windows.

## Conclusion

Understanding and utilizing priority inheritance and various platform-specific synchronization primitives is essential for building efficient and correct concurrent programs. Each operating system provides unique mechanisms and primitives, requiring careful consideration when designing cross-platform applications.