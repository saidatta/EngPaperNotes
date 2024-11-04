## Introduction
In this section, we'll discuss various futex operations supported by the futex syscall on Linux. Futex (Fast Userspace Mutex) operations provide efficient mechanisms for thread synchronization by allowing threads to wait on or wake up based on changes to a 32-bit atomic variable.
## Basic Futex Operations
### Overview
The `futex` syscall supports multiple operations. The first argument is always a pointer to the 32-bit atomic variable to operate on. The second argument is a constant representing the operation, such as `FUTEX_WAIT`. The remaining arguments depend on the operation. Optional flags include `FUTEX_PRIVATE_FLAG` and `FUTEX_CLOCK_REALTIME`.
### FUTEX_WAIT
This operation blocks the thread if the atomic variable's value matches the expected value. It takes two additional arguments: the expected value and a pointer to a `timespec` representing the maximum time to wait.
- **Parameters**: 
  - `expected`: The value the atomic variable is expected to have.
  - `timespec`: Pointer to a `timespec` struct for the maximum wait duration (nullable for no timeout).
- **Behavior**: 
  - If the atomic variable matches the expected value, the thread blocks until woken up or the timeout is reached.
  - May spuriously wake up without a corresponding wake operation.
- **Return Value**: Indicates whether the expected value matched and whether the timeout was reached.
```rust
pub fn futex_wait(a: &AtomicU32, expected: u32, timeout: Option<&libc::timespec>) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAIT,
            expected,
            timeout.map_or(ptr::null(), |t| t as *const libc::timespec),
        ) as i32
    }
}
```
### FUTEX_WAKE
This operation wakes up a specified number of threads blocked on the same atomic variable.
- **Parameters**: 
  - `num_threads`: Number of threads to wake up.
- **Behavior**: Wakes up the specified number of threads waiting on the atomic variable.
- **Return Value**: Number of threads actually woken up.
```rust
pub fn futex_wake(a: &AtomicU32, num_threads: i32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAKE,
            num_threads,
        ) as i32
    }
}
```
## Advanced Futex Operations
### FUTEX_WAIT_BITSET

This operation is similar to `FUTEX_WAIT`, but includes a 32-bit bitset to selectively wait for specific wake operations.

- **Parameters**: 
  - `expected`: The value the atomic variable is expected to have.
  - `timespec`: Pointer to a `timespec` struct for the maximum wait duration (nullable for no timeout).
  - `bitset`: 32-bit bitset to specify which wake operations to respond to.

- **Behavior**: 
  - If the bitset of the `FUTEX_WAIT_BITSET` and `FUTEX_WAKE_BITSET` operations do not overlap, the signal is ignored.
  - The timespec represents an absolute timestamp.

- **Return Value**: Indicates whether the expected value matched and whether the timeout was reached.

```rust
pub fn futex_wait_bitset(a: &AtomicU32, expected: u32, timeout: Option<&libc::timespec>, bitset: u32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAIT_BITSET,
            expected,
            timeout.map_or(ptr::null(), |t| t as *const libc::timespec),
            ptr::null::<libc::timespec>(),
            bitset,
        ) as i32
    }
}
```

### FUTEX_WAKE_BITSET

This operation is similar to `FUTEX_WAKE`, but uses a bitset to selectively wake specific wait operations.

- **Parameters**: 
  - `num_threads`: Number of threads to wake up.
  - `bitset`: 32-bit bitset to specify which wake operations to respond to.

- **Behavior**: Wakes up the specified number of threads if their bitset matches the wake bitset.

- **Return Value**: Number of threads actually woken up.

```rust
pub fn futex_wake_bitset(a: &AtomicU32, num_threads: i32, bitset: u32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAKE_BITSET,
            num_threads,
            ptr::null::<libc::timespec>(),
            ptr::null::<libc::timespec>(),
            bitset,
        ) as i32
    }
}
```

### FUTEX_REQUEUE

This operation wakes a specified number of threads and requeues the remaining threads to wait on a different atomic variable.

- **Parameters**: 
  - `num_wake`: Number of threads to wake up.
  - `num_requeue`: Number of threads to requeue.
  - `secondary_addr`: Address of the secondary atomic variable.

- **Behavior**: 
  - Wakes the specified number of threads.
  - Requeues the remaining threads to wait on the secondary atomic variable.

- **Return Value**: Number of threads actually woken up.

```rust
pub fn futex_requeue(a: &AtomicU32, num_wake: i32, num_requeue: i32, secondary: &AtomicU32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_REQUEUE,
            num_wake,
            num_requeue,
            secondary as *const AtomicU32,
        ) as i32
    }
}
```

### FUTEX_CMP_REQUEUE

This operation is similar to `FUTEX_REQUEUE` but includes an expected value check before requeueing.

- **Parameters**: 
  - `num_wake`: Number of threads to wake up.
  - `num_requeue`: Number of threads to requeue.
  - `secondary_addr`: Address of the secondary atomic variable.
  - `expected`: Expected value of the primary atomic variable.

- **Behavior**: 
  - If the primary atomic variable matches the expected value, it wakes and requeues the specified threads.
  - The operation is atomic with respect to other futex operations.

- **Return Value**: Sum of the number of threads woken and requeued.

```rust
pub fn futex_cmp_requeue(a: &AtomicU32, num_wake: i32, num_requeue: i32, secondary: &AtomicU32, expected: u32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_CMP_REQUEUE,
            num_wake,
            num_requeue,
            secondary as *const AtomicU32,
            expected,
        ) as i32
    }
}
```

### FUTEX_WAKE_OP

This specialized operation wakes threads on a primary atomic variable and conditionally on a secondary atomic variable.

- **Parameters**: 
  - `num_wake_primary`: Number of threads to wake on the primary atomic variable.
  - `num_wake_secondary`: Number of threads to wake on the secondary atomic variable.
  - `secondary_addr`: Address of the secondary atomic variable.
  - `op`: Encoded 32-bit value for the operation and comparison.

- **Behavior**: 
  - Modifies the secondary atomic variable.
  - Wakes a specified number of threads on the primary atomic variable.
  - Conditionally wakes a specified number of threads on the secondary atomic variable based on the comparison.

- **Return Value**: Total number of threads woken.

```rust
pub fn futex_wake_op(a: &AtomicU32, num_wake_primary: i32, num_wake_secondary: i32, secondary: &AtomicU32, op: u32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAKE_OP,
            num_wake_primary,
            num_wake_secondary,
            secondary as *const AtomicU32,
            op,
        ) as i32
    }
}
```

## Using FUTEX_PRIVATE_FLAG for Optimization

The `FUTEX_PRIVATE_FLAG` can be added to any futex operation to enable optimizations if all futex operations on the atomic variable come from threads within the same process.

```rust
pub const FUTEX_PRIVATE_FLAG: i32 = libc::FUTEX_PRIVATE_FLAG as i32;
```

## Example Usage

Let's combine the above futex operations in a practical example where we implement a basic synchronization primitive.

### Example: Synchronizing Threads with Futexes

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

pub fn futex_wait(a: &AtomicU32, expected: u32, timeout: Option<&libc::timespec>) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAIT,
            expected,
            timeout.map_or

(ptr::null(), |t| t as *const libc::timespec),
        ) as i32
    }
}

pub fn futex_wake_one(a: &AtomicU32) -> i32 {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAKE,
            1,
        ) as i32
    }
}

fn main() {
    let a = AtomicU32::new(0);

    thread::scope(|s| {
        s.spawn(|| {
            thread::sleep(Duration::from_secs(3));
            a.store(1, Ordering::Relaxed);
            futex_wake_one(&a);
        });

        println!("Waiting...");
        while a.load(Ordering::Relaxed) == 0 {
            futex_wait(&a, 0, None);
        }
        println!("Done!");
    });
}
```

## Conclusion

Futex operations provide a powerful and efficient way to manage thread synchronization by leveraging the operating system's capabilities. By understanding and utilizing these operations, we can build high-performance concurrency primitives tailored to specific use cases in Rust.