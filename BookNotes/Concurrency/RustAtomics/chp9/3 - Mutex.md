### Mutex

In this section, we will build a custom `Mutex<T>` from scratch. We'll use our `SpinLock<T>` as a reference and make necessary changes to implement blocking using the `atomic-wait` crate.

---
### 1. Type Definition

First, we define our `Mutex<T>` struct. Instead of using an `AtomicBool`, we use an `AtomicU32` to represent the lock state (0 for unlocked, 1 for locked). This allows us to use the atomic wait and wake functions.

```rust
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};
use atomic_wait::{wait, wake_one};

pub struct Mutex<T> {
    /// 0: unlocked
    /// 1: locked
    state: AtomicU32,
    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for Mutex<T> {}
```

### 2. MutexGuard

The `MutexGuard` type will ensure safe access to the protected value. It implements `Deref` and `DerefMut` traits to provide access to the inner value.

```rust
use std::ops::{Deref, DerefMut};

pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.value.get() }
    }
}
```

### 3. Constructor

The `Mutex::new` function initializes the mutex in an unlocked state with the given value.

```rust
impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicU32::new(0), // unlocked state
            value: UnsafeCell::new(value),
        }
    }
}
```

### 4. Locking the Mutex

The `lock` function attempts to acquire the lock. If the lock is already held, it waits until it is released.

```rust
impl<T> Mutex<T> {
    pub fn lock(&self) -> MutexGuard<T> {
        while self.state.swap(1, Ordering::Acquire) == 1 {
            wait(&self.state, 1);
        }
        MutexGuard { mutex: self }
    }
}
```

### 5. Unlocking the Mutex

The `Drop` implementation for `MutexGuard` is responsible for unlocking the mutex and waking one waiting thread if there are any.

```rust
impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.state.store(0, Ordering::Release);
        wake_one(&self.mutex.state);
    }
}
```

### 6. Optimizing for Waiting Threads

To optimize the mutex, we will track whether there are waiting threads by using a separate state value.

**Updated Type Definition:**

```rust
pub struct Mutex<T> {
    /// 0: unlocked
    /// 1: locked, no other threads waiting
    /// 2: locked, other threads waiting
    state: AtomicU32,
    value: UnsafeCell<T>,
}
```

**Updated Lock Function:**

```rust
impl<T> Mutex<T> {
    pub fn lock(&self) -> MutexGuard<T> {
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            while self.state.swap(2, Ordering::Acquire) != 0 {
                wait(&self.state, 2);
            }
        }
        MutexGuard { mutex: self }
    }
}
```

**Updated Unlock Function:**

```rust
impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        if self.mutex.state.swap(0, Ordering::Release) == 2 {
            wake_one(&self.mutex.state);
        }
    }
}
```

### 7. Visualization

**ASCII Visualization of Mutex State Transitions:**

```
+---------+      +--------+      +---------+
| Unlocked| ---> | Locked | ---> | Unlocked|
|   (0)   |      |   (1)  |      |   (0)   |
+---------+      +--------+      +---------+
     ^               |                |
     |               v                v
+--------+ <----- + Locked + ------> + Wait   +
| Wait   |        | & Wait  |        | Wake   |
| (2)    |        |  (2)    |        | One    |
+--------+        +---------+        +--------+
```

- **State 0:** Unlocked
- **State 1:** Locked without waiting threads
- **State 2:** Locked with waiting threads

### 8. Summary

- **Custom Mutex Implementation:** Building a custom mutex helps understand the intricacies of lock mechanisms and thread synchronization.
- **Atomic Operations:** Efficient use of atomic operations like `compare_exchange`, `swap`, `wait`, and `wake_one`.
- **Optimization:** Minimizing syscalls by effectively managing the lock state and only waking up necessary threads.

---
### Optimizing Further

#### Overview

In this section, we explore advanced optimization techniques for our `Mutex` implementation. We aim to reduce system call overhead by combining spinning and blocking mechanisms, enhancing performance in various contention scenarios.

---

### 1. Combining Spin Lock and Blocking

Spinning avoids the overhead of system calls by actively checking the lock state, suitable for short waits. However, it can be inefficient if the lock is held for longer periods. We combine spinning with blocking to leverage the benefits of both approaches.

#### Implementation Strategy:

- **Spin for a short time:** If the lock is likely to be released soon, avoid system calls by spinning.
- **Block if necessary:** If the lock is not released within the spin duration, use wait.

#### Code Example:

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use std::cell::UnsafeCell;
use atomic_wait::{wait, wake_one};

pub struct Mutex<T> {
    /// 0: unlocked
    /// 1: locked, no other threads waiting
    /// 2: locked, other threads waiting
    state: AtomicU32,
    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for Mutex<T> {}

pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.value.get() }
    }
}

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicU32::new(0), // unlocked state
            value: UnsafeCell::new(value),
        }
    }

    pub fn lock(&self) -> MutexGuard<T> {
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            // The lock was already locked. Handle contention.
            self.lock_contended();
        }
        MutexGuard { mutex: self }
    }

    fn lock_contended(&self) {
        let mut spin_count = 0;
        while self.state.load(Ordering::Relaxed) == 1 && spin_count < 100 {
            spin_count += 1;
            std::hint::spin_loop();
        }
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            return;
        }
        while self.state.swap(2, Ordering::Acquire) != 0 {
            wait(&self.state, 2);
        }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        if self.mutex.state.swap(0, Ordering::Release) == 2 {
            wake_one(&self.mutex.state);
        }
    }
}
```

#### ASCII Visualization of Spin and Block Mechanism:

```
+----------------+    +---------------+    +------------------+
| Initial State  |    |   Spinning    |    |    Blocking      |
| (state = 0)    | -> | (state = 1)   | -> | (state = 2)      |
+----------------+    +---------------+    +------------------+
      ^                     ^                       ^
      |                     |                       |
      +---------------------+-----------------------+
                Spin Count Exceeded or Lock Acquired
```

---

### 2. Performance Optimizations

#### Avoiding Unnecessary System Calls

To avoid unnecessary system calls, we introduce a third state (2) to track if there are waiting threads. This helps minimize `wake_one` calls when no threads are waiting.

#### Updated Mutex Definition:

```rust
pub struct Mutex<T> {
    /// 0: unlocked
    /// 1: locked, no other threads waiting
    /// 2: locked, other threads waiting
    state: AtomicU32,
    value: UnsafeCell<T>,
}
```

#### Optimized Lock and Unlock:

```rust
impl<T> Mutex<T> {
    pub fn lock(&self) -> MutexGuard<T> {
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            self.lock_contended();
        }
        MutexGuard { mutex: self }
    }

    fn lock_contended(&self) {
        let mut spin_count = 0;
        while self.state.load(Ordering::Relaxed) == 1 && spin_count < 100 {
            spin_count += 1;
            std::hint::spin_loop();
        }
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            return;
        }
        while self.state.swap(2, Ordering::Acquire) != 0 {
            wait(&self.state, 2);
        }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        if self.mutex.state.swap(0, Ordering::Release) == 2 {
            wake_one(&self.mutex.state);
        }
    }
}
```

---

### 3. Performance Benchmarking

#### Uncontended Benchmark

Measure performance in a single-threaded scenario where the mutex is locked and unlocked repeatedly.

```rust
use std::time::Instant;
use std::hint::black_box;

fn main() {
    let m = Mutex::new(0);
    black_box(&m);
    let start = Instant::now();
    for _ in 0..5_000_000 {
        *m.lock() += 1;
    }
    let duration = start.elapsed();
    println!("Locked 5,000,000 times in {:?}", duration);
}
```

#### Contended Benchmark

Measure performance in a multi-threaded scenario with high contention.

```rust
use std::thread;
use std::time::Instant;
use std::hint::black_box;

fn main() {
    let m = Mutex::new(0);
    black_box(&m);
    let start = Instant::now();
    thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..5_000_000 {
                    *m.lock() += 1;
                }
            });
        }
    });
    let duration = start.elapsed();
    println!("Locked 20,000,000 times in {:?}", duration);
}
```

### Results and Analysis

- **Uncontended Case:** Significant performance improvement observed in the 3-state mutex compared to the 2-state version.
- **Contended Case:** Mixed results depending on hardware and platform, showing the complexity and variability in real-world scenarios.

#### Conclusion

- **Spinning vs. Blocking:** Spinning can provide performance benefits for short wait times but may be less efficient for longer waits.
- **Platform-Dependent:** Performance gains can vary significantly across different platforms and hardware configurations.

---

### Summary

- **Advanced Mutex Implementation:** Combining spinning and blocking for efficient locking.
- **Performance Optimization:** Reduced unnecessary system calls using a third state and strategic spinning.
- **Benchmarking Insights:** Real-world performance varies, requiring careful consideration of specific use cases.

These detailed notes provide an in-depth understanding of advanced mutex optimizations in Rust, tailored for a Staff+ level engineer, focusing on practical implementation, performance analysis, and platform-specific considerations.