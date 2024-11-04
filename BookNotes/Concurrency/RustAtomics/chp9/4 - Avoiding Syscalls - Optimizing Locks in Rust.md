### Introduction

Optimizing locking primitives is mainly about avoiding unnecessary wait and wake operations. In this chapter, we explore various strategies to achieve this for condition variables, mutexes, and other synchronization primitives. The primary focus is on reducing syscall overhead and handling edge cases effectively.

---

### Condition Variable Optimization

#### Basic Concept

A condition variable allows threads to sleep until they are notified. It typically works with a mutex to protect shared data. Here’s a minimal implementation:

```rust
pub struct Condvar {
    counter: AtomicU32,
}

impl Condvar {
    pub const fn new() -> Self {
        Self { counter: AtomicU32::new(0) }
    }
}
```

#### Avoiding Unnecessary Wakes

To avoid unnecessary wake operations, we can keep track of the number of waiting threads. If there are no waiting threads, we skip the wake operation.

```rust
pub struct Condvar {
    counter: AtomicU32,
    num_waiters: AtomicUsize, // New field to track waiters
}

impl Condvar {
    pub const fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
            num_waiters: AtomicUsize::new(0),
        }
    }

    pub fn notify_one(&self) {
        if self.num_waiters.load(Ordering::Relaxed) > 0 {
            self.counter.fetch_add(1, Ordering::Relaxed);
            wake_one(&self.counter);
        }
    }

    pub fn notify_all(&self) {
        if self.num_waiters.load(Ordering::Relaxed) > 0 {
            self.counter.fetch_add(1, Ordering::Relaxed);
            wake_all(&self.counter);
        }
    }

    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        self.num_waiters.fetch_add(1, Ordering::Relaxed);
        let counter_value = self.counter.load(Ordering::Relaxed);
        let mutex = guard.mutex;
        drop(guard);
        wait(&self.counter, counter_value);
        self.num_waiters.fetch_sub(1, Ordering::Relaxed);
        mutex.lock()
    }
}
```

### Detailed Analysis

#### Memory Ordering

- **Relaxed Ordering**: We use relaxed ordering because the happens-before relationship is established by the mutex operations (lock and unlock).

#### Handling Spurious Wakeups

A thread might wake up without a corresponding signal. The condition variable's wait method will relock the mutex before returning, ensuring the thread continues only when the condition is met.

### Avoiding Spurious Wake-ups

Every time a thread is woken up, it’ll try to lock the mutex, potentially competing with other threads, which can significantly impact performance. To reduce this, we can track the number of threads allowed to wake up.

```rust
pub fn notify_one(&self) {
    if self.num_waiters.load(Ordering::Relaxed) > 0 {
        self.counter.fetch_add(1, Ordering::Relaxed);
        wake_one(&self.counter);
    }
}

pub fn notify_all(&self) {
    if self.num_waiters.load(Ordering::Relaxed) > 0 {
        self.counter.fetch_add(1, Ordering::Relaxed);
        wake_all(&self.counter);
    }
}
```

### Thundering Herd Problem

The thundering herd problem occurs when `notify_all()` wakes up many threads, all of which try to lock the same mutex. Only one succeeds, and the rest go back to sleep, wasting resources.

#### Mitigation Strategy

On operating systems that support a futex-like requeuing operation (e.g., `FUTEX_REQUEUE` on Linux), we can requeue all but one thread to wait on the mutex state instead.

```rust
impl Condvar {
    pub fn notify_all(&self) {
        if self.num_waiters.load(Ordering::Relaxed) > 0 {
            self.counter.fetch_add(1, Ordering::Relaxed);
            wake_all(&self.counter);
        }
    }
}
```

### Example Implementation

#### Basic Mutex Implementation

Here is an optimized `Mutex` implementation that avoids unnecessary syscalls by using spinning and tracking the number of waiting threads.

```rust
pub struct Mutex<T> {
    state: AtomicU32,
    value: UnsafeCell<T>,
}

impl<T> Mutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            state: AtomicU32::new(0),
            value: UnsafeCell::new(value),
        }
    }

    pub fn lock(&self) -> MutexGuard<T> {
        if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            self.lock_contended();
        }
        MutexGuard { mutex: self }
    }

    fn lock_contended(&self) {
        while self.state.swap(2, Ordering::Acquire) != 0 {
            wait(&self.state, 2);
        }
    }
}

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

impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        if self.mutex.state.swap(0, Ordering::Release) == 2 {
            wake_one(&self.mutex.state);
        }
    }
}
```

#### Optimizing the Lock Contention

To further optimize, we can use spinning for a short period before calling `wait()`.

```rust
fn lock_contended(state: &AtomicU32) {
    let mut spin_count = 0;
    while state.load(Ordering::Relaxed) == 1 && spin_count < 100 {
        spin_count += 1;
        std::hint::spin_loop();
    }
    if state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
        return;
    }
    while state.swap(2, Ordering::Acquire) != 0 {
        wait(state, 2);
    }
}
```

### Conclusion

Optimizing synchronization primitives in Rust involves minimizing syscall overhead and handling edge cases efficiently. By using spinning, tracking waiting threads, and leveraging platform-specific features like futex requeueing, we can create highly performant and robust condition variables and mutexes.

### Visualizations

#### Mutex State Transitions

```plaintext
State Transitions:
0 - Unlocked
1 - Locked, no waiters
2 - Locked, with waiters

Thread A (acquiring lock):
----------------------------------
state = 0 -> state = 1 (locked)

Thread B (waiting for lock):
----------------------------------
state = 1 -> state = 2 (locked, with waiters)
```

#### Condition Variable Wait Flow

```plaintext
Thread A (waiting):
----------------------------------
lock(mutex)
while !condition {
    condvar.wait(mutex)
}
unlock(mutex)

Thread B (notifying):
----------------------------------
lock(mutex)
modify(data)
unlock(mutex)
condvar.notify_one()
```

### Equations

#### Spin Loop Duration Calculation

```plaintext
Spin Duration (T_spin) = spin_count * T_iteration

Where:
- T_spin is the total spin duration.
- spin_count is the number of spin iterations.
- T_iteration is the time taken for a single spin iteration.
```

### Code Example

#### Condition Variable Test

```rust
#[test]
fn test_condvar() {
    let mutex = Mutex::new(0);
    let condvar = Condvar::new();
    let mut wakeups = 0;
    thread::scope(|s| {
        s.spawn(|| {
            thread::sleep(Duration::from_secs(1));
            *mutex.lock() = 123;
            condvar.notify_one();
        });
        let mut m = mutex.lock();
        while *m < 100 {
            m = condvar.wait(m);
            wakeups += 1;
        }
        assert_eq!(*m, 123);
    });
    assert!(wakeups < 10);
}
```

This test ensures that the condition variable correctly puts the thread to sleep and wakes it up based on signals, confirming its correct behavior.