### 1. Overview
- **Summary:** This chapter focuses on building custom synchronization primitives such as mutexes, condition variables, and reader-writer locks using Rust. We will implement these locks from scratch to understand their inner workings and improve our knowledge of concurrency in Rust.
- **Key Takeaways:**
  - Custom implementation of synchronization primitives enhances understanding.
  - Using platform-specific functionalities like futex (fast user-space mutex) for efficiency.
  - Leveraging the `atomic-wait` crate for cross-platform support.
---
### 2. Detailed Analysis

#### 2.1 Tools and Setup
**Using the `atomic-wait` crate:**
- Add `atomic-wait = "1"` to `Cargo.toml`.
- Import the functions: `wait`, `wake_one`, `wake_all`.

```toml
[dependencies]
atomic-wait = "1"
```

```rust
use atomic_wait::{wait, wake_one, wake_all};
use std::sync::atomic::{AtomicU32, Ordering};
```

#### 2.2 Building a Basic Mutex

**Concept:**
- A mutex ensures mutual exclusion, allowing only one thread to access the critical section at a time.

**Implementation:**

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use atomic_wait::{wait, wake_one};

pub struct MyMutex {
    lock: AtomicU32,
}

impl MyMutex {
    pub fn new() -> Self {
        MyMutex {
            lock: AtomicU32::new(0),
        }
    }

    pub fn lock(&self) {
        while self.lock.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            wait(&self.lock, 1);
        }
    }

    pub fn unlock(&self) {
        self.lock.store(0, Ordering::Release);
        wake_one(&self.lock);
    }
}
```

- **Explanation:**
  - The `lock` method tries to acquire the lock using `compare_exchange`.
  - If the lock is already held, it waits using the `wait` function.
  - The `unlock` method releases the lock and wakes up a waiting thread.

**Usage Example:**

```rust
use std::sync::Arc;
use std::thread;

let mutex = Arc::new(MyMutex::new());

let handles: Vec<_> = (0..10).map(|i| {
    let mutex = Arc::clone(&mutex);
    thread::spawn(move || {
        mutex.lock();
        println!("Thread {} has acquired the lock", i);
        mutex.unlock();
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

---
#### 2.3 Condition Variables
**Concept:**
- Condition variables allow threads to wait for certain conditions to become true.
**Implementation:**

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use atomic_wait::{wait, wake_one, wake_all};
use std::sync::Mutex;

pub struct MyCondVar {
    waiters: AtomicU32,
}

impl MyCondVar {
    pub fn new() -> Self {
        MyCondVar {
            waiters: AtomicU32::new(0),
        }
    }

    pub fn wait(&self, mutex: &MyMutex) {
        self.waiters.fetch_add(1, Ordering::SeqCst);
        mutex.unlock();
        wait(&self.waiters, 1);
        mutex.lock();
    }

    pub fn notify_one(&self) {
        if self.waiters.load(Ordering::SeqCst) > 0 {
            wake_one(&self.waiters);
        }
    }

    pub fn notify_all(&self) {
        wake_all(&self.waiters);
    }
}
```

- **Explanation:**
  - The `wait` method increments the number of waiters, releases the mutex, and waits.
  - The `notify_one` method wakes up one waiting thread.
  - The `notify_all` method wakes up all waiting threads.

**Usage Example:**

```rust
let mutex = Arc::new(MyMutex::new());
let condvar = Arc::new(MyCondVar::new());

let handles: Vec<_> = (0..10).map(|i| {
    let mutex = Arc::clone(&mutex);
    let condvar = Arc::clone(&condvar);
    thread::spawn(move || {
        mutex.lock();
        condvar.wait(&mutex);
        println!("Thread {} was notified", i);
        mutex.unlock();
    })
}).collect();

thread::sleep(Duration::from_secs(1));
condvar.notify_all();

for handle in handles {
    handle.join().unwrap();
}
```
---
#### 2.4 Reader-Writer Lock

**Concept:**
- A reader-writer lock allows multiple readers or one writer at a time.
**Implementation:**
```rust
use std::sync::atomic::{AtomicU32, Ordering};
use atomic_wait::{wait, wake_all};

pub struct MyRwLock {
    readers: AtomicU32,
    writer: AtomicU32,
}

impl MyRwLock {
    pub fn new() -> Self {
        MyRwLock {
            readers: AtomicU32::new(0),
            writer: AtomicU32::new(0),
        }
    }

    pub fn read_lock(&self) {
        while self.writer.load(Ordering::SeqCst) != 0 {
            wait(&self.writer, 1);
        }
        self.readers.fetch_add(1, Ordering::SeqCst);
    }

    pub fn read_unlock(&self) {
        if self.readers.fetch_sub(1, Ordering::SeqCst) == 1 {
            wake_all(&self.writer);
        }
    }

    pub fn write_lock(&self) {
        while self.writer.compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed).is_err() {
            wait(&self.writer, 1);
        }
        while self.readers.load(Ordering::SeqCst) != 0 {
            wait(&self.readers, 0);
        }
    }

    pub fn write_unlock(&self) {
        self.writer.store(0, Ordering::SeqCst);
        wake_all(&self.writer);
    }
}
```

- **Explanation:**
  - The `read_lock` method waits for any writer to finish and increments the reader count.
  - The `read_unlock` method decrements the reader count and wakes writers if no readers are left.
  - The `write_lock` method waits for any writer to finish and ensures no readers are active.
  - The `write_unlock` method releases the writer lock and wakes up waiting threads.

**Usage Example:**

```rust
let rwlock = Arc::new(MyRwLock::new());

let readers: Vec<_> = (0..5).map(|i| {
    let rwlock = Arc::clone(&rwlock);
    thread::spawn(move || {
        rwlock.read_lock();
        println!("Reader {} acquired the lock", i);
        rwlock.read_unlock();
    })
}).collect();

let writers: Vec<_> = (0..2).map(|i| {
    let rwlock = Arc::clone(&rwlock);
    thread::spawn(move || {
        rwlock.write_lock();
        println!("Writer {} acquired the lock", i);
        rwlock.write_unlock();
    })
}).collect();

for handle in readers.into_iter().chain(writers) {
    handle.join().unwrap();
}
```

---

### 3. Summary

- **Custom Locks:** Implementing custom mutexes, condition variables, and reader-writer locks deepens understanding of concurrency.
- **Platform-Specific Tools:** Using the `atomic-wait` crate abstracts away platform-specific details.
- **Practical Examples:** Demonstrate the usage and importance of each synchronization primitive in real-world scenarios.
