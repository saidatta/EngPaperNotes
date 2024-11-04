### 1. Overview
- **Summary:** This chapter covers the fundamental concepts of concurrency in Rust, including thread creation, data sharing, and thread safety.
- **Key Takeaways:**
  - Threads are spawned using `std::thread::spawn`.
  - Rust ensures thread safety with ownership and borrowing rules.
  - Common concurrency primitives include `Mutex`, `RwLock`, and `Arc`.
---
### 2. Detailed Analysis
#### 2.1 Thread Basics
**Main Thread and Spawning Threads:**
In Rust, every program starts with a single main thread. New threads can be spawned using `std::thread::spawn`.
```rust
use std::thread;

fn main() {
    thread::spawn(f);
    thread::spawn(f);

    println!("Hello from the main thread.");
}

fn f() {
    println!("Hello from another thread!");
    let id = thread::current().id();
    println!("This is my thread id: {:?}", id);
}
```
**Thread Id:**
- Each thread is assigned a unique identifier (`ThreadId`).
- Accessible via `thread::current().id()`.
---
**Example Output (Non-Deterministic):**
```plaintext
Hello from the main thread.
Hello from another thread!
This is my thread id: ThreadId(3)
Hello from another thread!
This is my thread id: ThreadId(2)
```
**Joining Threads:**
To ensure threads complete before the main thread exits, use the `join` method.
```rust
fn main() {
    let t1 = thread::spawn(f);
    let t2 = thread::spawn(f);

    println!("Hello from the main thread.");

    t1.join().unwrap();
    t2.join().unwrap();
}
```
**Example Output (Deterministic):**
```plaintext
Hello from the main thread.
Hello from another thread!
This is my thread id: ThreadId(3)
Hello from another thread!
This is my thread id: ThreadId(2)
```
---
**Output Locking:**
- `println!` macro uses `std::io::Stdout::lock()` to ensure output is not interleaved.
---
#### 2.2 Passing Closures to Threads

**Move Closures:**
```rust
let numbers = vec![1, 2, 3];

thread::spawn(move || {
    for n in &numbers {
        println!("{n}");
    }
}).join().unwrap();
```
- `move` keyword ensures closure captures ownership.
---
**Returning Values from Threads:**
```rust
let numbers = Vec::from_iter(0..=1000);

let t = thread::spawn(move || {
    let len = numbers.len();
    let sum = numbers.iter().sum::<usize>();
    sum / len  // Return value
});

let average = t.join().unwrap();  // Retrieve value
println!("average: {average}");
```
---
#### 2.3 Thread Builder
- `std::thread::Builder` allows customization of thread attributes (e.g., name, stack size).
```rust
use std::thread;

let builder = thread::Builder::new()
    .name("worker".into())
    .stack_size(32 * 1024);
let handler = builder.spawn(|| {
    println!("Thread with custom attributes");
}).unwrap();

handler.join().unwrap();
```
---
#### 2.4 Scoped Threads
**Scoped Threads:**
```rust
let numbers = vec![1, 2, 3];

thread::scope(|s| {
    s.spawn(|| {
        println!("length: {}", numbers.len());
    });
    s.spawn(|| {
        for n in &numbers {
            println!("{n}");
        }
    });
});
```
- Ensures threads do not outlive the scope they are created in.
---
#### 2.5 Shared Ownership and Reference Counting
**Static Variables:**
```rust
static X: [i32; 3] = [1, 2, 3];

thread::spawn(|| dbg!(&X));
thread::spawn(|| dbg!(&X));
```
**Leaking Box:**
```rust
let x: &'static [i32; 3] = Box::leak(Box::new([1, 2, 3]));

thread::spawn(move || dbg!(x));
thread::spawn(move || dbg!(x));
```
---
**Reference Counting:**
- `Rc` (not thread-safe)
- `Arc` (thread-safe, atomic reference counting)
```rust
use std::sync::Arc;

let a = Arc::new([1, 2, 3]);
let b = a.clone();

thread::spawn(move || dbg!(a));
thread::spawn(move || dbg!(b));
```
---
#### 2.6 Borrowing and Data Races
**Borrowing Rules:**
- Immutable Borrowing (`&T`): Shared access, no mutation.
- Mutable Borrowing (`&mut T`): Exclusive access, allows mutation.
```rust
fn f(a: &i32, b: &mut i32) {
    let before = *a;
    *b += 1;
    let after = *a;
    if before != after {
        x(); // Never happens
    }
}
```
---
#### 2.7 Undefined Behavior and Unsafe Code
**Example of Unsafe Code:**
```rust
let a = [123, 456, 789];
let b = unsafe { a.get_unchecked(index) };
```
**Potential Undefined Behavior:**
```rust
match index {
   0 => x(),
   1 => y(),
   _ => z(index),
}

let a = [123, 456, 789];
let b = unsafe { a.get_unchecked(index) };
```
---
### 3. Key Concepts and Primitives

**Mutex:**
- Provides mutual exclusion to shared data.
```rust
use std::sync::{Arc, Mutex};
use std::thread;

let data = Arc::new(Mutex::new(0));

let threads: Vec<_> = (0..10).map(|_| {
    let data = Arc::clone(&data);
    thread::spawn(move || {
        let mut data = data.lock().unwrap();
        *data += 1;
    })
}).collect();

for t in threads {
    t.join().unwrap();
}

println!("Result: {}", *data.lock().unwrap());
```

**RwLock:**
- Provides more granular read/write locking.

```rust
use std::sync::{Arc, RwLock};
use std::thread;

let data = Arc::new(RwLock::new(0));

let r1 = {
    let data = Arc::clone(&data);
    thread::spawn(move || {
        let r = data.read().unwrap();
        println!("Read: {}", *r);
    })
};

let w = {
    let data = Arc::clone(&data);
    thread::spawn(move || {
        let mut w = data.write().unwrap();
        *w += 1;
        println!("Write: {}", *w);
    })
};

r1.join().unwrap();
w.join().unwrap();
```

---

### 4. Summary

- **Concurrency in Rust:** Rust's concurrency model is built on strong guarantees provided by its ownership and borrowing rules.
- **Thread Safety:** Rust ensures thread safety by default, making concurrent programming safer and more predictable.
- **Primitives and Tools:** Rust provides a rich set of concurrency primitives (`Mutex`, `RwLock`, `Arc`) to facilitate safe and efficient concurrent programming.

---

### Obsidian Notes: Interior Mutability in Rust

---

#### Chapter: Interior Mutability

---

### 1. Overview
- **Summary:** This chapter delves into the concept of interior mutability in Rust, allowing mutation through shared references under controlled conditions. This is crucial for safe concurrency and efficient thread communication.
- **Key Takeaways:**
  - Interior mutability allows shared references to mutate data.
  - Key types include `Cell`, `RefCell`, `Mutex`, `RwLock`, and `Atomic` types.
  - Understanding `Send` and `Sync` traits is essential for thread safety.

---
### 2. Detailed Analysis
#### 2.1 Interior Mutability
**Concept:**
- Interior mutability allows modifying data through shared references (`&T`).
- Different from exclusive borrowing (`&mut T`), which ensures no other borrows.

**Important Types:**
- `Cell<T>`: Allows mutation of `T` through shared references, only for single-threaded contexts.
- `RefCell<T>`: Allows borrowing and mutation with runtime borrow checking, also single-threaded.
- `Mutex<T>` and `RwLock<T>`: Provide interior mutability in multi-threaded contexts.
---
#### 2.2 Cell
**Example:**
```rust
use std::cell::Cell;

fn f(a: &Cell<i32>, b: &Cell<i32>) {
    let before = a.get();
    b.set(b.get() + 1);
    let after = a.get();
    if before != after {
        x(); // might happen
    }
}
```
- `Cell<T>` allows `get` and `set` operations.
- Can only hold `Copy` types.

**Advanced Example with `Vec`:**

```rust
fn f(v: &Cell<Vec<i32>>) {
    let mut v2 = v.take(); // Replaces the contents of the Cell with an empty Vec
    v2.push(1);
    v.set(v2); // Put the modified Vec back
}
```

---

#### 2.3 RefCell

**Borrowing with `RefCell`:**

```rust
use std::cell::RefCell;

fn f(v: &RefCell<Vec<i32>>) {
    v.borrow_mut().push(1); // We can modify the `Vec` directly.
}
```

- `RefCell<T>` provides `borrow` and `borrow_mut` methods.
- Panics at runtime if borrow rules are violated.

---

#### 2.4 Mutex and RwLock

**Mutex:**

```rust
use std::sync::Mutex;

fn main() {
    let n = Mutex::new(0);
    thread::scope(|s| {
        for _ in 0..10 {
            s.spawn(|| {
                let mut guard = n.lock().unwrap();
                for _ in 0..100 {
                    *guard += 1;
                }
            });
        }
    });
    assert_eq!(n.into_inner().unwrap(), 1000);
}
```

- `Mutex<T>` ensures exclusive access.
- Uses `MutexGuard` to access the inner value.

**RwLock:**

```rust
use std::sync::RwLock;

fn main() {
    let data = RwLock::new(0);

    {
        let read_guard = data.read().unwrap();
        println!("Read: {}", *read_guard);
    }

    {
        let mut write_guard = data.write().unwrap();
        *write_guard += 1;
    }

    {
        let read_guard = data.read().unwrap();
        println!("Read: {}", *read_guard);
    }
}
```

- `RwLock<T>` allows multiple readers or one writer.
- Provides `RwLockReadGuard` and `RwLockWriteGuard`.

---

#### 2.5 Atomics

**Atomic Types:**

- `AtomicU32`, `AtomicPtr<T>`, etc.
- Ensure atomic operations for simple data types.

**Example:**

```rust
use std::sync::atomic::{AtomicU32, Ordering};

fn main() {
    let counter = AtomicU32::new(0);

    thread::scope(|s| {
        for _ in 0..10 {
            s.spawn(|| {
                for _ in 0..100 {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    });

    println!("Final counter: {}", counter.load(Ordering::SeqCst));
}
```

- `fetch_add`, `load`, and `store` methods provide atomic operations.
- Ordering ensures memory ordering constraints.

---

#### 2.6 UnsafeCell

**Foundation of Interior Mutability:**

```rust
use std::cell::UnsafeCell;

struct MyCell<T> {
    value: UnsafeCell<T>,
}

impl<T> MyCell<T> {
    fn new(value: T) -> MyCell<T> {
        MyCell {
            value: UnsafeCell::new(value),
        }
    }

    fn get(&self) -> *mut T {
        self.value.get()
    }
}

fn main() {
    let cell = MyCell::new(42);
    unsafe {
        *cell.get() = 43;
    }
    println!("Value: {}", unsafe { *cell.get() });
}
```

- `UnsafeCell<T>` is the core primitive.
- Allows mutable access via raw pointers.

---

#### 2.7 Thread Safety: Send and Sync

**Send Trait:**

- Indicates a type can be transferred between threads.

**Sync Trait:**

- Indicates a type can be shared between threads.

**Example with `PhantomData`:**

```rust
use std::marker::PhantomData;

struct NotSync {
    handle: i32,
    _marker: PhantomData<*const ()>,
}

unsafe impl Send for NotSync {}
```

- `PhantomData` is used to affect trait implementations without actual data.

---

### 3. Summary

- **Interior Mutability:** Provides flexibility in concurrent contexts.
- **Types:** `Cell`, `RefCell`, `Mutex`, `RwLock`, and `Atomic` types each serve specific use cases.
- **Safety:** Rust enforces thread safety via `Send` and `Sync` traits, preventing data races and ensuring safe concurrency.

#### Lock Poisoning
---
### 1. Overview
- **Summary:** This chapter covers the concept of lock poisoning in Rust, a mechanism to handle inconsistent states caused by panics while holding a lock. It explains how `Mutex` and `RwLock` handle lock poisoning, and provides practical examples.
- **Key Takeaways:**
  - Lock poisoning marks a `Mutex` as poisoned if a thread panics while holding the lock.
  - Proper handling of poisoned locks is crucial to maintain data consistency.
  - Understanding the nuances of `MutexGuard` and lock lifetimes can prevent common pitfalls.

---
#### Lock Poisoning
**Concept:**
- **Lock Poisoning:** A mechanism to handle potential inconsistencies caused by a thread panicking while holding a `Mutex`.
- When a `Mutex` is poisoned, any subsequent attempt to lock it will result in an `Err` to indicate its poisoned state.

**Example:**

```rust
use std::sync::{Mutex, Arc};
use std::thread;

let data = Arc::new(Mutex::new(0));

let data_clone = Arc::clone(&data);
let handle = thread::spawn(move || {
    let mut lock = data_clone.lock().unwrap();
    *lock += 1;
    panic!("Thread panicked!");
});

let _ = handle.join();  // Join the thread to handle the panic

match data.lock() {
    Ok(_) => println!("Lock acquired successfully."),
    Err(poisoned) => {
        let mut lock = poisoned.into_inner();
        *lock += 1;
        println!("Recovered from poison, new value: {}", *lock);
    }
}
```

- In this example, the thread panics after incrementing the integer, poisoning the `Mutex`.
- The main thread handles the poisoned lock by recovering and incrementing the value.

---

#### 2.2 Lifetime of MutexGuard

**Automatic Unlocking:**
- The `MutexGuard` automatically unlocks the `Mutex` when it goes out of scope.
- Explicitly dropping the guard can control the lock duration more precisely.

**Example:**

```rust
let data = Arc::new(Mutex::new(Vec::new()));

{
    let mut vec = data.lock().unwrap();
    vec.push(1);
}  // `vec` goes out of scope and the lock is released

// Single statement usage
data.lock().unwrap().push(2);
```

- In the above example, the `Mutex` is locked and unlocked within a single statement, ensuring minimal lock duration.

**Pitfall Example:**

```rust
if let Some(item) = data.lock().unwrap().pop() {
    process_item(item);
}
// `MutexGuard` is held until the end of the if-let statement, causing unnecessary lock duration

// Corrected Version
let item = data.lock().unwrap().pop();
if let Some(item) = item {
    process_item(item);
}
```

- By splitting the lock and the conditional, the `Mutex` is held for a shorter duration.

---

#### 2.3 Reader-Writer Lock

**Concept:**
- A `RwLock` allows multiple readers or a single writer.
- Suitable for data that is frequently read but infrequently written.

**Example:**

```rust
use std::sync::{RwLock, Arc};
use std::thread;

let data = Arc::new(RwLock::new(0));

// Reader thread
let data_clone = Arc::clone(&data);
let reader = thread::spawn(move || {
    let lock = data_clone.read().unwrap();
    println!("Read value: {}", *lock);
});

// Writer thread
let data_clone = Arc::clone(&data);
let writer = thread::spawn(move || {
    let mut lock = data_clone.write().unwrap();
    *lock += 1;
    println!("Updated value to: {}", *lock);
});

reader.join().unwrap();
writer.join().unwrap();
```

- `RwLock` provides `read` and `write` methods for acquiring read and write locks, respectively.
- The `RwLockReadGuard` and `RwLockWriteGuard` ensure the correct access patterns are upheld.

---

#### 2.4 Mutexes in Other Languages

**Comparison with C++:**
- In Rust, a `Mutex` contains the data it protects, ensuring safe access patterns.
- In C++, a `std::mutex` does not contain data, and it is the programmer's responsibility to manage the association.

**Example of Stand-Alone Mutex in Rust:**

```rust
use std::sync::Mutex;

let hardware_mutex = Mutex::new(());

// Use the mutex to protect access to some external hardware
```

- Even for external resources, it is often better to wrap them in a `Mutex` to enforce safe access patterns.

---

#### 2.5 Waiting: Parking and Condition Variables

**Thread Parking:**

```rust
use std::thread;
use std::sync::{Mutex, Arc};
use std::collections::VecDeque;
use std::time::Duration;

let queue = Arc::new(Mutex::new(VecDeque::new()));

thread::scope(|s| {
    let t = s.spawn(|| {
        loop {
            let item = queue.lock().unwrap().pop_front();
            if let Some(item) = item {
                println!("Consumed: {}", item);
            } else {
                thread::park();
            }
        }
    });

    for i in 0.. {
        queue.lock().unwrap().push_back(i);
        t.thread().unpark();
        thread::sleep(Duration::from_secs(1));
    }
});
```

- The consuming thread parks itself when the queue is empty and is unparked by the producing thread when a new item is added.

**Condition Variables:**

```rust
use std::sync::{Condvar, Mutex, Arc};
use std::thread;
use std::collections::VecDeque;
use std::time::Duration;

let queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));

thread::scope(|s| {
    let queue = Arc::clone(&queue);
    s.spawn(move || {
        loop {
            let (lock, cvar) = &*queue;
            let mut q = lock.lock().unwrap();
            while q.is_empty() {
                q = cvar.wait(q).unwrap();
            }
            if let Some(item) = q.pop_front() {
                println!("Consumed: {}", item);
            }
        }
    });

    for i in 0.. {
        let (lock, cvar) = &*queue;
        let mut q = lock.lock().unwrap();
        q.push_back(i);
        cvar.notify_one();
        thread::sleep(Duration::from_secs(1));
    }
});
```

- Using `Condvar`, the producer notifies the consumer when new items are added, avoiding busy waiting and spurious wake-ups.

---

### 3. Summary

- **Lock Poisoning:** Provides a mechanism to handle inconsistent states caused by panics, ensuring data integrity.
- **MutexGuard Lifecycle:** Proper management of `MutexGuard` lifetimes is crucial to avoid holding locks longer than necessary.
- **Reader-Writer Lock:** Offers a more flexible locking mechanism for data that is read often and written infrequently.
- **Thread Parking and Condition Variables:** Effective techniques for managing thread synchronization and waiting for conditions.

---

These detailed notes provide a comprehensive understanding of lock poisoning in Rust, tailored for a Staff+ level engineer, emphasizing practical application, advanced concurrency handling, and ensuring data consistency in multi-threaded environments.