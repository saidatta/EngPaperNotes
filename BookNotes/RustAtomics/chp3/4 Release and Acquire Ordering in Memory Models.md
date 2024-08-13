Release and acquire memory ordering establish a happens-before relationship between threads, ensuring proper synchronization for concurrent programming. Release memory ordering applies to store operations, while acquire memory ordering applies to load operations.
## Basic Concepts
### Definition
- **Release Ordering**: Used in store operations, it ensures that all previous writes by the thread are visible to other threads that acquire the same variable.
- **Acquire Ordering**: Used in load operations, it ensures that subsequent reads by the thread will see the writes from threads that have released the same variable.
- **AcqRel**: A combination of acquire and release ordering, typically used in atomic read-modify-write operations like `fetch_add` or `compare_and_exchange`.
### Establishing Happens-Before Relationship
A happens-before relationship is formed when an acquire-load operation observes the result of a release-store operation. In this case, the store and everything before it happened before the load and everything after it.
## Example: Passing Data Between Threads
Let's consider an example where one thread sends a 64-bit integer to the main thread using an atomic boolean to signal readiness.

```rust
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

static DATA: AtomicU64 = AtomicU64::new(0);
static READY: AtomicBool = AtomicBool::new(false);

fn main() {
    thread::spawn(|| {
        DATA.store(123, Ordering::Relaxed);
        READY.store(true, Ordering::Release); // Release store
    });

    while !READY.load(Ordering::Acquire) { // Acquire load
        thread::sleep(Duration::from_millis(100));
        println!("waiting...");
    }

    println!("{}", DATA.load(Ordering::Relaxed)); // Prints 123
}
```
### Explanation
- The spawned thread uses `READY.store(true, Ordering::Release)` to signal that `DATA` is ready.
- The main thread uses `READY.load(Ordering::Acquire)` to wait for the signal.
- When the acquire-load observes the release-store, a happens-before relationship is established, ensuring that the main thread will see the correct value of `DATA`.
## More Formally
The happens-before relationship using acquire and release can be visualized as follows:
### Visualization

```plaintext
Thread A                        Thread B
----------                      ----------
DATA.store(123, Relaxed);       
READY.store(true, Release);     while !READY.load(Acquire) {
                                    thread::sleep(Duration::from_millis(100));
                                    println!("waiting...");
                                }
                                println!("{}", DATA.load(Relaxed)); // Prints 123
```

### Total Modification Order

Imagine two threads release-storing a value into the same atomic variable, and a third thread acquire-loading from that variable. The acquire-load will form a happens-before relationship with the specific release-store operation that it observes.

## Example: Mutex Implementation

Mutexes commonly use acquire and release ordering. Unlocking a mutex involves a release-store, while locking it involves an acquire-load.

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

static mut DATA: String = String::new();
static LOCKED: AtomicBool = AtomicBool::new(false);

fn f() {
    if LOCKED.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
        // Safety: We hold the exclusive lock, so nothing else is accessing DATA.
        unsafe { DATA.push('!') };
        LOCKED.store(false, Ordering::Release);
    }
}

fn main() {
    thread::scope(|s| {
        for _ in 0..100 {
            s.spawn(f);
        }
    });
}
```

### Explanation

- **Locking**: Uses `compare_exchange` with acquire ordering to ensure that the lock is acquired atomically.
- **Unlocking**: Uses `store` with release ordering to ensure that changes to `DATA` are visible to other threads before releasing the lock.

## Example: Lazy Initialization

Consider a scenario where we want to lazily initialize a large data structure, ensuring that only one thread performs the initialization.

```rust
use std::sync::atomic::{AtomicPtr, Ordering};
use std::ptr;

struct Data;

fn generate_data() -> Data {
    // Expensive initialization
    Data
}

fn get_data() -> &'static Data {
    static PTR: AtomicPtr<Data> = AtomicPtr::new(ptr::null_mut());
    let mut p = PTR.load(Ordering::Acquire);
    if p.is_null() {
        p = Box::into_raw(Box::new(generate_data()));
        if let Err(e) = PTR.compare_exchange(ptr::null_mut(), p, Ordering::Release, Ordering::Acquire) {
            // Safety: p comes from Box::into_raw right above,
            // and wasn't shared with any other thread.
            drop(unsafe { Box::from_raw(p) });
            p = e;
        }
    }
    // Safety: p is not null and points to a properly initialized value.
    unsafe { &*p }
}
```

### Explanation

- **Acquire Load**: Reads the pointer to check if the data has already been initialized.
- **Release Store**: Writes the pointer after initializing the data, ensuring that all changes are visible.
- **Atomic Pointer**: Used to manage the shared data safely across threads.

### Visualization

```plaintext
Thread A                Thread B                  Thread C
-----------             -----------               -----------
p = PTR.load(Acquire);  p = PTR.load(Acquire);    p = PTR.load(Acquire);
if p.is_null() {        if p.is_null() {
    p = Box::into_raw(      p = Box::into_raw(
        Box::new(              Box::new(
            generate_data()        generate_data()
        )                    )
    );                   );
    PTR.compare_exchange(  PTR.compare_exchange(
        null_mut(),           null_mut(),
        p,                    p,
        Release,              Release,
        Acquire               Acquire
    );                   ); // fails
}                      }
```

### Key Points

1. **Acquire-Release Synchronization**: Ensures proper visibility of changes across threads.
2. **Atomic Pointer**: Allows for safe concurrent initialization.
3. **Memory Ordering**: Critical for preventing instruction reordering that could lead to incorrect behavior.

In summary, release and acquire ordering are essential tools for ensuring proper synchronization between threads in concurrent programming. They provide a formal mechanism to establish happens-before relationships, guaranteeing that certain operations occur in a specific order relative to each other. Understanding and correctly applying these concepts is crucial for designing robust, high-performance concurrent systems.