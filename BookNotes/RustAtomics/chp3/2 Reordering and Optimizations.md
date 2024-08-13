In this chapter, we explore the concept of memory ordering in Rust and how it defines the execution order of operations in concurrent programs. Memory ordering is essential for writing correct and efficient concurrent code, as it dictates how operations on shared variables are observed by different threads. The memory model in Rust is abstract, providing a set of rules that are independent of specific hardware architectures, allowing for portability and future-proofing.

### Processor and Compiler Optimizations

Processors and compilers employ various techniques to optimize program execution. These optimizations may include reordering instructions to enhance performance, provided the reordering does not alter the program's observable behavior.

Consider the following function:

```rust
fn f(a: &mut i32, b: &mut i32) {
    *a += 1;
    *b += 1;
    *a += 1;
}
```

The compiler might optimize this code by reordering and merging operations, as follows:

```rust
fn f(a: &mut i32, b: &mut i32) {
    *a += 2;
    *b += 1;
}
```

During execution, the processor might further reorder these instructions based on data availability in caches, as long as the final result remains the same.

### Impact on Multi-Threaded Programs

These optimizations are safe within a single-threaded context. However, in multi-threaded programs, such reordering can lead to inconsistencies if not properly controlled. When working with shared data between threads, we must explicitly control the memory ordering to ensure correct program behavior.

## Memory Ordering in Rust

In Rust, memory ordering for atomic operations is controlled using the `std::sync::atomic::Ordering` enum. This enum provides a limited set of options, abstracting the underlying compiler and processor mechanisms to ensure architecture-independent and future-proof code.

The available orderings in Rust are:
- **Relaxed ordering:** `Ordering::Relaxed`
- **Release and acquire ordering:** `Ordering::{Release, Acquire, AcqRel}`
- **Sequentially consistent ordering:** `Ordering::SeqCst`

### Relaxed Ordering

Relaxed ordering, specified by `Ordering::Relaxed`, allows operations to be freely reordered around it. This ordering only guarantees atomicity without imposing any additional synchronization constraints.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

fn relaxed_example() {
    let x = AtomicUsize::new(0);
    x.store(1, Ordering::Relaxed);
    let y = x.load(Ordering::Relaxed);
    println!("y: {}", y); // y is guaranteed to be 1
}
```

### Release and Acquire Ordering

Release and acquire ordering provide synchronization between threads, ensuring that operations before a release are visible to threads that acquire the same atomic variable.

- **Release ordering** (`Ordering::Release`): Ensures all previous writes are visible before the release operation.
- **Acquire ordering** (`Ordering::Acquire`): Ensures all subsequent reads see data up to the acquire operation.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

fn release_acquire_example() {
    let x = AtomicUsize::new(0);
    let y = AtomicUsize::new(0);

    thread::spawn(move || {
        x.store(42, Ordering::Release);
        y.store(1, Ordering::Release);
    });

    while y.load(Ordering::Acquire) == 0 {}
    assert_eq!(x.load(Ordering::Acquire), 42);
}
```

### Sequentially Consistent Ordering

Sequentially consistent ordering, specified by `Ordering::SeqCst`, ensures the strictest form of ordering, providing a total ordering of all operations.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

fn seq_cst_example() {
    let x = AtomicUsize::new(0);
    let y = AtomicUsize::new(0);

    thread::spawn(move || {
        x.store(42, Ordering::SeqCst);
        y.store(1, Ordering::SeqCst);
    });

    while y.load(Ordering::SeqCst) == 0 {}
    assert_eq!(x.load(Ordering::SeqCst), 42);
}
```
## The Memory Model

### Abstract Memory Model

Rust's memory model defines memory ordering in terms of an abstract model with strict rules that decouple it from specific hardware architectures. This model ensures that concurrent atomic operations are well-defined, while allowing compilers and processors to optimize within the constraints of these rules.

### Happens-Before Relationship

The memory model defines the order in which operations happen in terms of happens-before relationships. This abstract model doesn't discuss machine instructions, caches, buffers, or timing. Instead, it defines when one operation is guaranteed to happen before another, leaving the order of other operations undefined.

The basic happens-before rule states that everything within the same thread happens in order. For example:

```rust
fn example() {
    f();
    g();
}
```

In the above code, `f()` happens-before `g()`.

### Cross-Thread Happens-Before Relationships

Cross-thread happens-before relationships occur in specific cases, such as when spawning and joining a thread, unlocking and locking a mutex, and through atomic operations with non-relaxed memory ordering.

#### Example with Relaxed Ordering

Consider the following example where `a` and `b` are executed concurrently by different threads:

```rust
static X: AtomicI32 = AtomicI32::new(0);
static Y: AtomicI32 = AtomicI32::new(0);

fn a() {
    X.store(10, Ordering::Relaxed);
    Y.store(20, Ordering::Relaxed);
}

fn b() {
    let y = Y.load(Relaxed);
    let x = X.load(Relaxed);
    println!("{x} {y}");
}
```

In this case, the operations within each thread happen in order, but there are no cross-thread happens-before relationships due to relaxed ordering. This allows for various possible outputs, including `0 0`, `10 20`, `10 0`, and `0 20`.

### Spawning and Joining Threads

Spawning a thread creates a happens-before relationship between what happened before the `spawn()` call and the new thread. Joining a thread creates a happens-before relationship between the joined thread and what happens after the `join()` call.

#### Example with Spawning and Joining
The assertion in the following example cannot fail:
```rust
static X: AtomicI32 = AtomicI32::new(0);

fn main() {
    X.store(1, Ordering::Relaxed);
    let t = thread::spawn(f);
    X.store(2, Ordering::Relaxed);
    t.join().unwrap();
    X.store(3, Ordering::Relaxed);
}

fn f() {
    let x = X.load(Ordering::Relaxed);
    assert!(x == 1 || x == 2);
}
```
In this example, the happens-before relationships formed by the spawn and join operations ensure that the load from `X` happens after the first store and before the last store.
## Summary

The memory model in Rust provides a framework for understanding and controlling the order of operations in concurrent programs. By defining happens-before relationships and using memory orderings, we can ensure correct synchronization between threads while allowing for necessary optimizations. Understanding and correctly applying these concepts is crucial for writing efficient and correct concurrent programs.