https://claytonwramsey.github.io/blog/dumpster.html
---
## Overview
- **Topic**: Building a custom garbage collector in Rust, a language typically not needing one.
- **Purpose**: To manage allocations in Rust, particularly for complex cases where standard solutions like `Rc` and `Arc` are insufficient.
- **Repository**: [GitHub Link](https://github.com/)
- **Crate**: Available at crates.io.
## Background: Rust's Memory Model
- **Affine Typing & Borrow Checker**: Ensures values are bound to one identifier at a time; references don't outlive their scope.
  - *Example*: 
    ```rust
    let x = vec![1, 2, 3];
    let y = x; // x is moved
    // println!("{x:?}"); // Error: x has been moved
    ```
  - *Workaround*: Use borrowing, e.g., `let y = &x;`.
## Shared Ownership via Garbage Collection
- **Rust's Standard Library**: Provides `Rc` (single-threaded) and `Arc` (atomic, multi-threaded) for reference-counted garbage collection.
- **Limitation**: Inability to handle cyclic references.
  - *Example*:
    ```rust
    use std::{cell::OnceCell, rc::Rc};
    struct Foo(OnceCell<Rc<Foo>>);
    let x = Rc::new(Foo(OnceCell::new()));
    x.0.set(Rc::clone(&x)); // Creates a cycle
    ```
## Battle Plan for Custom GC
- **Goal**: Create a `Gc` structure, similar to `Rc` and `Arc`, but capable of cycle detection.
- **Tools**:
  - `Drop` Trait: Utilize for cleanup when an object is dropped.
  - `Collectable` Trait: Mandatory for types to be managed by `Gc`.
## Challenges
- **Undetectable Moves**: No Rust mechanism to detect when a value is moved.
- **Variable-Sized Data**: Support for `?Sized` types in `Gc`.
- **Type Erasure**: Rust's generics via monomorphization complicate runtime type information.
## Implementation Approach
- **Graph-Theoretical Definitions**:
  - *Allocation Graph*: Directed graph with nodes as allocations.
  - *Accessibility*: Node is accessible if reachable from the root node.
### Pseudocode for Accessibility Check
```python
def is_accessible(n):
    # Counts and DFS to find reachable nodes
    # Mark nodes as accessible
    return n in reachable
```
## Code: Single-Threaded `Gc`
- **Structure**: Similar to `Rc`, with modifications for cycle detection.
- **Example**:
  ```rust
  pub struct Gc<T: ?Sized> {
      ptr: NonNull<GcBox<T>>
  }
  struct GcBox<T: ?Sized> {
      ref_count: Cell<usize>,
      value: T
  }
  impl<T: ?Sized> Drop for Gc<T> {
      // Custom drop logic
  }
  ```
## Trait Hackery
- **Collectable Trait**: Forces garbage-collected values to implement certain behavior.
- **Visitor Pattern**: Delegates a visitor to garbage-collected fields.
- **Implementation Example**:
  ```rust
  pub trait Collectable {
      fn accept<V: Visitor>(&self, visitor: &mut V);
  }
  impl<T: Collectable> Collectable for [T] {
      // Implementation for arrays
  }
  ```
---
## Bulk Dropping
- **Problem Identified**: The `drop` function, as initially implemented, runs in quadratic time due to nested drops.
- **Solution**: Implement bulk dropping to avoid quadratic-time cleanup.
### Concept: Bulk Dropping
- **Idea**: Record pointers to each allocation to be destroyed, along with their `drop` implementations. Perform a single pass to mark and then drop all inaccessible allocations.
- **Benefit**: Significantly reduces time complexity from quadratic to linear.
### Code Implementation: Bulk Dropping
- **Thread-Local `COLLECTING` State**:
  ```rust
  thread_local! {
      static COLLECTING: Cell<bool> = Cell::new(false);
  }
  ```
- **Modified `Drop` for `Gc`**:
  - Check `COLLECTING` state before performing drop operations.
  - Bypasses manipulation of reference counts if `COLLECTING` is true.
  - Adds an additional traversal for handling outbound edges.
  - Example:
    ```rust
    impl<T: Collectable + ?Sized> Drop for Gc<T> {
        fn drop(&mut self) {
            // if COLLECTING, return immediately
            // else, perform standard drop operations
        }
    }
    ```

## Handling Type Erasure and Allocation Identification
- **Challenge**: Store pointers as keys in a hashmap and reconstruct pointers using type information.
- **Solution**: Use two kinds of pointers: thin for comparisons, fat for type reconstruction.

### Code Example: DFS Implementation
- **Function**: `unsafe fn dfs<T: Collectable + ?Sized>`
- **Purpose**: Traverse the graph, decrementing counts, and identifying allocations for cleanup.
- **Visitor Pattern**: Implement `Visitor` for traversing collectable types.
- **Example**:
  ```rust
  struct DfsVisitor<'a> {
      counts: &'a mut HashMap<AllocationId, usize>,
  }
  impl Visitor for DfsVisitor<'_> {
      // visit_gc implementation
  }
  ```

## Amortizing Collection Effort
- **Goal**: Reduce average-case time complexity of garbage-collection operations.
- **Method**: Use a "dirty" set (referred to as a dumpster) to store potentially inaccessible allocations.
- **Strategy**: Periodically clean the dumpster, distributing the collection effort over time.
- **Memory Safety**: Implement checks to ensure allocations in the dumpster are not prematurely freed.

## Concurrent Garbage Collection
- **Challenge**: Adapt the algorithm for concurrent environments.
- **Solution**: Implement a collecting tag system and atomic operations.
- **Key Components**:
  - Global atomic `COLLECTING_TAG`.
  - Annotate each `Gc` with the current value of `COLLECTING_TAG`.
  - Adjust the collection process to consider these tags.

### Code Snippet: Concurrent `Gc`
- **Structure**:
  ```rust
  pub struct Gc<T: Collectable + Send + Sync + ?Sized> {
      ptr: NonNull<GcBox<T>>,
      tag: AtomicUsize,
  }
  ```

## Resolving Concurrency Issues and Correctness Proof
- **Concurrency Issue**: Prevent erroneous early deallocations due to concurrent modifications.
- **Solution**: Tag allocations and check these tags during DFS.
- **Proof**: Outline a theorem stating that DFS will not incorrectly mark an allocation as inaccessible under concurrent execution.

## Weak Reference Counts for Memory Safety
- **Problem**: Allocations in the dumpster may still be accessible by other threads.
- **Solution**: Introduce weak reference counts to track these cases.
- **Implementation**:
  ```rust
  struct GcBox<T: Collectable + Send + Sync + ?Sized> {
      strong: AtomicUsize,
      weak: AtomicUsize,
      // ...
  }
  ```

## Handling Mutex Locks in Garbage Collection
- **Issue**: Deadlocks due to locking Mutex within the garbage collection process.
- **Solution**: Make `accept` method in `Collectable` trait fallible. Failing to acquire a lock implies allocation is accessible.

### Code Example: Mutex Handling
- **Mutex Implementation for `Collectable`**:
  ```rust
  impl<T: Collectable + ?Sized> Collectable for Mutex<T> {
      fn accept<V: Visitor>(&self, visitor: &mut V) -> Result<(), ()> {
          // Implementation details
      }
  }
  ```

---

Continuing with the detailed Obsidian notes for the article on custom garbage collection in Rust, this section will delve into the challenges of implementing a concurrent garbage collector, its performance, and some additional insights from the development process.

---
## Sharing Your Dumpsters: Enhancing Concurrent Performance

### Problem with Global Dumpster
- **Issue**: Accessing a single global dumpster (e.g., using `Mutex<HashMap>`) severely limits parallelism, negatively impacting performance on multi-core systems.
- **Observation**: Even concurrent hashmaps like `chashmap` and `dashmap` faced performance issues with increased thread count.

### Solution: Thread-Local Dumpsters
- **Concept**: Each thread maintains its own local dumpster. A global "garbage truck" holds allocations marked for cleanup.
- **Mechanism**: When a thread's dumpster is full (based on a heuristic), its contents are transferred to the global garbage truck.
- **Pseudocode**:
  ```python
  dumpster = set() # local to this thread
  garbage_truck = mutex(set()) # global to all threads

  def mark_dirty(allocation):
      dumpster.add(allocation)
      if is_full(dumpster):
          garbage_truck.lock()
          for trash_bag in dumpster:
              garbage_truck.add(trash_bag)
          garbage_truck.unlock()
          dumpster = set()
  ```
- **Advantage**: Reduces contention over the global structure, improving concurrent performance.

## Performance Analysis

### Different Implementations
- **Thread-Local GC**: Avoids concurrency issues.
- **Thread-Safe GC**: Offers more features but with some performance trade-offs.

### Benchmarking and Comparison
- **Approach**: Compare various garbage collectors in Rust based on similar APIs and functionality.
- **Garbage Collectors Tested**: Including `Rc`, `Arc`, `bacon-rajan-cc`, `cactusref`, `elise`, `gc`, `gcmodule`, `rcgc`, `shredder`, and both `dumpster` versions.

### Single-Threaded Performance
- **Method**: Measure runtime for 1 million operations involving creation and deletion of references.
- **Results**: Show that `shredder` is slower compared to others. `dumpster`'s unsync version performs well, closely following `Rc`, `Arc`, and `bacon-rajan-cc`.

### Multi-Threaded Performance
- **Method**: Parallel scaling test across different thread counts.
- **Findings**:
  - `shredder` performs poorly in multi-threaded scenarios.
  - `dumpster` (sync versions) scales reasonably well with thread count, though not as smoothly as `Arc`.
  - The sync version of `dumpster` is about 3x slower than `Arc`, which is considered a decent outcome for this project.

## Code: Project Summary and Acknowledgements

### Project Overview
- **Outcome**: Successfully built a garbage collector for Rust, applicable in both single-threaded and multi-threaded contexts.
- **Performance**: Demonstrated reasonable performance, particularly in the thread-local implementation.
- **Availability**: The implementation is available on crates.io under GNU GPLv3.

### Acknowledgements
- Special thanks to reviewers and supporters who helped refine the project and the post.

## Postscript: Additional Insights and Challenges

### Concurrency Handling Ideas
- **Original Concepts**:
  - Wrapping a Mutex around every allocation's reference count.
  - Using lower bit tagging for space efficiency.
  - Implementing dynamic dispatch for Java-like object handling.
- **Challenges**: Many initial ideas for handling concurrency were unsuccessful or impractical.

### Nightly Features and Limitations
- **Desired Features**: Implementing `CoerceUnsized` for garbage collectors and manipulating fat pointers' address portion.
- **Stable vs. Nightly**: Certain optimizations and features are not accessible in stable Rust due to compiler limitations.

### Alternative Approaches and Miri Tool
- **Visitor Pattern Limitations**: Challenges in making `Collectable` object-safe.
- **Miri**: Emphasized the utility of `cargo miri` for debugging unsafe Rust code.

### Testing and Debugging
- **Fuzzing**: Used to generate random reference graphs for testing garbage collector efficiency.
- **Debugging Complexities**: Difficulty in identifying issues in concurrent scenarios, necessitating custom visualization tools for reference graphs.

---
Creating detailed Obsidian notes for the article on the "dumpster" garbage collector in Rust will involve organizing the content into clear, structured sections with key points, examples, and relevant technical details.

---

# Detailed Obsidian Notes: "dumpster" Garbage Collector for Rust

## Overview of dumpster
- **Purpose**: A cycle-detecting garbage collector for Rust designed to manage and free unreachable allocations.
- **Key Features**:
  - **API Compatibility**: Acts as a drop-in replacement for standard Rust reference-counted shared allocations (`Rc` and `Arc`).
  - **Performance**: Offers high performance with implementations for both thread-local and concurrent garbage collection.
  - **Flexibility**: Allows any reference structure within garbage-collected allocations.
  - **Ease of Use**: Simplifies making custom types collectable with provided derive macros.
  - **Support for ?Sized Data**: Enables storing dynamically sized data in a garbage-collected pointer.

## How dumpster Works
- **Distinct Approach**: Unlike traditional tracing garbage collectors that track roots for sweeping, dumpster extends reference-counted garbage collection with a cycle-detection algorithm.
- **Functionality**: Effectively cleans up self-referential data structures that are otherwise challenging for standard reference counting.

## Library Contents
- **Two Implementations**:
  - `unsync`: Thread-local, non-`Send` garbage collector.
  - `sync`: Thread-safe garbage collector.
- **Compatibility**: Both implementations can be used interchangeably.
- **Derive Macro**: Facilitates creating custom collectable types.

## Examples
- **Usage Example**:
  ```rust
  use dumpster::{Collectable, unsync::Gc};
  use std::cell::RefCell;

  #[derive(Collectable)]
  struct Foo {
      ptr: RefCell<Option<Gc<Foo>>>,
  }

  let foo = Gc::new(Foo { ptr: RefCell::new(None) });
  *foo.ptr.borrow_mut() = Some(foo.clone());
  drop(foo);
  dumpster::unsync::collect(); // Trigger a collection
  ```
- **Context**: Demonstrates creating and using a self-referential structure with `Gc`. Shows how `Gc` can handle potential memory leaks that `Rc` cannot.

## Installation
- **Dependency Addition**:
  ```toml
  [dependencies]
  dumpster = "0.1.2"
  ```

## Optional Features
- **`derive` Feature**:
  - **Default Enabled**: Simplifies implementing `Collectable` for custom types.
  - **Usage Example**:
    ```rust
    #[derive(Collectable)] // Automates implementation
    struct Foo(RefCell<Option<Gc<Foo>>>);
    ```
- **`coerce-unsized` Feature**:
  - **Default Disabled**: Allows using `Gc` with `!Sized` types.
  - **Nightly Rust Requirement**: Necessary for enabling dynamic sizing support.
  - **Installation with Feature**:
    ```toml
    [dependencies]
    dumpster = { version = "0.1.2", features = ["coerce-unsized"]}
    ```