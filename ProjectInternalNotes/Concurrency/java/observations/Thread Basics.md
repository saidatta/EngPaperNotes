https://juejin.cn/post/7242965902877671485

Volatile keyword.

- **Guarantees data visibility:** By adding `volatile`, you ensure each thread always reads the latest value of the modified variable, preventing "dirty reads" where a thread gets an outdated value.
- **Disables instruction rearrangement:** `volatile` prevents the compiler from rearranging instructions around it, crucial in multi-threaded scenarios where order matters.

**Low-level implementation:**

- **Memory barrier:** Adding `volatile` often generates a "lock prefix" instruction, which acts as a memory barrier. This barrier enforces three functionalities:
    - **Instruction reordering:** Ensures instructions after the barrier happen after operations before it, and vice versa.
    - **Cache write-through:** Forces modified data in the cache to be immediately written to main memory.
    - **Cache invalidation:** In case of a write operation, invalidates the corresponding cache line on other CPUs to ensure everyone gets the updated value.

**Additional points:**

- `volatile` does not imply thread safety on its own. It ensures visibility but not atomicity (i.e., multiple threads might access the variable simultaneously, leading to unexpected behavior).
- Use `volatile` cautiously, as it can impact performance due to memory barriers and potential cache flushes. Consider alternatives like synchronization primitives (e.g., mutexes) for robust thread safety.

Overall, your explanation provides a clear understanding of `volatile`'s benefits and its underlying mechanism. Remember to use it judiciously and weigh its trade-offs carefully in your multi-threaded programming.