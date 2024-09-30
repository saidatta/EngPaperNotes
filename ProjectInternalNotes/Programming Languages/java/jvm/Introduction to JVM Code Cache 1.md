## 1. Overview

This article provides an in-depth understanding of the **JVM Code Cache**—a dedicated area in JVM memory for storing bytecode compiled into **native code** by the **Just-In-Time (JIT) compiler**. We'll also explore how to **tune the Code Cache**, monitor its usage, and the segmented code cache structure introduced in **Java 9**.

---
## 2. What Is the Code Cache?

The **JVM Code Cache** is a memory region where compiled bytecode is stored in the form of **native code**. The compiled native code is known as **nmethod**, which may represent a complete Java method or an **inlined** Java method.
- **nmethod**: A block of executable native code corresponding to a Java method or an inlined method.
- **JIT Compiler**: The primary consumer of the Code Cache; hence, the memory is also referred to as the **JIT Code Cache**.
This caching mechanism allows Java methods to run faster since the JVM doesn't need to interpret bytecode repeatedly. Instead, the native code is executed directly by the CPU.

---
## 3. Code Cache Tuning

### 3.1 Default Configuration

The **Code Cache** has a **fixed size**. Once it's full, the **JIT compiler** will stop compiling new code, which will lead to performance degradation, as the JVM will revert to interpreting methods. If the cache becomes full, you’ll see this warning:
```
CodeCache is full… The compiler has been disabled
```
### 3.2 Tuning Options
To avoid performance degradation, we can **tune the Code Cache** by modifying the following JVM options:
- **InitialCodeCacheSize**: Initial size of the Code Cache. Default: `160 KB`.  
- **ReservedCodeCacheSize**: Maximum size of the Code Cache. Default: `48 MB`.
- **CodeCacheExpansionSize**: The size by which the Code Cache expands when needed. Default: `32 KB` or `64 KB`.
```bash
# Example JVM tuning options
-XX:InitialCodeCacheSize=1m -XX:ReservedCodeCacheSize=128m -XX:CodeCacheExpansionSize=64k
```
### 3.3 Code Cache Flushing
When increasing the `ReservedCodeCacheSize` becomes insufficient or impractical, the JVM can flush the Code Cache using the **UseCodeCacheFlushing** option. By default, **flushing** is disabled.
**Enable flushing**:
```bash
-XX:+UseCodeCacheFlushing
```
#### Conditions for Code Cache Flushing
The JVM will flush the cache under the following conditions:
1. **Cache Full**: The code cache is full and its size exceeds a certain threshold.
2. **Interval Passed**: A certain time interval has passed since the last cleanup.
3. **Method Hotness**: The method’s hotness counter is low. The JVM tracks how often a method is executed; if a method is not “hot” enough (i.e., frequently used), its compiled code may be discarded.

---
## 4. Code Cache Usage

To **monitor the usage** of the JVM Code Cache, use the following JVM option:
```bash
-XX:+PrintCodeCache
```
This will output details like the **size**, **used space**, and **free space** of the code cache during application execution. Example output:
```plaintext
CodeCache: size=32768Kb used=542Kb max_used=542Kb free=32226Kb
```
### Explanation of Output:
- **size**: Maximum size of the Code Cache, corresponding to the `ReservedCodeCacheSize`.
- **used**: The currently occupied memory in the Code Cache.
- **max_used**: The maximum size of the memory that has been used.
- **free**: The remaining memory available for future compiled code.

This output helps identify:
- When the **Code Cache** is nearing capacity.
- When **flushing** occurs.
- Whether the JVM has reached critical memory usage in the Code Cache.

---
## 5. Segmented Code Cache (Java 9+)

Starting from **Java 9**, the JVM introduced a **segmented Code Cache** to optimize memory management and improve performance. The cache is divided into **three segments**, each serving a distinct purpose:
### 5.1 Segments Overview

1. **Non-method Code Segment**:
   - Stores internal JVM-related code (e.g., the bytecode interpreter).
   - Default size: **5 MB**.
   - Configurable with: `-XX:NonNMethodCodeHeapSize`.
2. **Profiled Code Segment**:
   - Contains **lightly optimized code** with **short lifetimes**.
   - Default size: **122 MB**.
   - Configurable with: `-XX:ProfiledCodeHeapSize`.
3. **Non-profiled Code Segment**:
   - Stores **fully optimized code** with **longer lifetimes**.
   - Default size: **122 MB**.
   - Configurable with: `-XX:NonProfiledCodeHeapSize`.
### 5.2 Advantages of Segmentation
- **Separation of Short-Lived and Long-Lived Code**: By separating lightly optimized, short-lived methods from heavily optimized, long-lived methods, the JVM can perform **more efficient flushing** and **garbage collection** of the cache.
- **Improved Method Sweeper Performance**: The **Method Sweeper**—a mechanism responsible for cleaning up unused code—can operate more efficiently since it scans smaller memory regions.
### ASCII Visualization of Segmented Code Cache

```plaintext
+-------------------------+       +-----------------------+
| Non-method Segment       |       | Profiled Code Segment |
| (5 MB)                  |       | (122 MB)              |
+-------------------------+       +-----------------------+
| Non-profiled Code Segment (122 MB)                        |
+-----------------------------------------------------------+
```

---

## 6. Conclusion

The **JVM Code Cache** is critical for maintaining Java application performance by storing compiled native code. Proper tuning and monitoring of the Code Cache can help avoid performance degradation and excessive memory usage.

With **Java 9’s segmented Code Cache**, the JVM has introduced a more efficient way to manage different types of compiled code, ensuring better memory optimization and faster execution of hot methods.

### Key Takeaways:
- **Tuning**: Adjust `ReservedCodeCacheSize` and related flags to optimize cache size.
- **Monitoring**: Use `-XX:+PrintCodeCache` to track memory usage and identify potential bottlenecks.
- **Segmentation**: Java 9+ divides the cache into segments, enhancing memory management and improving performance for long-running applications.

By understanding and leveraging the JVM Code Cache, developers can ensure that their Java applications perform optimally, especially in **high-performance** or **long-running environments**.

--- 

### Code Snippets

**Example JVM options for Code Cache tuning**:
```bash
-XX:InitialCodeCacheSize=2m -XX:ReservedCodeCacheSize=128m -XX:CodeCacheExpansionSize=64k
-XX:+PrintCodeCache
```

**Enable Code Cache flushing**:
```bash
-XX:+UseCodeCacheFlushing
```

---

