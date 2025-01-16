https://liujiacai.net/blog/2018/12/29/how-java-synchronizer-work/
## Overview
- **Category**: Programming Language  
- **Tag**: Java  
- **Published**: 2018-12-29  
- **Last Updated**: 2024-12-24  
- **Topics Covered**:
  - **JUC** (Java Util Concurrent)  
  - **AbstractQueuedSynchronizer (AQS)**  
  - **wait/notify** in Java  
  - **lock/unlock** with `LockSupport.park/unpark`  
  - **Operating System-level Synchronization Primitives** (Semaphores, Monitors, Condition Variables)  
  - **POSIX Implementation** (pthread_mutex, pthread_cond, sem_t)  
  - **Hoare vs. Mesa Monitor Semantics**  
  - **Monitors in Java** (`synchronized`, `Object.wait()`, `Object.notify()`)  
  - **Interruptible** concurrency  
  - **Big Picture** diagram of concurrency  
  - **Summary & References**  
---
## 1. Introduction
Modern server applications often rely on **multi-threading** to exploit hardware parallelism. In Java, threads map **1:1** with native OS threads. Thread **synchronization**—managing safe access to shared resources—remains one of the more complicated aspects of concurrent programming.

This article provides a **deep dive** into:
- How **Java** implements thread synchronization constructs under the hood.
- How these constructs map to **OS** primitives (like semaphores and condition variables).
- The **Java** concurrency toolkit **JUC** and its foundation, **AbstractQueuedSynchronizer**.
- The difference between **wait/notify** (object monitors) and **park/unpark** (LockSupport).
- Basic OS-level constructs: **Semaphores**, **Monitors**, **Condition Variables**.
- How **interruptibility** is handled in Java’s concurrency libraries.

**Prerequisites**: Familiarity with classes like `CyclicBarrier` and `CountDownLatch` in Java.  

---
## 2. JUC (Java Util Concurrent)
### 2.1 Overview
Introduced in **Java 1.5**, the **java.util.concurrent (JUC)** package provides robust concurrency tools:
- **Executors** for thread pooling and scheduling.
- **Queues** (blocking queues) for buffering tasks.
- **TimeUnit** for time-based utilities.
- **Concurrent collections** (e.g., `ConcurrentHashMap`).
- **Thread synchronization classes** (synchronizers such as `CountDownLatch`, `CyclicBarrier`, `Semaphore`, etc.).
**Thread synchronization** is arguably the most crucial piece, as the core difficulty in concurrency is controlling **access to shared resources**.
### 2.2 AbstractQueuedSynchronizer (AQS)
Key classes in JUC, such as:
- **Semaphore**
- **CountDownLatch**
- **CyclicBarrier**
- **Phaser**
- **Exchanger**

…are **all** powered by an abstract base class called **`AbstractQueuedSynchronizer (AQS)`**.

> *AQS* maintains:
> 1. A **FIFO** wait queue of threads.
> 2. An **atomic int** representing synchronization state.

This design supports **fairness**, **reentrancy**, **shared vs. exclusive locks**, and **interruptibility**. On top of AQS, Java implements a variety of high-level constructs (latches, barriers, etc.).

**Reference**:  
- [Meituan Tech’s “Java 'Lock' Things You Must Say” (Part 6)](https://tech.meituan.com/)  
- AQS Javadoc in OpenJDK.

---

## 3. How Java Implements Blocking and Notification

### 3.1 `wait/notify` (Object Monitor)

```java
public final native void wait(long timeout) throws InterruptedException;
public final native void notify();
public final native void notifyAll();
```

These methods are **`native`** in the JVM source. Looking into the OpenJDK C++ code:

```c++
// java.base/share/native/libjava/Object.c
static JNINativeMethod methods[] = {
    {"wait",      "(J)V", (void *)&JVM_MonitorWait},
    {"notify",    "()V",  (void *)&JVM_MonitorNotify},
    {"notifyAll", "()V",  (void *)&JVM_MonitorNotifyAll},
    ...
};
```

The suffix *Monitor* suggests that the JVM uses an **object monitor** behind the scenes. Each Java object has an associated **monitor** (or lock word), enabling `synchronized` blocks/methods as well as the `wait/notify` mechanism.

### 3.2 `lock/unlock` via `LockSupport`
Lock-based concurrency (e.g., `ReentrantLock`) relies on a helper class **`LockSupport`** which exposes:
```java
public class LockSupport {
    public static void unpark(Thread thread) { ... } 
    public static void park(Object blocker) { ... } 
}
```

Internally, these delegate to:

```java
UNSAFE.unpark(thread); // wakes a thread
UNSAFE.park(false, 0L); // blocks a thread
```

At the **native level** (`Unsafe.cpp` in HotSpot JVM), every Java thread has a **`Parker`** object using **`pthread_mutex_t`** and **`pthread_cond_t`** (on POSIX systems) to handle blocking:
```cpp
class Parker : public os::PlatformParker {
 public:
  void park(bool isAbsolute, jlong time);
  void unpark();
};

// Within unsafe.cpp
thread->parker()->park(isAbsolute != 0, time);
```

So, under the hood, **`park/unpark`** map to **POSIX condition variables** and **mutex** operations.

---

## 4. Synchronization Primitives at the OS Level

### 4.1 Semaphores

**Semaphore** is a classic concurrency mechanism introduced by **Edsger Dijkstra** in 1965.  
- A **semaphore** is a non-negative integer (`val`), with two main atomic operations: **Wait (P)** and **Signal (V)**:
  ```cpp
  wait(Sem):
      while (Sem.val <= 0) {
         // block thread
      }
      Sem.val--;

  signal(Sem):
      Sem.val++;
      // wake one blocked thread if any
  ```
- **Binary semaphores** (value 0 or 1) → act like **mutexes**.
- **Counting semaphores** → track available resources (like N buffer slots).

#### Usage Scenarios
1. **Mutual Exclusion**  
   - Protect a critical section with a **binary semaphore**.  
2. **Scheduling Shared Resources**  
   - E.g., **Producer-Consumer** with two semaphores, `fullSem` and `emptySem`.

#### POSIX Semaphores
Declared in **`<semaphore.h>`**:
```c
sem_t theSem;
sem_init(&theSem, 0, initialVal);
sem_wait(&theSem);
sem_post(&theSem);
sem_getvalue(&theSem, &result);
```

##### Disadvantages
- **Lack of structure**: easy to misuse `sem_wait/sem_post` orders.  
- **Global visibility**: deadlocks can be tricky to debug.  

This leads to **monitor**-based solutions.

---

### 4.2 Monitors

Proposed by **C.A.R. Hoare** (1974). A **monitor** is essentially:
- A **lock** ensuring only one thread accesses the monitor at a time.
- **Condition variables** inside the monitor for thread coordination.

**Pseudocode**:
```plaintext
monitor MonitorName {
    // shared data

    condition cv;

    public void someMethod() {
       ...
       if (!conditionMet) {
          cv.wait();
       }
       ...
    }
}
```

#### Condition Variables
- **`wait(cv, m)`**: Thread waits on condition `cv` and *releases lock `m`* so others can proceed.
- **`signal(cv)`**: Wake **one** waiting thread.
- **`broadcast(cv)`**: Wake **all** waiting threads.

##### POSIX Condition Variables
```c
pthread_cond_t myCV;
pthread_cond_init(&myCV, NULL);
pthread_cond_wait(&myCV, &someLock);
pthread_cond_signal(&myCV);
pthread_cond_broadcast(&myCV);
```

**Key Points**:
1. Must hold the **mutex** before calling `pthread_cond_wait` or `pthread_cond_signal`.
2. `pthread_cond_wait` automatically releases the mutex while waiting, then re-acquires it before returning.
3. `signal` wakes **one** waiting thread (if any), but that thread still competes for the lock.

---

### 4.3 Hoare vs. Mesa Monitor Semantics

**Hoare Monitors** (classic 1974 approach):
- Calling `signal` **immediately** transfers the monitor lock to the waiting thread.
- The signaling thread is blocked.

**Mesa Monitors** (Xerox PARC ~1980, used in Java/C#/pthreads):
- The thread calling `signal` continues.
- The waiting thread goes into the **ready queue** and must re-acquire the lock to proceed.

**Practical Difference**:
- Under Mesa, once awakened, a thread must **re-check** the condition because it might have changed while waiting.

---

## 5. Monitors in Java

- **Each object** in Java can act as a **monitor**:
  - `synchronized(obj)` → acquires the **obj** lock.
  - `obj.wait()` → the thread releases **obj**’s lock, waits on the condition in that monitor.
  - `obj.notify()`/`obj.notifyAll()` → signals waiting threads in that monitor.

**In essence**, `wait/notify/notifyAll` in Java are monitor-based synchronization. The difference from `LockSupport` is primarily **the use of object’s intrinsic lock** vs. a more flexible lock implementation in JUC.

---

## 6. Big Picture

Below is an overview of concurrency-related APIs and how they map to OS primitives.

```mermaid
flowchart LR
    A((Java Thread)) --> B[LockSupport.park/unpark <br> (uses Parker)]
    A --> C[Object.wait/notify <br> (monitor in JVM)]
    B --> D[Under the Hood <br> pthread_cond_t & pthread_mutex_t <br> Semaphores / Condition Vars]
    C --> D
    D --> E[OS Kernel <br> Schedulers, Syscalls]
```

*(Adapted from Princeton’s CS concurrency course slides.)*

---

## 7. Interruptible

### 7.1 Interrupt Flag & Waking a Blocked Thread
- **`Thread.interrupt()`** sets an internal **interrupt flag**.
- If the thread is blocked on `park()`, it checks the interrupt flag on wake-up and throws `InterruptedException` if set.
- For `wait()`, the logic is similar: the blocked thread also checks its interrupt status after being awakened.

### 7.2 Special Case: `Selector.select()`
- In NIO, `Selector.select()` can block for I/O events. Interrupting a thread blocked in an I/O call is trickier:
  - Java uses an **`Interruptible`** object at the native layer to close the channel if an interrupt happens.
  - Example:
    ```java
    protected final void begin() {
        if (interruptor == null) {
            interruptor = new Interruptible() {
                public void interrupt(Thread target) {
                    // close the channel forcibly
                }
            };
        }
        blockedOn(interruptor);
        if (Thread.currentThread().isInterrupted())
            interruptor.interrupt(Thread.currentThread());
    }
    ```
- Once interrupted, the channel is closed, causing the blocked I/O to return with an exception. This is more expensive but ensures correct interrupt semantics.

---

## 8. Summary

1. **Multi-Threading** in Java:
   - Maps to **native threads**.
   - Complex but historically common approach to concurrency.
2. **JUC**:
   - Provides advanced synchronizers built on **AQS**.
3. **Blocking in Java**:
   - `wait/notify` → implemented in the JVM with **object monitors**.
   - `park/unpark` → implemented with **`pthread_cond_t`** and **pthread_mutex_t** in HotSpot.
4. **OS Primitives**:
   - **Semaphores** (Dijkstra, 1965) – atomic wait/signal operations.
   - **Monitors** (Hoare, 1974) – higher-level synchronization, use **condition variables**.
   - **POSIX** provides `pthread_cond`, `pthread_mutex`, `sem_t`.
5. **Hoare vs. Mesa**:
   - Java uses **Mesa** semantics → `notify` just signals, the awakened thread must re-acquire the lock and re-check condition.
6. **Interruptibility**:
   - Java uses a **flag check** approach.  
   - Special logic for I/O blocking (`Selector`).
Being aware of the underlying OS mechanisms helps understand **why** Java concurrency behaves as it does (e.g., waiting threads must re-check conditions, `InterruptedException` is thrown after re-acquiring locks, etc.).

---
## 9. References

- [Carl Mastrangelo - Java’s Mysterious Interrupt](https://carlmastrangelo.com/blog/javas-mysterious-interrupt)
- **Analysis of Java’s LockSupport.park() implementation**  
- Courseware **COMP3151/9151** Foundations of Concurrency (Semaphores, Monitors, POSIX threads, Java)  
- [Semaphores & Monitors Slides (CS61 - Harvard)](https://cs61.seas.harvard.edu/)  
- **Mutexes and Semaphores Demystified**  
- [*Operating System Concepts* by Silberschatz, Galvin, Gagne](https://book.douban.com/subject/1888733/)  
- [Wikipedia: Mutual Exclusion](https://en.wikipedia.org/wiki/Mutual_exclusion)  
- [StackOverflow - Condition Variable vs. Semaphore](https://stackoverflow.com/questions/3513045/)  
- [StackOverflow - Lock vs. Mutex vs. Semaphore, what’s the difference?](https://stackoverflow.com/questions/2332765/)  

```