### 3. An Executable Operating System Model

**Philosophy 2: Emulate State Machines with Executable Models**

State machines are not only theoretical constructs but can also be developed as executable models that emulate the behavior of processes and operating systems. This section discusses implementing a lightweight executable operating system model using modern programming languages, simplifying non-trivial textbook cases, and using models as behavioral specifications of real systems.

#### 3.1 Emulating an Operating System
##### State Machines (Processes) and System Calls

The OS model is implemented in Python, using generators (stackless coroutines) to emulate processes. Each process's memory is represented by its local variables. System calls are emulated by `yield`, saving the local state and transferring control back to the caller.

**Example: Emulated Application Process**
```python
def main(msg):
    i = 0
    while (i := i + 1):
        yield 'SYS_write', msg, i  # write(msg, i)
        yield 'SYS_sched'          # sched()
```

**Operating System Main Loop**
```python
class OperatingSystem:
    def __init__(self, procs):
        self._procs = procs
        self._current = procs[0]

    def run(self):
        while True:
            syscall, *args = self._current.__next__()
            match syscall:
                case 'SYS_write':
                    print(*args)
                case 'SYS_sched':
                    self._current = random.choice(self._procs)

OperatingSystem([main('ping'), main('pong')]).run()
```

##### Process APIs
Deep-copying a generator object is not allowed in Python. Therefore, `fork()` is implemented by creating a new `OperatingSystem` object and replaying all executed system calls to obtain a deep copy of the process. 

**System Calls Overview**
| System Call | Description |
|-------------|-------------|
| `fork()` | Create current thread’s heap and context clone |
| `spawn(f, xs)` | Spawn a heap-sharing thread executing `f(xs)` |
| `sched()` | Switch to a non-deterministic thread |
| `choose(xs)` | Return a non-deterministic choice among `xs` |
| `write(xs)` | Write strings `xs` to standard output |
| `bread(k)` | Return the value of block `k` |
| `bwrite(k, v)` | Write block `k` with value `v` to a buffer |
| `sync()` | Persist all outstanding block writes to storage |
| `crash()` | Simulate a non-deterministic system crash |

##### Threads and Shared Memory
Shared memory among threads is emulated by the global `heap` variable, updated before switching processes/threads. `spawn(f, *xs)` creates a new generator with shared heap, while replay-based `fork()` obtains a deep copy of the heap.

##### Devices
Writing to the debug console appends the message to a buffer. Reading is implemented by `choose()` from possible inputs. The emulated block device is a key-value mapping, simulating real disks with a volatile buffer.

#### 3.2 Modeling Operating System Concepts
Executable models simplify non-trivial textbook cases, helping to debug and reproduce complex interactions across system layers. 

**Example: Fork and Buffer Mechanism**
```c
for (int i = 0; i < 2; i++) {
    int pid = fork();
    printf("%d\n", pid);
}
```
**Output:**
```
1000
1001
0
0
1002
0
```
**Explanation:**
The above example illustrates the non-trivial behavior of `fork()` related to buffer mechanisms in standard C libraries.

**Emulated Model:**
```python
def main():
    heap.buf = ''
    for _ in range(2):
        pid = sys_fork()  # heap.buf is deeply copied
        sys_sched()       # non-deterministic context switch
        heap.buf += f'{pid}\n'
    sys_write(heap.buf)  # flush buffer at exit
```

**Understanding Synchronization**
```python
def Tworker(name, delta):
    for _ in range(N):
        while heap.mutex == ' ':  # mutex_lock()
            sys_sched()           # spin wait
        heap.mutex = ' '
        while not (0 <= heap.count + delta <= BUFSIZE):
            sys_sched()
        heap.mutex = ' '          # cond_wait()
        heap.cond.append(name)
        while name in heap.cond:  # spin wait
            sys_sched()
        while heap.mutex == ' ':  # reacquire lock
            sys_sched()
        heap.mutex = ' '
        if heap.cond:             # cond_signal()
            t = sys_choose(heap.cond)
            heap.cond.remove(t)
            sys_sched()
        heap.count += delta       # produce or consume
        heap.mutex = ' '          # mutex_unlock()
        sys_sched()

def main():
    heap.mutex = ' '
    heap.count = 0
    heap.cond = []
    sys_spawn(Tworker, 'Tp', 1)   # producer
    sys_spawn(Tworker, 'Tc1', -1) # consumer
    sys_spawn(Tworker, 'Tc2', -1) # consumer
```

**File System Consistency and Journaling**
```python
def main():
    head = sys_bread(0)  # log the write to block #B
    free = max(log.values(), default=0) + 1  # allocate log
    sys_bwrite(free, f'contents for #{B}')
    sys_sync()

    head = head | {B: free}  # write updated log head
    sys_bwrite(0, head)
    sys_sync()

    for k, v in head.items():  # install transactions
        content = sys_bread(v)
        sys_bwrite(k, content)
        sys_sync()

    sys_bwrite(0, {})  # clear log head
    sys_sync()
```

#### 3.3 Application: Specification of Systems

Models serve as high-level reference implementations and behavioral specifications for real systems. They help verify properties like safety and liveness in mutex models.

**Model Checking Algorithm**
```python
Q = {[]}  # queue of traces pending checking
S = set()  # set of checked states
while Q:
    tr = Q.pop()
    s, choices = replay(tr)
    if s not in S:
        S.add(s)  # add the unexplored state to S
        for c in choices:
            Q.append(tr + [c])  # extend tr with c and append to Q
```

---

This comprehensive guide integrates state transition systems into OS education, providing a rigorous, interactive, and practical approach to understanding and teaching operating systems.