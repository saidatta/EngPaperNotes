https://www.usenix.org/system/files/atc23-jiang-yanyan.pdf
**Author:** Yanyan Jiang, Nanjing University
#### Abstract
This paper introduces a principled approach to teaching operating systems (OS), emphasizing state transition systems. It presents a minimal OS model with nine system calls covering process-based isolation, thread-based concurrency, and crash consistency. This model includes a checker and interactive state space explorer to examine all possible system behaviors exhaustively.

---
### 1. Introduction
Albert Einstein once said, "Everything should be made as simple as possible, but no simpler." This philosophy underpins the teaching of OS design and implementation. The foundation, established by texts like Tanenbaum’s *Operating Systems: Design and Implementation* and Arpaci-Dusseau’s *Operating Systems: Three Easy Pieces*, approaches OS as layered abstractions over processors, memory, and storage.

Modern systems emphasize speed, scalability, reliability, and security, driven by innovations like hardware/software co-design, cross-stack integration, program analysis, and formal methods. This paper unifies these themes using state transition systems to bridge theoretical concepts and practical implementations.
### 2. State Machines: First-class Citizens of Operating Systems
#### Philosophy 1: Everything is a State Machine
The paper's key idea is considering state transition systems central to OS teaching. This abstraction is essential for understanding the state of modern multi-processor systems and multi-threaded programs.
**Operating System as State Machine Manager:**
- The OS multiplexes CPUs across processes and threads using data structures like page tables.
- Process management APIs (fork, execve, exit) manipulate state machines.
**Example: Hello World Program**
```c
#include "sys/syscall.h"
mov $SYS_write, %rax // write(
mov $1, %rdi // fd=1,
mov $hello, %rsi // buf=hello,
mov $16, %rdx // count=16
syscall // );
mov $SYS_exit, %rax // exit(
mov $1, %rdi // status=1
syscall // );
hello: .ascii "Hello, OS World\n"
```
**State Machine Visualization:**
```
Initial State: s0
Registers: rax = SYS_write, rdi = 1, rsi = hello, rdx = 16
Memory: [0x... = "Hello, OS World\n"]
Transition: syscall -> State: s1
```
In this example:
- The program starts in an initial state `s0`.
- The state of the system includes the values in the CPU registers and memory.
- The `syscall` instruction causes a transition from `s0` to `s1`, changing the state of the system.![[Screenshot 2024-07-01 at 12.42.30 PM.png]]
- ![[Screenshot 2024-07-01 at 12.42.57 PM.png]]
### 3. Emulating State Machines with Executable Models
**Executable Model Advantages:**
1. Foundation for exploring OS concepts (synchronization, fork behavior, file system consistency).
2. Behavioral specification of real OS (motivates formally verified systems like seL4, Hyperkernel).

**System Calls Overview:**
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
### 4. Enumeration and MOSAIC
**Enumeration for Demystifying OS:**
- The emulator handles non-determinism by returning sets of possible choices as callbacks.
- MOSAIC (Modeled Operating System And Interactive Checker) uses lightweight formal methods to check process parallelism, shared memory concurrency, and crash consistency.
### Detailed Examples and Concepts

#### 2.1 Introducing State Machines in the Operating System Class
**State Machine Abstraction:**
- States defined by register/memory values.
- Transitions as single-step instruction executions.
**Code Example: Process Initial State Inspection in GDB**
```bash
gdb a.out
(gdb) stepi
(gdb) info registers
(gdb) x /10x $sp
```
#### 2.2 Operating System as a State Machine Manager
**Visualization: OS Managing State Transitions**
```
Process A       Process B
  0 1 2 3        0 1 2 3
  . . . .        . . . .
OS (State Manager)
  0 1 2 3 4 5 6 7
   A B A B A B
```

**Process APIs Explained:**
- `fork()`: Copies current state machine.
- `execve()`: Resets state machine to a new program.
- `exit()`: Removes state machine from OS.
#### 2.3 State Machines Meet Operating Systems
**Advanced Concepts:**
1. **Time-travel Debugging:**
   - Record state differences for reverse execution.
   - Example: GDB's reverse debugging.
2. **Trace and Profiling:**
   - Insert probes at function calls/returns.
   - Tools: `ftrace`, `Kprobe`.
3. **Runtime Checkers:**
   - AddressSanitizer, ThreadSanitizer, Lockdep.
   - Verify against specific bug patterns.
4. **Symbolic Execution:**
   - Enumerate all possible states using symbolic values.
   - Example: `x` representing any integer.

**Practical Implementation:**
```python
# Simulate state transitions in a minimal OS model
class OSModel:
    def __init__(self):
        self.state = 'initial'
        self.memory = {}
        self.threads = []

    def fork(self):
        new_thread = self.state.copy()
        self.threads.append(new_thread)
        return new_thread

    def execve(self, program):
        self.state = program.initial_state()
        self.memory = program.memory_layout()
    
    def exit(self):
        self.threads.remove(self.state)

# Simulate a basic program
program = OSModel()
program.execve(HelloWorldProgram)
program.fork()
```
### Conclusion

**Contributions:**
1. Proposed a "state-machine first" approach to OS teaching.
2. Designed and implemented MOSAIC, a minimal executable OS model.
3. Integrated these concepts into a popular undergraduate OS course.

**Resources:**
- [MOSAIC GitHub Repository](https://github.com/jiangyy/mosaic)

---

This comprehensive guide integrates state transition systems into OS education, providing a rigorous, interactive, and practical approach to understanding and teaching operating systems.