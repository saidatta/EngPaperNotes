### 5. A New Operating System Course

We designed a new operating system course from scratch, based on "The Three Easy Pieces" and the teaching philosophies:
- Everything is a state machine
- Emulate state machines with executable models
- Enumeration demystifies operating systems

This section outlines the impacts of the state machine perspective (Section 5.1) and the model checker (Section 5.2) on the course design, followed by discussions (Section 5.3).

---

### 5.1 State Machines and Operating Systems

Introducing key OS concepts using state machines has several advantages, providing a coherent understanding of computer systems.

#### Debugging Real Systems
Students often struggle with debugging real systems, even minimal ones. The state-machine perspective helps students understand that all bugs are anomalies in the state-machine's execution trace. Given enough time, one can find the root cause by identifying the first abnormal state. This principle, although impractical for large systems, encourages clever debugging techniques.

**Example: printf-debugging**
- Provides a high-level digest of the state-machine trace
- Narrows down the scope of the initial anomalous state
- Defensive programming with assertions

**Classroom Story: Diagnosing 100% CPU Usage**
- **Issue:** Unexpected CPU usage in an idle workload
- **Tool:** `perf` tool identified a hotspot in an `xhci`-related function
- **Root Cause:** A short-circuited USB port

#### Concurrency and State Machines
Concurrency is a significant topic in OS courses. Model checking represents concurrent programs as state transition systems and is computationally intensive due to state explosion.

**Example: Data Race**
- Data race: simultaneous access of a shared memory location by two threads/processors (with at least one performing a write)
- **Exercise:** Students migrate a process between processors, avoiding data races on the kernel's interrupt stack.

#### Demystifying Compilers
C programs can also be represented by state transition systems. Using a non-recursive "Tower of Hanoi" implementation, we illustrate that the "runtime state" of C programs includes static variables, heap memory, and stack frames.

**Example: Non-Recursive Tower of Hanoi**
```c
void hanoi(int n, int from, int to, int via) {
    if (n == 1) {
        printf("%d -> %d\n", from, to);
    } else {
        hanoi(n - 1, from, via, to);
        hanoi(1, from, to, via);
        hanoi(n - 1, via, to, from);
    }
}

typedef struct { int pc, n, from, to, via; } Frame;
#define call(...) ({*(++top) = (Frame) {0, __VA_ARGS__};})
#define ret() ({top--;})
#define jmp(loc) ({f->pc = (loc) - 1;})

void hanoi_nr(int n, int from, int to, int via) {
    Frame stk[64], *top = stk - 1, *f;
    call(n, from, to, via);
    while ((f = top) >= stk) {
        switch (f->pc) {
            case 0: if (f->n == 1) {
                printf("%d -> %d\n", f->from, f->to); jmp(4);
            } break;
            case 1: call(f->n - 1, f->from, f->via, f->to); break;
            case 2: call(1, f->from, f->to, f->via); break;
            case 3: call(f->n - 1, f->via, f->to, f->from); break;
            case 4: ret(); break;
            default: assert(0);
        }
        f->pc++;
    }
}
```
**Visualization: State Machine Perspective of C Programs**
```plaintext
Stack Frames
hanoi(PC=5) n=3, f=1, t=3, v=2
hanoi(PC=6) n=3, f=1, t=3, v=2
hanoi(PC=6) n=3, f=1, t=3, v=2
hanoi(PC=6) n=3, f=1, t=3, v=2
hanoi(PC=1) n=2, f=1, t=2, v=3
```

---

### 5.2 Modeling and Model Checking in Action

**Models and Emulation**
Models in an OS course need not be limited to high-level languages. We advocate minimal but functionally "working" models.

**Example: Simplified ELF Dynamic Linker and Loader**
- Problem: ELF format complexity
- Solution: Simplified binary format using GCC and binary utilities
- Implementation: 200 lines of C code

**Formal Methods in OS**
We introduce formal methods by making students draw a state transition graph to prove the safety and liveness of Peterson’s mutex algorithm. MOSAIC automates this process, replacing human labor with emulation. The interactive visualizer in a Jupyter notebook received positive feedback.

**Gap Between Models and Real Systems**
Models do not fully reflect real systems but make assumptions explicit. For example, Peterson’s algorithm assumes a sequentially consistent memory model.

---

### 5.3 Student Acceptance and Discussions

**Student Feedback**
- Positive reception of the course materials
- Comments on the comprehensive explanation of OS principles
- Industry professionals support the state-machine approach

**Usefulness of Models**
Models help establish concepts and control complexity by hiding low-level details. Executable models make OS concepts rigorous.

**Limitations**
- Over-simplification may lead to overlooking real-world challenges
- "Hands-on approach" remains essential
- MOSAIC’s limitations: main must be a Python generator, deterministic program assumptions

---

### Conclusion

The new operating system course, based on state machine perspectives and executable models, has been well-received. The integration of MOSAIC model checker enhances understanding and provides a rigorous, interactive approach to teaching OS principles.

This comprehensive guide integrates state transition systems and exhaustive enumeration into OS education, offering a rigorous, interactive, and practical approach to understanding and teaching operating systems.