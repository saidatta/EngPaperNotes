tags: [DeterministicSimulation, DistributedSystems, DST, Testing]
> **Context**: In distributed systems, concurrency and nondeterminism can cause subtle or hard-to-reproduce bugs. **Deterministic Simulation Testing** (DST) is an approach that runs multiple nodes of a system in a **single-thread** environment with a controlled clock and controlled randomness, enabling reproducible test runs (by reusing the same seed) and systematic fault injection.

This note summarizes key points, potential benefits, and caveats of DST, referencing examples from FoundationDB, TigerBeetle, Antithesis, and more.
## Table of Contents
1. [Overview](#overview)  
2. [Randomness and Time](#randomness-and-time)  
3. [Converting Existing Functions to DST](#converting-existing-functions-to-dst)  
   - [Example: Retry with Backoff](#example-retry-with-backoff)
4. [Single Thread and Async I/O](#single-thread-and-async-io)
5. [A Distributed System in One Process](#a-distributed-system-in-one-process)
6. [Other Sources of Non-Determinism](#other-sources-of-non-determinism)
7. [Considerations and Limitations](#considerations-and-limitations)
   - [7.1 The Edges](#71-the-edges)
   - [7.2 Crafting Workloads](#72-crafting-workloads)
   - [7.3 Mocking Real-World Behavior](#73-mocking-real-world-behavior)
   - [7.4 Code Changes Breaking Seeds](#74-code-changes-breaking-seeds)
   - [7.5 Performance and Overheads](#75-performance-and-overheads)
8. [What About Jepsen?](#what-about-jepsen)
9. [Conclusion](#conclusion)
10. [Further Reading](#further-reading)

---
## 1. Overview
**Traditional** debugging of distributed systems is difficult: concurrency, partial failures, network reorderings, and so on produce “heisenbugs” that might not be reproducible. Even if you see an error once, you might not see it again.
**Deterministic Simulation Testing** proposes:
1. **Single-thread** all distributed nodes in one process.  
2. Control **randomness** (PRNG with known seeds) → guaranteed reproducibility if seed is the same.  
3. Control **time** (logical or simulated clock).  
4. In the simulator, systematically inject **faults** (like network drops, partial writes, crashes) in a reproducible manner.

Hence, when a test fails at seed X, you can **rerun** with the same seed and reliably produce the same error. This is a huge advantage for debugging.

---
## 2. Randomness and Time
Real systems rely on random numbers (e.g., random scheduling decisions, random backoff) and system clock or timers for time-based logic. DST doesn't forbid them; it just **injects** or **manages** them:
- A **global seed** for random number generation across the entire system, making random calls deterministic given that seed.
- A **controlled clock** for time-based logic, so the system’s notion of time is advanced by the simulator, not real OS time.

**Reproduce** a bug → you just re-run the entire system with the same seed and same “time script.”

---
## 3. Converting Existing Functions to DST
### Example: Retry with Backoff
Consider this snippet:
```pseudo
# retry.pseudocode

class Backoff:
  def init:
    this.rnd = rnd.new(seed = time.now())  # nondeterministic
    this.tries = 0

  async def retry_backoff(f):
    while this.tries < 3:
      if f():
        return
      await time.sleep(this.rnd.gen())
      this.tries++
```

**Problem**: `time.now()` is uncontrollable. Also `time.sleep(...)` uses real time. We want them to be **simulated**. So we pass in a clock or RNG as parameters:

```pseudo
class Backoff:
  def init(this, clock, seed):
    this.clock = clock
    this.rnd = rnd.new(seed)
    this.tries = 0

  async def retry_backoff(f):
    while this.tries < 3:
      if f():
        return
      await this.clock.sleep(this.rnd.gen())
      this.tries++
```

Now in the simulator, we can forcibly set `seed` and manipulate `clock.sleep(...)` to run in a single-threaded environment with discrete “ticks.”

---
## 4. Single Thread and Async I/O
**True** concurrency is tricky. DST can handle concurrency by forcing all code to be single-thread asynchronous. The simulator controls the *scheduling* of tasks. If you're using a language runtime (like Go, Rust tokio, or Node.js), you might have to carefully adapt or patch the runtime for deterministic scheduling.

**Hence** many DST projects avoid “real threads” in the test environment, or they rely on specialized patches (like polar signals’ approach for Go or custom libraries in Rust).

---
## 5. A Distributed System in One Process
**Multiple** nodes are run as separate modules in the same process, each with its own “simulated” network and disk. The simulator schedules them in a single-thread event loop:
- Node A calls “send,” the simulator logs a “send” event, eventually the simulator “delivers” that message to Node B at a random or deterministic time as an event → B’s code runs.

**Large external services** (like actual Kafka) are typically replaced by in-process mocks. Real OS-level concurrency is replaced by a simulated environment.

---
## 6. Other Sources of Non-Determinism
**CPU instruction reordering** or hardware subtlety might cause nondeterminism. Without a hypervisor-level approach (like rr, Hermit, or Antithesis) to intercept them, you can’t fully ensure bitwise equivalence. However, many DST practitioners accept partial determinism. They avoid instructions that break it or rely on an environment that’s “close enough.”

---
## 7. Considerations and Limitations
### 7.1 The Edges
DST usually *omits* or *mocks* the edges: system calls, external DBs, third-party services. So you’re only truly testing your “core logic.” That’s good for your distributed algorithm correctness. But you still might miss real integration issues.
### 7.2 Crafting Workloads
**Sim** and fuzz-based random seeds are only as good as the coverage of workloads you design. You must carefully produce or randomize plausible sequences of operations, entity manipulations, etc. This can be quite labor-intensive.
### 7.3 Mocking Real-World Behavior
Your simulator must define how network, disk, or error injection can happen. Are we dropping half the packets? Are we occasionally returning corrupt data? If the real environment has corner cases you didn’t model, you may miss them.
### 7.4 Code Changes Breaking Seeds
When the code changes, the sequence of states at the same seed might not produce the same bug. Reproducibility is strongly version-specific. Usually you store the seed with the commit or keep a regression test.
### 7.5 Performance and Overheads
You might re-run your entire system many times. Large test coverage can be expensive. Also, single-thread simulation can be slow for a big system, but typically it’s faster than a real cluster. So it might be a net gain.

---
## 8. What About Jepsen?
**Jepsen** is a black-box approach for distributed DBs, combining fault injection (like partitioning, reintroducing nodes, etc.) with correctness checks (like linearizability). But it doesn’t ensure deterministic replays. If Jepsen finds a bug once, you might not replicate that scenario easily.

DST is more **fine-grained** and allows easy bug reproduction from the same seed. Meanwhile, Jepsen tests your actual cluster environment with real concurrency. They complement each other.

---
## 9. Conclusion

**Deterministic Simulation Testing** is a powerful technique to find and reproduce concurrency bugs in distributed systems. By:

- Forcing single-threaded concurrency
- Providing a controlled clock & random seed
- Injecting errors systematically

… developers can get consistent replays of complex failures. However, it’s no silver bullet:

- You still must carefully mock external components, manage partial determinism, craft robust workloads, and re-run tests as the code changes.
- Tools like **Antithesis** or **rr** can help at the hypervisor or OS-level, but they’re complex or incomplete for certain use cases.  

**Despite** these challenges, DST has proven extremely effective at organizations like FoundationDB or TigerBeetle, revealing deep concurrency or state-machine bugs that typical tests or fuzzing might never discover. If you can incorporate it into your system design from the start (especially controlling time + randomness with a well-structured asynchronous model), the payoff can be huge for reliability.

---

## Further Reading

- Will Wilson’s [“Testing Distributed Systems w/ Deterministic Simulation”](https://youtu.be/----)
- Tyler Neely & Pekka Enberg discussions on DST
- [“(Mostly) Deterministic Simulation Testing in Go”](https://resoundingly.com/)
- TigerBeetle’s approach: [TigerBeetle official site](https://tigerbeetle.com/)  