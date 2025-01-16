tags: [PhD, DistributedSystems, Testing, DST, Simulation]
# Overview
Distributed systems are notoriously **difficult to test**. Interactions can be **chaotic** and **non-reproducible**, making it challenging to track down and fix bugs. A single ephemeral network glitch or timing-related event might cause a rare failure that is **virtually impossible** to recreate in a normal test environment. 

**Deterministic Simulation Testing (DST)** aims to solve this by allowing you to:
1. **Consolidate** multiple nodes of a distributed system into a **single-threaded** test harness.
2. **Control** or **inject** sources of non-determinism (randomness, time, I/O errors, network faults).
3. **Replay** failures with an **identical seed** for the random number generator (and clock), enabling **bug reproduction** with near-100% fidelity.

While DST is powerful, it also comes with **limitations** and **significant effort** to mock out or control real-world behaviors (disk, network, concurrency). These notes explore the main concepts, examples, and trade-offs.

---

# Table of Contents
1. [Key Principles of DST](#key-principles-of-dst)
2. [Randomness and Time Control](#randomness-and-time-control)
3. [Dependency Injection for DST](#dependency-injection-for-dst)
4. [Single-Threaded Execution + Async I/O](#single-threaded-execution--async-io)
5. [Examples](#examples)
   - [Example 1: Retrying with Backoff](#example-1-retrying-with-backoff)
   - [Example 2: File Reads with Fault Injection](#example-2-file-reads-with-fault-injection)
   - [Example 3: Multi-Node Distributed System](#example-3-multi-node-distributed-system)
6. [Considerations and Drawbacks](#considerations-and-drawbacks)
   - [Edge Behaviors & Mock Boundaries](#1-edge-behaviors--mock-boundaries)
   - [Workload Generation](#2-workload-generation)
   - [Mock Knowledge & Accuracy](#3-mock-knowledge--accuracy)
   - [Non-Reproducible Seeds After Code Change](#4-non-reproducible-seeds-after-code-change)
   - [Time & Compute Costs](#5-time--compute-costs)
7. [Comparisons with Other Approaches (e.g., Jepsen)](#comparisons-with-other-approaches-eg-jepsen)
8. [Conclusion](#conclusion)
9. [References & Further Reading](#references--further-reading)

---

## Key Principles of DST
1. **Run everything in one process (or single host).** Each node in your “distributed” system is just another instance of your application logic but **wrapped** in a single simulation environment.
2. **Control concurrency**: Rely on **asynchronous** instead of multi-threaded concurrency. The simulator orchestrates all events (like I/O calls, network sends, sleeps) in a **deterministic** order.
3. **Inject** or **mock** **randomness**, **time**, **I/O**, **network** errors, etc. The test harness ensures that each test run is the only arbiter of how these “external” factors behave.
4. **Replay**: Because the same seed and event scheduling lead to the same sequence of events, you can **reproduce** a discovered bug exactly, drastically simplifying debugging.

---

## Randomness and Time Control
Two major sources of **chaos** in real systems are:
1. **Random numbers** (e.g., for backoff, shuffling, or unique ID generation).
2. **Time** (e.g., delays, scheduling, time-based logic).

**DST** doesn’t ban them; it just **centralizes** and **controls** them:
- A **global seed** is used by the simulator to drive all random sequences.  
- A **mock clock** that can be **manually advanced** to simulate time passage, delays, or timeouts.

When a failure occurs under a particular seed or clock schedule, re-running with the same parameters produces the same system state for **debugging**.

---

## Dependency Injection for DST
To isolate the system from “real” hardware or OS, you usually **inject**:
- **Clock**: Instead of `time.now()`, pass a `clock` object that returns `sim_time`.
- **Randomness**: Instead of `random()`, pass a `rnd` or `seed` object that is controlled by your simulator.
- **Network**: Instead of calling raw sockets, pass a `network` object with instrumented send/receive.
- **Disk**: Instead of reading/writing from a real filesystem, pass a `disk` object with instrumented read/write.

**Language Example**: A pseudo-code snippet for controlling the clock and random seed:

```pseudo
def start(clock, seed):
  # Actual app logic, references clock.now(), rnd(), etc.
  pass

def main:
  # Production path
  real_clock = system_clock()
  real_seed  = random_seed()
  start(real_clock, real_seed)

def sim_main:
  # DST path
  sim_clock = simulate_clock()
  sim_seed  = environment_or_default_seed()
  start(sim_clock, sim_seed)
```

---

## Single-Threaded Execution + Async I/O
Most DST frameworks **cannot** handle multi-threaded concurrency in a straightforward manner without a specialized kernel or hypervisor to intercept all system calls. Instead, DST often assumes a **single-threaded** event loop with **asynchronous** I/O. 

- **Go** runtime example: Some projects have forked or heavily modified the Go runtime to remove internal concurrency randomness.  
- **Rust** example: The [turmoil](https://github.com/tokio-rs/turmoil) project tries to provide a **deterministic** scheduler for asynchronous tasks in Rust.  
- **Custom** approach: Many implement a minimal event loop or coroutines where all **yield** points are under simulator control.

---

## Examples

### Example 1: Retrying with Backoff
A common scenario: a function retriest until success, with backoff and random sleeps. Here’s a simplified example:

```pseudo
# retry.pseudocode

class Backoff:
  def init(clock):
    this.clock = clock
    this.rnd   = rnd.new(seed = this.clock.now())  # seeded by "now" for example
    this.tries = 0

  async def retry_backoff(f):
    while this.tries < 3:
      if f():
        return
      delay_ms = this.rnd.gen()
      await this.clock.sleep(delay_ms)
      this.tries++
```

**Key DST Points**:
- **Inject** `clock` and `rnd`.  
- The test harness can systematically vary return values of `f()` and the magnitude of `delay_ms`.

**Simulation** might look like:

```pseudo
# sim.psuedocode
import "retry.pseudocode"

sim_clock = SimClock()       # custom simulation clock
seed      = get_seed_from_env_or_now()
rnd       = Rnd(seed)

# Create the backoff instance with controlled clock
backoff = Backoff(sim_clock)

# Randomly fail ~50% of the time
failures = 0
f = () => {
  if rnd.rand() > 0.5:
    failures++
    return false
  return true
}

# Step through the simulation
promise = backoff.retry_backoff(f)
for t in 0..9999:
  sim_clock.tick(1)
  if promise.resolved():
    break

assert_condition_on_failures(failures)
```

If a bug arises, the **same** `seed` can be used to replicate the exact sequence.

---

### Example 2: File Reads with Fault Injection
Consider a function that reads a file in chunks, possibly facing partial reads or end-of-file:

```pseudo
# readfile.pseudocode

async def read_file(io, name, into_buffer):
  file_handle = await io.open(name)
  temp_buf    = [4096]byte
  while true:
    err, n_read = await file_handle.read(temp_buf)
    if err == io.EOF:
      break
    if err != nil:
      throw err

    # Bug: incorrectly copying the entire 4096, not the actual read size
    into_buffer.append(temp_buf[0 : 4096])
```

**Simulation** with forced partial reads:

```pseudo
# sim_readfile.pseudocode

seed = os.env.SEED or time.now()
rnd  = Rnd(seed)

sim_disk_data = rnd.rand_bytes(10MB)
sim_io = {
  open: (filename) => sim_filehandle
}
sim_filehandle = {
  pos: 0,
  async read(buf):
    # randomly read 0..buf.size bytes
    read_len = rnd.rand_in_range(0, buf.size)
    # simulate partial read
    copy(sim_disk_data[pos : pos + read_len], buf[0 : read_len])
    pos += read_len
    if pos >= sim_disk_data.size:
      return (EOF, read_len)
    return (nil, read_len)
}

into_buffer = []
try:
  await read_file(sim_io, "testfile", into_buffer)
  # This assertion fails if the bug triggers
  assert_equal(into_buffer.size, sim_disk_data.size)
catch e:
  print("Bug discovered with seed:", seed)
  throw e
```

**Result**: The code appends all 4096 bytes each time instead of the actual read length. The DST environment reveals the bug and makes it reproducible with the same seed.

---

### Example 3: Multi-Node Distributed System
For a self-contained distributed system (e.g., a replicated key-value store using Raft), you can run **multiple node instances** in the **same** process. Each node:
- Has a **mock** network stack to send/receive messages.
- Has a **mock** disk or state.

You randomize:
- **Packet drops**, **latency** injection.
- **Disk errors** (read/write failures).
- **Node restarts** or crashes.

```pseudo
# sim_distsys.pseudocode

seed = get_seed()
rnd  = Rnd(seed)

simulate_net = new MockNetwork(rnd)
simulate_disk = new MockDisk(rnd)

nodes = [
  start_node(id=0, net=simulate_net, disk=simulate_disk, seed=seed),
  start_node(id=1, net=simulate_net, disk=simulate_disk, seed=seed),
  start_node(id=2, net=simulate_net, disk=simulate_disk, seed=seed)
]

history = []
try:
  for i in 1..100:
    key   = rnd.rand_bytes(16)
    value = rnd.rand_bytes(16)
    # insert into random node
    nodes[rnd.rand_int(0,2)].insert(key, value)
    history.add((key, value))

    # random node crash?
    if rnd.rand() < 0.2:
      crashed_node = rnd.choice(nodes)
      crashed_node.restart()

  # Check consistency invariants
  for node in nodes:
    assert_consistent_state(node, history)

except e:
  print("Error discovered, seed =", seed)
  throw e
```

When a bug arises (e.g., a missing update after a certain crash order), the same seed replays all events identically.

---

## Considerations and Drawbacks
DST is powerful but **not** a panacea. Several caveats apply:

### 1. Edge Behaviors & Mock Boundaries
- You inevitably **mock** or intercept system calls like disk, network, time.  
- **Real** systems (Kafka, Postgres, Redis, etc.) are not trivially included in the single process. The more external dependencies, the less realistic your simulation.  
- If you rely on a **partial** mock (e.g., you intercept network calls but not disk I/O), you might miss certain classes of disk faults.

### 2. Workload Generation
- Similar to **fuzzing** or **property-based testing**, you must carefully design **meaningful workloads**.  
- It’s easy to set a random distribution that never triggers the corner cases you want to test.  
- Tuning simulation parameters (latency, error rates, concurrency patterns) is an **iterative**, **science-like** process.

### 3. Mock Knowledge & Accuracy
- If your mocks don’t account for weird real-world behaviors (data corruption, partial writes, out-of-order bytes), you may not uncover certain failure modes.
- Real hardware or OS-level concurrency issues are partially out of scope in simpler DST harnesses (unless using advanced solutions like **Antithesis**).

### 4. Non-Reproducible Seeds After Code Change
- A seed that triggers a failure in version X of your code might not do the same in version X+1.  
- However, it’s still extremely valuable for debugging the bug in version X, and can be turned into a stable test scenario for future regressions.

### 5. Time & Compute Costs
- You might need to run large numbers of random seeds to achieve good coverage.  
- Each new code commit might invalidate previous seeds. Maintaining a continuous integration pipeline that runs multiple seeds or a **24/7** fuzzing-like environment is common.

---

## Comparisons with Other Approaches (e.g., Jepsen)
- **Jepsen**: Great for black-box testing of external systems (like databases) with limited fault injection (network partitions, process kills). But **not** deterministic.  
- **DST**: Potentially deeper coverage (you can inject *any* fault you can simulate) **and** perfect reproducibility. But it’s a bigger engineering lift to embed your entire system in a single test harness.

---

## Conclusion
**Deterministic Simulation Testing** is a **powerful strategy** for building confidence in complex distributed systems. By **controlling** randomness, time, and concurrency, you can systematically discover and **reproduce** elusive bugs. However, DST:

1. Requires **significant engineering effort** to mock or wrap system calls.  
2. Benefits heavily from well-designed **test workloads**.  
3. Cannot fully simulate all real-world conditions if you rely on too many external dependencies.  

Despite these hurdles, many systems (FoundationDB, TigerBeetle, Polar Signals) have used DST to achieve **remarkable** reliability. The key is **understanding** both the **capabilities** and **limitations** so you can tailor DST to your unique distributed application.

---

## References & Further Reading

1. **FoundationDB** - Known for its *Simulation* approach, credited with extremely few bugs discovered in production.  
2. **TigerBeetle** - High-performance accounting database using a deterministic simulator for disk/network fault injection.  
3. **Antithesis** - Advanced approach that uses hypervisor-level control for deterministic execution + fault injection.  
4. **Turmoil (tokio-rs)** - Early experimental library for deterministic scheduling in Rust’s async environment.  
5. “**Jepsen**” - A well-known external black-box fault injection framework (not deterministic, but useful for verifying consistency properties).  
6. **Polar Signals** DST approach - Compiling Go to WASM and forking the runtime to control scheduling.

> “**It’s terrifyingly easy to build a DST system that appears to be doing a ton of testing, but never explores very much of the state space.**” — Will Wilson (FoundationDB)

```