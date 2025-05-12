Great. I’ll create detailed Obsidian notes explaining how Windows 11 handles context switching at the kernel level, including for processes, threads, and interrupts. I’ll include relevant structures (like ETHREAD/KTHREAD), example tools (e.g., WinDbg), stack layouts, and code insights. I’ll also visualize how Windows uses kernel/user stacks, APCs, and processor control blocks during switches.

## Overview of Context Switching in Windows
In Windows, a **context switch** occurs when the CPU stops running one thread and starts running another. This involves saving the state (context) of the currently running thread and restoring the saved state of the next thread to run. The “context” includes the CPU registers (instruction pointer, general-purpose registers, etc.), stack pointers, and memory mapping info for the thread’s process. By performing context switches rapidly, Windows achieves multitasking – multiple threads (and processes) share the CPU effectively.

**Key aspects of a context switch** (on typical x86/x64 hardware) include saving the outgoing thread’s instruction pointer, CPU registers, and **kernel stack pointer**, and loading the incoming thread’s registers and stack pointer. If the new thread belongs to a different process, the **memory address space** must also be switched by updating the CPU’s page table base register (CR3) to point to the new process’s page directory. After this, execution resumes at the new thread’s saved instruction pointer. Context switches can be triggered by various events: a running thread completing its CPU time slice, a higher-priority thread becoming ready, a thread blocking on I/O, or an external interrupt that causes the scheduler to run. Each context switch has a cost (saving/loading CPU state, flushing memory translation caches, etc.), so the OS scheduler tries to optimize and minimize unnecessary switches.

**Windows 11’s kernel scheduler** (often called the **Dispatcher**) is responsible for deciding which thread runs next on each CPU and performing context switches. It maintains per-CPU ready queues of threads for each priority level and chooses the highest-priority ready thread for the next context switch. Context switching in Windows is highly optimized: it uses per-processor data structures, operates at elevated IRQL to synchronize, and employs techniques like lazy saving of floating-point state and per-CPU deferred ready lists to reduce overhead. We will delve into the types of context switches and the internal mechanisms and data structures (like **ETHREAD/KTHREAD**, **EPROCESS/KPROCESS**, and **KPRCB**) that make it all work.

## Types of Context Switches in Windows

From the Windows kernel perspective, there are three main types of context switches: **thread context switches**, **process context switches**, and **interrupt context switches**. They all involve switching execution from one context to another, but differ in scope and trigger:

### Thread Context Switch

A **thread context switch** is the most common type – the CPU switches from running one thread to another thread. In Windows, threads are the schedulable entities (not processes), so the scheduler always context-switches at the thread level. A thread switch can happen voluntarily or involuntarily:

- **Voluntary (Cooperative) Switch:** A thread may relinquish the CPU on its own, for example by waiting on an object or calling `Sleep()` or `SwitchToThread()`. In this case, the thread enters a wait state and the scheduler is invoked to run another ready thread.
    
- **Involuntary (Preemptive) Switch:** The kernel may preempt a running thread – for instance, when its **time quantum** (timeslice) expires or when a higher-priority thread becomes ready. Windows uses _priority-based preemptive scheduling_, so if a new thread of higher priority wakes up (or a realtime thread needs the CPU), the running thread will be preempted and a context switch to the higher-priority thread occurs. Also, a periodic clock interrupt (running at the end of a thread’s quantum) triggers the scheduler to preempt and switch threads if needed.
    

During a thread context switch, the OS saves the CPU state of the outgoing thread and loads the state of the incoming thread:

- The outgoing thread’s volatile CPU registers and program counter (instruction pointer) are saved (pushed onto its kernel stack or into its KTHREAD structure). The kernel stack pointer for that thread is saved in its KTHREAD so it knows where to resume.
    
- The incoming thread’s saved CPU context is loaded. The kernel will set the CPU’s stack pointer to the new thread’s **kernel stack** and restore the saved registers and instruction pointer, causing the new thread to resume execution where it left off.
    
- If the incoming thread is at user-mode when it resumes, the CPU will switch to the thread’s user-mode stack automatically upon returning to user space (more on stack switching below).
    

All thread context switches occur in kernel mode – even if a thread is preempted while running user code, an interrupt/exception will transition into kernel mode to perform the switch. Windows uses an interrupt request level (IRQL) called **DISPATCH_LEVEL** to synchronize scheduling: the scheduler raises the CPU’s IRQL to DISPATCH_LEVEL (level 2) while choosing the next thread and switching contexts. This prevents normal thread execution on that CPU during the switch. On multiprocessor systems, additional locking is used to coordinate scheduling among CPUs (e.g. each CPU’s dispatcher lock, or finer-grained locks per thread).

**Thread vs Process Switch:** If the new thread belongs to the **same process** as the old thread, the context switch is a pure thread switch – the virtual address space remains the same, so it’s relatively fast. If the new thread belongs to a **different process**, then a process context switch occurs as well (described next). Either way, from the scheduler’s view it’s picking a thread to run; the difference is whether the memory context (CR3) needs to change.

### Process Context Switch

A **process context switch** refers to the additional steps needed when the scheduler switches to a thread that belongs to a different process than the previously running thread. In Windows, each process has its own virtual memory space (address space), so changing processes means the memory mapping (page table) must be changed. This is effectively a thread switch **plus** a memory context switch:

- In addition to saving CPU registers, the kernel must switch the **page directory base** to the new process’s page table. On x86/x64 processors, this means loading the CR3 register with the physical address of the new process’s page directory (or PML4 on x64). In Windows, this value is stored in the process’s control block (KPROCESS/EPROCESS) as the **Directory Table Base**.
    
- Changing CR3 updates the CPU’s notion of the valid address space. This usually **flushes the Translation Lookaside Buffer (TLB)** (unless certain optimizations like PCID are used – discussed later), meaning cached virtual→physical address translations for the old process are discarded. The new process will incur TLB misses initially until its working set addresses are reloaded into the TLB. Thus, process switches are more expensive than switching between threads of the same process.
    
- The kernel also changes any CPU-specific registers that track the current process, for example the `%gs` base on x64 that points to the KPCR (though KPCR remains per-CPU). Windows provides a CPU instruction (`mov cr3, ...`) or uses an OS routine to perform this update during the context switch.
    

In Windows implementation, the context switch code will load the new thread’s **address space pointer** as part of the switch if the process is different. The Windows scheduler is aware of this cost – for example, it tries to avoid unnecessary migrations of threads across processes/cores to keep the cache and TLB “warm.” The scheduler’s policies like **ideal processor** (see below) also help reduce frequent cross-processor/process switches if possible.

To summarize, a process context switch includes everything a thread switch does _plus_ a memory map switch. The data structure that makes this possible is the process’s page table base stored in the EPROCESS/KPROCESS, and it’s loaded into CR3 to activate that process’s memory context. Modern CPUs and Windows have features to mitigate the overhead (e.g. PCIDs to retain TLB entries across switches, described later). But if you see a high rate of context switches between threads of different processes, it can indicate performance overhead due to frequent address space changes (measured by “Context Switches/sec” in performance monitors).

### Interrupt Context Switch

An **interrupt context switch** occurs when the CPU is interrupted to handle an **interrupt or exception**, causing a temporary switch in context from a thread to an interrupt handler. This is different from a thread/process switch because the CPU is not switching to a different thread; instead, it’s executing an interrupt service routine (ISR) on behalf of the hardware or system, often using a special context. However, from the running thread’s perspective, its execution is _interrupted_ and a new context (the interrupt context) runs, which will later return control back to a (possibly different) thread. Key points:

- When an interrupt (e.g. a clock tick, I/O device interrupt, or system call trap) occurs, the processor **automatically saves** certain registers on the current **kernel stack** (or a special interrupt stack) and transfers control to a predefined interrupt handler in the kernel. On x86/x64, the CPU pushes a _trap frame_ (containing at least EIP/RIP, CS, EFLAGS, and in some cases SS and ESP if coming from user mode) onto the stack. This trap frame and additional state saved by the OS represent the interrupted thread’s context.
    
- The CPU may also perform a **mode switch** from user mode to kernel mode if the interrupt arrived while the thread was in user mode. In that case, it automatically switches to the thread’s kernel stack (using the stack pointer stored in the Task State Segment for ring 0 on that CPU) before pushing the trap frame. Each thread has a dedicated kernel-mode stack for this reason.
    
- At this point, the CPU is said to be in an **interrupt context** (at an elevated IRQL in Windows). The interrupted thread is essentially paused. The interrupt handler runs in kernel mode, often at a high IRQL (e.g. device interrupts run at DIRQL, and the clock or scheduler interrupts at CLOCK_LEVEL/DPC level). Because normal thread scheduling is disabled above DISPATCH_LEVEL, the interrupt routine will perform its work and then potentially signal the scheduler if a thread switch should happen.
    

Windows distinguishes interrupt handling in two phases: the immediate interrupt service (often very short, at high IRQL) and possibly a deferred phase via a _Deferred Procedure Call (DPC)_ which runs at IRQL=DISPATCH_LEVEL. If an ISR wakes a higher-priority thread or a timer fires, Windows will schedule a DPC (or set a need for rescheduling) that runs shortly after the ISR. This DPC runs in a special context (technically still not a normal thread, but the scheduler uses it to switch threads safely).

It’s helpful to consider that an interrupt causes a **context save** (but not a full thread switch yet). After the interrupt/DPC routine finishes, the scheduler may perform a thread switch before returning to user mode. For example, the clock tick interrupt (timer interrupt) runs, finds that the running thread’s quantum expired, and triggers a thread switch to a new thread – this happens while still in kernel mode before lowering IRQL. In effect, the CPU went from thread A (user mode) → timer ISR (kernel, interrupt context) → scheduler code (kernel) → thread B (kernel or user mode).

**Stack usage for interrupts:** Windows uses dedicated **interrupt stacks** on x64 for certain scenarios. When an external interrupt occurs, Windows often switches to a special per-CPU **interrupt stack** to handle it. This prevents using the interrupted thread’s own kernel stack (which might be nearly full or paged out) for potentially significant interrupt processing. The kernel reserves one or more interrupt stacks per processor, and critical interrupts (like NMI, double-fault) are assigned to fixed **IST (Interrupt Stack Table) entries** in the CPU’s TSS. Windows x64 defines up to 7 IST entries per CPU for different purposes (e.g. NMI uses one). So, on an interrupt, the CPU may load a known-good interrupt stack (via the TSS/IST mechanism) before running the ISR, _effectively performing a context switch to a special stack context_. When the interrupt routine finishes, the CPU will restore the previous stack pointer and registers from the trap frame to resume the interrupted thread or to whichever thread the scheduler selects next.

In summary, interrupts cause a temporary context switch **into the kernel’s interrupt handling context**. The original thread’s context is preserved (in a trap frame on its stack or an interrupt stack) and will be restored when returning from the interrupt, unless a scheduling decision results in a different thread being resumed. Even in that case, the original thread’s context remains saved and it will be put back in the ready queue or wait state as appropriate. Thus, while we don’t call interrupt handling a “context switch” in the scheduling sense, it absolutely involves context saving/restoring and is a critical part of how Windows transitions between execution contexts (user->kernel, etc.). Often, an interrupt is the _precursor_ to a thread context switch (e.g. the clock interrupt leads to a thread switch).

**Example:** A disk I/O interrupt occurs on CPU0 while Thread X is running in user mode. The CPU switches to kernel mode and uses the interrupt stack to execute the disk ISR at IRQL = DIRQL. The ISR signals that an I/O request for Thread Y has completed, making Thread Y ready to run. The ISR schedules a DPC or sets an event and returns. The system then runs the DPC at IRQL = 2 (DISPATCH_LEVEL) on that CPU, which calls the scheduler. The scheduler sees that Thread Y has higher priority than the currently running Thread X (or that Thread X was waiting on I/O), so it performs a thread context switch: saving X’s context, switching memory context to Y’s process, etc., and resumes execution in Thread Y. This all happens before dropping back to IRQL 0. Finally, the CPU returns to either Thread Y’s user mode or continues in kernel mode in Thread Y if it was scheduled to run in kernel. Thread X’s context is now saved and won’t resume until it’s chosen again by the scheduler.

## Key Data Structures for Context Switching

Windows kernel uses several key **data structures** to manage threads, processes, and processors during context switching. Understanding these structures gives insight into what exactly is saved/restored or consulted during a switch:

- **ETHREAD / KTHREAD (Thread Objects):** An ETHREAD (Executive Thread) is the data structure representing a thread in Windows. It contains an embedded KTHREAD (Kernel Thread) structure (accessible via ETHREAD.Tcb) which is the kernel’s thread control block. KTHREAD holds scheduling-related fields and pointers needed for context switching.
    
- **EPROCESS / KPROCESS (Process Objects):** An EPROCESS (Executive Process) represents a process. It includes an embedded KPROCESS (accessible via EPROCESS.Pcb) which contains the kernel-level process info such as the page directory base, quantum settings, and list of threads.
    
- **KPRCB (Kernel Processor Control Block):** This is a per-CPU structure that the scheduler uses to track what’s happening on each processor – current thread, next thread, idle thread, and the ready queues for that processor, among other things.
    
- **Kernel Stacks and Trap Frames:** Each thread has a kernel stack (in kernel memory) used when it runs in kernel mode. The trap frame (saved register context) on the stack is used to restore context on interrupts or exceptions.
    
- **Dispatcher Objects and Lists:** Although not a single structure, the dispatcher refers to various lists and queues (ready list, wait list, deferred ready list) and the spinlocks that protect them.
    

Let’s examine these in more detail:

### KTHREAD and ETHREAD (Thread Control Blocks)

Every thread in Windows has an **ETHREAD** structure (defined in NTOS kernel) which contains bookkeeping info (like the thread ID, a pointer to owning process, security token, exit code, etc.) for executive subsystems. Inside ETHREAD is the **KTHREAD** (also called the Thread Control Block, TCB) which the kernel scheduler uses. The KTHREAD is crucial for context switching: it contains the thread’s CPU state when the thread is not running, and scheduling parameters. Key fields in KTHREAD/ETHREAD related to context switching include:

- **Stack Pointers:** The thread’s **KernelStack** (current kernel stack pointer) or pointers to the base and limit of the kernel stack. When a thread is switched out, the current stack pointer is saved in the KTHREAD, so that when the thread is next resumed, the kernel knows which stack and where to resume. On x86, for example, `KTHREAD.InitialStack` and `KTHREAD.StackLimit` describe the thread’s kernel stack region. The context switch code saves the _current_ stack pointer into KTHREAD (e.g. `RspBase` on x64 or similar).
    
- **Thread State and Wait Information:** Fields indicating whether the thread is Running, Ready, Waiting, etc., and what it might be waiting on. The scheduler updates these states. If a thread is waiting (blocked), it won’t be picked for context switch until the wait is satisfied.
    
- **Priority and Quantum:** The thread’s **priority** (both base and current dynamic priority) is stored in KTHREAD. This determines its position in ready queues. The **quantum** (time slice remaining) may be tracked per thread or per process. Windows might store a thread’s quantum in the KTHREAD (or it’s computed dynamically); each tick, the scheduler decrements a running thread’s quantum.
    
- **Affinity and Ideal Processor:** The KTHREAD contains the thread’s **affinity mask** (which processors it can run on) and an **IdealProcessor** number. The ideal processor is the preferred CPU for this thread, chosen to maximize CPU cache usage and load balancing. It’s set when the thread is created (distributed among CPUs) and can change – `KeSetIdealProcessorThread` can adjust it. The scheduler will try to run the thread on this CPU if available. KTHREAD also tracks the **Last processor** it ran on.
    
- **Context Switch Count:** There is a counter for how many context switches the thread has undergone (often exposed through tools). This might be in KTHREAD or ETHREAD. For example, Process Explorer can retrieve a thread’s context switch count (it shows “Context Switches” and “Context Switch Delta” for each thread, which come from the kernel’s accounting).
    
- **APC and Suspend info:** Not directly for context switching, but KTHREAD has pointers for APC (Asynchronous Procedure Calls) state, which can schedule user-mode code to run when a thread is about to return to user mode. If a **Kernel APC** is pending, the context switch code in Windows will trigger an interrupt at APC_LEVEL (IRQL 1) on the new thread to deliver it. The KTHREAD keeps track of deferred APCs. Similarly, a **suspend count** is tracked to know if the thread is suspended.
    
- **Links to Scheduling Queues:** KTHREAD has linkage fields for various lists:
    
    - **ReadyListEntry** (or similar) to link the thread into a dispatcher ready queue if it’s in the Ready state.
        
    - **WaitListEntry** to link into a wait list if the thread is waiting on an object or is in a **Deferred Ready** state.
        
    - **ThreadListEntry** to link into the list of all threads in its EPROCESS (so the process knows its threads).
        
    - **QueueListEntry** if the thread is part of a Kernel queue (KQUEUE) for worker threads.
        
- **Kernel Stack Swap**: Windows can **swap out** thread kernel stacks to conserve kernel memory when a thread waits for a long time (especially on x86 with limited kernel address space). A flag `SwapBusy` or a pointer to a backing store might be present to handle swapped-out stacks, but this is an advanced detail. If a thread’s kernel stack is swapped out, its KTHREAD indicates that and it won’t be scheduled until the stack is swapped back in (which is handled by the memory manager and swapper).
    

In practice, when a context switch happens, the outgoing thread’s KTHREAD is updated with its final stack pointer and possibly other register context, and its state is set to Ready or Wait as appropriate. The incoming thread’s KTHREAD provides the kernel with the new stack pointer and has its state set to Running. The actual general-purpose registers (like RAX, RBX, etc.) are not all individually stored in the KTHREAD structure; instead, they are saved on the thread’s **stack (trap frame)** or in an **architectural context block** (like the **KTHREAD.Context** area for extended processor state). For example, floating-point/SIMD state might be stored in an _extended context_ area or only saved on demand (lazy saving).

### EPROCESS and KPROCESS (Process Control Blocks)

The **EPROCESS** structure represents a process in Windows. It contains information needed to manage the process’s address space and threads. The relevant part for context switching is the **KPROCESS** (kernel process block), which is embedded in EPROCESS (EPROCESS.Pcb). Key fields related to context switching:

- **Directory Table Base (DTB):** This is the physical address of the process’s page directory (for x86) or PML4 (for x64), essentially the value that goes into CR3 for this process. In EPROCESS, this is often stored in a field called `DirectoryTableBase` (and on x64 there might be two entries, one for user mode and one for kernel mode address map if kernel page table isolation is used). When the scheduler switches to a thread in a different process, it will load this value into CR3 to switch the address space.
    
- **List of Threads:** EPROCESS has a **ThreadListHead** which links to all ETHREADs belonging to the process. Not directly used during the low-level context switch, but it’s how the OS can iterate over threads (e.g. to terminate them or to enumerate for tools).
    
- **Process Affinity:** The process’s allowable processors (affinity mask) which is inherited by threads unless a thread has its own affinity. The scheduler uses this in conjunction with thread affinity.
    
- **Base Priority (Priority Class):** The process’s base priority class (e.g. Idle, Normal, High, Realtime). This, combined with a thread’s relative priority, determines the thread’s actual base priority. Changing a process’s priority class will affect the base priorities of its threads (which can influence context switching by altering scheduling order).
    
- **Quantum settings:** The process can influence thread quantums. Windows has a concept of **foreground boost**where the process with input focus (foreground window) might get longer quanta (in Windows 10/11, this is usually managed differently than older versions, but historically the “priority separation” setting in the registry could give longer quanta to foreground processes). The KPROCESS may hold quantum values or a pointer to scheduling class. For instance, there is a field for quantum index or a flag if quantum should be varied. (In Windows Internals it’s noted that on client editions the foreground process threads get a 3× quantum boost by default).
    
- **Scheduling Flags:** EPROCESS/KPROCESS might have flags like “Process in swap-out state” or “Disable quantum boost”. If a process is swapped out (not in memory), its threads might be in a special wait state and their kernel stacks possibly swapped out too. The scheduler won’t schedule those until the memory manager brings the process back in (this is more relevant to old Windows with heavy memory pressure).
    
- **Memory Management Fields:** Not directly for context switch, but EPROCESS includes the working set information etc. For example, if the process is out of memory, the scheduler might not run its threads until memory is freed (this is indirect – the memory manager can block threads).
    

During a context switch, the EPROCESS of the incoming thread becomes the “current process” on that CPU. The kernel variable `PsGetCurrentProcess()` (or the KPCR’s current EPROCESS pointer) is updated to this new process. The CR3 is loaded from KPROCESS.PageDirectoryBase (DTB). If any CPU-specific structures (like segmentation on x86, or user mode GS base on x64 which often points to the thread’s TEB) need updating, that happens too.

_Example:_ If Thread A (in Process P1) is running and the scheduler picks Thread B (in Process P2) on that CPU, the context switch will save A’s context, then: `CR3 <= P2.DirectoryTableBase` (switch address space), `CurrentProcess <= P2`, then load B’s registers. This effectively installs P2’s page tables so that when B runs, it sees its own memory. The old process P1’s state remains in its EPROCESS; if P1 had no other threads running on any CPU, its address space is now inactive until a thread of P1 runs again (at which time CR3 will be reloaded with P1’s page directory and TLB will be repopulated).

### KPRCB (Per-Processor Control Block)

Each logical processor in a Windows system has a **KPRCB** structure (Kernel Processor Control Block) which holds the CPU’s scheduling state and other per-CPU data. This is one of the most important structures for context switching on multiprocessor systems. The **KPCR** (Kernel Processor Control Region) contains a pointer to its KPRCB (KPCR.Prcb). The PRCB includes fields such as:

- **CurrentThread:** Pointer to the KTHREAD of the thread currently running on that processor. The macro `KeGetCurrentThread()` essentially reads this field (via the KPCR) to get the running thread. When a context switch happens, this field is updated to point to the new thread.
    
- **NextThread:** Pointer to a KTHREAD that has been selected to run next on that processor (if a context switch is in progress or pending). In some cases, the dispatcher uses this to store the chosen thread before the actual switch occurs. After the switch, NextThread might be null and CurrentThread will be the new thread. (This field is used when the dispatcher is about to context switch; the assembly routine might rely on it or on parameters.)
    
- **IdleThread:** Pointer to the KTHREAD for the idle thread for that processor【29†】. If no real thread is runnable, the scheduler will switch to the idle thread (which has minimal workload, just halts or checks for work).
    
- **DispatcherReadyListHead[32]:** An array of 32 list heads, one for each priority level 0–31, for threads that are in the Ready state on this processor. Each entry is a linked list of KTHREADs waiting to be scheduled at that priority. The PRCB also has a **ReadySummary** bitmask (32-bit) where each bit corresponds to a priority level and is set if there is at least one thread ready at that priority. This allows the scheduler to quickly find the highest priority with any ready threads by scanning or using bit operations on ReadySummary.
    
- **DeferredReadyListHead:** A singly-linked list head for threads that have been made ready but are in the **Deferred Ready** state. Deferred ready is a mechanism to avoid holding locks for too long: when a thread is made ready from an interrupt or another processor, instead of immediately inserting it into the ready queue (which requires acquiring the dispatcher lock), Windows may add it to a per-CPU deferred ready list. Before the scheduler next runs or the processor’s IRQL drops, it will process this list and transfer those threads to the normal ready queues. This improves concurrency by deferring the actual insertion to a safer time.
    
- **Interrupt and DPC Management:** Fields like **DpcQueue**, **DpcCount**, **DpcStack**. The PRCB tracks pending DPCs for the CPU. The **DpcStack** is a pointer to a special DPC stack (if configured) used when executing Deferred Procedure Calls. As mentioned, if a normal thread is running with a near-full kernel stack and an interrupt schedules a DPC, Windows can switch to a dedicated DPC stack to execute the DPC, to avoid using up the thread’s stack. The PRCB.DpcStack points to the base of this region【28†】. On x64, the larger kernel address space usually made a separate DPC stack optional, but Windows still maintains this for reliability (especially on x86). The PRCB also has counters for DPCs and an array for DPC queues (High/Medium/Low importance).
    
- **CPU Scheduling Stats:** The PRCB keeps track of context switch count, idle time, interrupt counts, etc. For example, there’s a field for the number of **ContextSwitches** that have occurred on that CPU (this is system-wide, incremented each switch). Tools like Performance Monitor’s “Context Switches/sec” counter use these stats. There are also counters for idle schedule time and perhaps cache affinity metrics.
    
- **Locks and Synchronization:** In older Windows, there was a **DispatcherLock** in the PRCB (a spinlock) to protect the ready queues on that CPU. In newer Windows (post-Windows XP), finer-grained per-thread locking is used for context switch, and the dispatcher lock is held only for brief global state changes. The PRCB might still have a lock for certain scheduler operations (like inserting deferred ready threads).
    
- **Group and NUMA information:** On systems with processor groups (Windows uses groups to extend beyond 64 CPUs) or NUMA nodes, the PRCB contains data about the processor’s group and NUMA node, used for scheduling decisions to keep threads close to their memory or to handle group scheduling. For example, PRCB has an attribute for the group number and a pointer to a NUMA node structure. This influences the ideal processor selection as well.
    
- **PrcbFlags:** Bit flags indicating various states (e.g. whether the CPU is in idle, whether it’s in a interrupt, whether thread quantum end is pending, etc).
    
- **Routine callbacks:** The PRCB contains pointers or addresses of certain routines (like interrupt dispatch, or the address of the current kernel idle loop routine etc.), mostly for internal use or for specific functionality.
    

The KPRCB is accessed via the KPCR (on x64, `gs:[0]` points to KPCR, and `gs:[0x180]` or similar offset gives CurrentThread). In WinDbg, you can inspect the PRCB by using `!pcr` (which will show the PRCB address and key fields) or `dt nt!_KPRCB` if symbols are available. For example, `!pcr` will show something like current IRQL, CurrentThread, NextThread, IdleThread for that CPU, and addresses of IDT, GDT, TSS, etc【29†】. Also, the PRCB’s ready lists aren’t directly dumped by `!pcr`, but you can use `!ready` to see all ready threads (by priority) system-wide, which essentially goes through each PRCB’s ready queues.

During a context switch, the PRCB of the CPU performing the switch is updated. If a new thread is selected:

- `PRCB.NextThread` might be set to the new thread’s KTHREAD while the old thread is still CurrentThread. Once the low-level swap is performed, the PRCB.CurrentThread is set to the new thread and NextThread is cleared. The previous thread might get queued or marked waiting.
    
- If the thread is leaving the CPU, that CPU’s PRCB might also update some accounting (like last processor used for that thread – often stored in the thread’s KTHREAD as `PreviousMode` or a similar field, and the PRCB may log a “last thread” etc.).
    
- The PRCB’s ReadySummary bit for the new thread’s priority is decremented (if that was the last thread at that priority, the bit is cleared).
    
- If the outgoing thread still has quantum left and is just preempted (not waiting), it will be put back in the ready queue (likely at the head or tail depending on its priority and scheduling policy). The PRCB’s ready lists are updated accordingly.
    

All these structures work together: For instance, when a thread’s wait is satisfied, the kernel will ready the thread by inserting its KTHREAD into a PRCB’s ready list (possibly via the deferred ready mechanism). That KTHREAD’s state is changed to Ready. The target CPU may be chosen based on the thread’s ideal processor or affinity (it might not always be the current CPU). The PRCB of that target CPU will have its ReadySummary bit set for that thread’s priority. If that CPU was idle or running a lower-priority thread, a reschedule interrupt may be sent to it to prompt a context switch to the new thread.

### Additional Structures: Trap Frames and Context Records

While not explicitly asked, it’s worth noting how register state is stored. When a context switch happens, Windows often uses the current thread’s **kernel stack** to save state:

- On an interrupt or system call trap, a **trap frame** is pushed on the kernel stack (this is a struct that holds registers like rax, rcx, rdx, r8-r11 (volatile registers), return address, and non-volatile registers might be saved in a separate **exception frame** or by the callee). This represents the CPU state at the moment of interrupt. The trap frame will be used if we context switch away and come back later.
    
- When the dispatcher decides to switch threads, it calls an assembly routine (e.g. **KiSwapContext**) which saves the remaining processor state that wasn’t already saved. For example, on x64, non-volatile registers (rbx, rbp, rdi, rsi, r12–r15, xmm6–xmm15 if used, etc.) are saved either in the KTHREAD or on the stack. KiSwapContext will record enough information so that when the thread is resumed later, it can reconstruct all registers. It then switches the stack pointer to the new thread and restores that thread’s registers.
    
- In some cases, Windows uses a data structure called **KCONTEXT** or uses the user-mode **CONTEXT** structure format to save/restore thread contexts (especially when performing user-mode context swaps, like user-mode scheduling or when a thread is created/suspended). But during normal kernel scheduling, this isn’t done via high-level CONTEXT records, but via low-level stack operations as described.
    

## Stack Management (User vs Kernel Mode, Interrupt Stacks)

Each thread in Windows has **two stacks**: one for user mode and one for kernel mode. Proper handling of stacks is critical for context switching, especially when transitioning between user and kernel or servicing interrupts.

### User-Mode and Kernel-Mode Stacks per Thread

When a thread is created, the system allocates a **user stack** in the process’s user-mode address space (typically 1MB reserved by default, with a certain committed portion that can grow). This stack is used when the thread executes in user mode. The thread also gets a **kernel stack**, located in kernel memory (in a non-pageable portion of the system address space). Kernel stacks are smaller – usually on the order of 12 KB (plus a guard page) on x86, and 24 KB on x64, though these can vary. The kernel stack is used whenever the thread transitions to kernel mode (on system calls, page faults, interrupts, etc.). Only code running in kernel mode can access the kernel stack.

**Switching stacks on user→kernel transition:** When a user-mode thread makes a system call or is interrupted:

- The CPU (on x86) uses a Task State Segment (TSS) to load a new stack pointer (ESP0) from the TSS into ESP, switching to the kernel stack. On x64, a similar mechanism occurs for interrupts: the Interrupt Descriptor Table entry for a system call or interrupt indicates a stack switch if transitioning from user mode, usually by using the TSS as well (even though x64 has a fast SYSENTER/SYSCALL, it still references an MSR for the kernel stack or uses a single TSS stack for all ring3->0 transitions). Essentially, the CPU **automatically switches** to the thread’s kernel stack before pushing the trap frame. The kernel had set up the TSS with the pointer to the top of that thread’s kernel stack (KTHREAD’s InitialStack).
    
- After switching to the kernel stack, the CPU pushes the thread’s previous user EIP, CS, EFLAGS, ESP, SS onto this new stack (this is the trap frame base). The kernel then continues execution on the kernel stack. Thus, the thread is now using its kernel stack until it returns to user mode.
    

This means at any given time, a thread is either using its user stack (executing user code in ring 3) or its kernel stack (executing in ring 0). The two are never used simultaneously. The **context switch mechanism deals mostly with kernel stacks** because the switch occurs in kernel mode. If a thread is resumed in user mode, the CPU will switch back to its user stack automatically upon the return to user (IRET or SYSCALL return). The kernel doesn’t manually manipulate user stack pointer except when setting up a new thread’s initial context or unwinding exceptions.

**Kernel Stack contents:** When a context switch happens, the kernel stack of the outgoing thread will contain a **call stack of kernel routines** that were executing (if any) plus possibly a trap frame at the bottom (if it entered from user or interrupt). The context switch code (SwapContext) will save registers on this stack (essentially extending it a bit). The KTHREAD’s stored stack pointer will end up pointing to the location where it should resume (which is the base of those saved registers). The new thread’s kernel stack pointer (from its KTHREAD) is loaded, and the CPU’s ESP/RSP now points into the new thread’s kernel stack where that thread last left off. That thread’s kernel stack will have its saved context (trap frame or saved regs) that will now be popped/restored.

**Kernel Stack Swapping**: In low-memory scenarios, Windows can swap out a thread’s kernel stack to disk if the thread is waiting (this is called **Kernel Stack Swapping**). In such a case, the thread’s KTHREAD has a flag indicating its stack is swapped and a pointer to a backing store. The thread cannot run until the stack is swapped in. The scheduler will skip swapped-out threads until they are swapped in (done by the Balance Set Manager). This is transparent to context switching logic except the thread is marked as **Transition** state and not ready. When swapped in, the KTHREAD.KernelStack pointer is restored to a valid memory address.

### DPC Stack and Interrupt Stack (per Processor)

To handle asynchronous work at high IRQL, Windows on certain architectures uses separate stacks:

- **DPC Stack:** As noted, each processor can have a dedicated DPC stack. When a Deferred Procedure Call runs, if the current thread’s kernel stack is deemed too deep or not available, the OS can switch to the DPC stack. Typically, if a DPC is triggered while the CPU was in the **Idle thread**, the idle thread’s stack is mostly empty and can be used without switching. But if a DPC is triggered while a regular thread is running in kernel, Windows might switch to the DPC stack. According to Windows Internals, on x86, a DPC will run on a special DPC stack if the current thread’s stack usage is above a threshold or if a flag is set. The PRCB.DpcStack pointer is used for this switch【28†】. The switch to DPC stack is a mini context switch: the OS saves the current ESP, loads the DpcStack pointer into ESP, executes the DPC routine, and then switches back. This way, the DPC execution (which is essentially an “interrupt” like activity at IRQL 2) doesn’t risk overflowing the thread’s kernel stack or being blocked if that stack was paged out.
    
- **Interrupt Stack:** On x64, Windows reserves one or more **interrupt stacks** per CPU (often one is enough for most device interrupts, plus a couple for NMIs, double-faults, etc.). As the BSODTutorial blog notes, when an external interrupt occurs, Windows will **switch to the interrupt stack** for that CPU. This is accomplished via the Interrupt Stack Table in the TSS. Each CPU’s TSS has up to 7 IST entries pointing to dedicated stack memory. For example, IST1 might be used for standard device interrupts. In the IDT entry for, say, the clock interrupt or a device interrupt vector, Windows can specify an IST index. If an interrupt is delivered, the CPU will load the corresponding IST stack pointer, regardless of the current mode, and push the context there. This means even if a thread is in kernel mode on its kernel stack, a device interrupt can force a switch to the interrupt stack (ensuring a fresh stack). This helps avoid corrupting the thread’s stack or dealing with unknown stack state (particularly for NMIs or machine check exceptions, you want a reliable stack). The interrupted thread’s context (registers) is still saved on the interrupt stack as a trap frame. After servicing the interrupt, the kernel will use the trap frame to either resume the same thread or perform a thread switch.
    

In Windows 11 x64, the usage is likely: one IST for NMI, one for double fault, maybe one for machine check, etc. General device interrupts might or might not use IST – often they don’t on x64 unless configured, but the text suggests they might (“external happens... kernel switches to the interrupt stack”). It could be that they use an IST for all hardware interrupts to simplify stack handling at high IRQL.

**Stack layout example:** Suppose a thread is running in user mode and a keyboard interrupt occurs:

1. CPU looks at IDT entry for keyboard IRQ -> sees it should go to ring0, possibly using IST (or just the thread’s kernel stack if coming from user). Assuming typical case: it uses the thread’s kernel stack because coming from user, it will load from TSS the RSP0 (thread’s kernel stack top).
    
2. CPU switches to thread’s kernel stack, pushes SS, RSP (user), pushes CS, RIP, EFLAGS (forming initial trap frame).
    
3. If IST is configured for this interrupt, it might instead load the specified IST stack pointer (from TSS IST table) and push registers there. Let’s say it does use an IST; then the trap frame is on the interrupt stack.
    
4. The ISR (Interrupt Service Routine) in the HAL/driver runs. If it needs to schedule a DPC, it marks it and exits.
    
5. The CPU executes an IRET or similar. However, before that, at DPC level the system may run queued DPCs. The system will switch to DPC stack if needed (on x86 it would, on x64 maybe not always). Let’s assume it uses DPC stack – it saves current RSP (which might be the interrupt stack or the thread kernel stack) and loads PRCB.DpcStack. Executes DPC routine(s).
    
6. One of those DPCs is the scheduler’s DPC (sometimes called _dispatch interrupt_ or _quantum end_ DPC) which decides to switch thread.
    
7. The scheduler chooses a new thread, does the context switch routine: saves remaining context of Thread A (either on A’s kernel stack or the current stack), updates PRCB.CurrentThread, loads Thread B’s kernel stack pointer.
    
8. Now the system will eventually lower IRQL back to 0 and return from the interrupt. At this point, instead of returning to Thread A, it will end up “returning” into Thread B’s context (because the context was switched). Thread B might either continue in kernel or if it was supposed to be in user, the last part of context switch would set it up so that the return-from-interrupt goes to a user-mode address of Thread B.
    
9. Thread A is now off CPU, its kernel stack still contains the trap frame and saved state from when the interrupt happened. That will be used later when Thread A is scheduled again (especially if it needs to return to user mode, the trap frame needs to be processed – in fact, if A was preempted in user mode, it might not even “know” it was interrupted; the next time it runs, it could be on a different CPU continuing after the system call, since the kernel may complete the system call on its behalf).
    

The combination of using separate stacks for different execution contexts helps stability:

- **Idle thread stack:** Each CPU’s idle thread has a stack (usually small). When CPU is idle, any interrupt (like a new DPC or work) can run on the idle stack without issue.
    
- **DPC stack:** Ensures a DPC doesn’t run on a possibly deep call stack of a random thread.
    
- **Interrupt stack:** Ensures even if an interrupt comes in on top of deep kernel code, it has a fresh space.
    

From a data structure standpoint, the **KPCR (per-CPU region)** holds pointers to these special stacks or to the TSS. The PRCB contains the DpcStack pointer. The TSS (Task State Segment, one per CPU) is not a Windows-managed structure but a CPU structure that Windows sets up at boot. It holds the **RSP0** (base stack for ring0 on interrupts from user) and the IST array for x64. WinDbg’s `!pcr` output shows the TSS selector/base for each CPU【29†】. If needed, you can also dump the TSS to see the IST pointers (but that requires special steps since TSS is not directly exposed via WinDbg commands). However, knowing that these exist is usually enough at the software level.

## Scheduler and Dispatcher: Selecting the Next Thread

The Windows 11 kernel scheduler (often referred to as the **Dispatcher** in Windows Internals literature) is the component that decides **which thread runs next** on a given processor and initiates context switches. Key concepts in scheduling include thread priorities, ready queues, CPU affinity, and quantum (time slice). Here’s how the scheduler and dispatcher operate at a high level:

- **Priorities and Ready Queues:** Windows uses a **32-level priority system** (0–31). 0 is lowest (idle), 1–15 are dynamic levels for variable priority threads, 16–31 are real-time priorities. Each thread has a current priority. The scheduler maintains, for each processor, an array of ready queues (one per priority) in the PRCB. Threads that are ready to run (not waiting or running) are placed in the queue corresponding to their priority. There is also a **ReadySummary** bit mask indicating which priorities have any threads queued. This allows O(1) selection of the highest priority thread: the scheduler finds the highest set bit in ReadySummary (which corresponds to the highest priority with at least one ready thread).
    
- **Selecting next thread:** When a running thread can no longer continue (it blocks, its quantum expires, or it’s preempted), the dispatcher selects the next thread:
    
    1. It finds the highest priority ready thread (using ReadySummary and queues).
        
    2. That thread is removed from its ready queue (dequeued). It enters the **Standby** state – meaning it’s selected to run next on that CPU. (Standby is like “about to run”; only one thread per CPU can be in Standby at a time.)
        
    3. The scheduler will then perform the context switch to that thread, changing it to the Running state.
        
    4. The previously running thread will either go to Waiting state (if it was waiting on something or yielded), or to Ready state (if it was preempted and still ready to run again), or Terminated state if it finished.
        
    5. If the previous thread is still ready (say, preempted by a higher priority thread before its quantum ended), it is put back into the ready queue for its priority. In Windows, if a thread’s quantum hasn’t expired but it’s preempted, when it goes back to ready, by default it goes to the **front** of its queue (because it still has time left). Threads that voluntarily yielded or whose quantum expired go to the **back** of their priority queue.
        
- **Dispatcher Lock:** To manipulate these queues safely on multiprocessors, Windows traditionally used a **Dispatcher Lock** (a spinlock) that all CPUs must acquire to modify the ready queues or do certain state transitions. In modern Windows (Windows 7+), many scheduling operations are lock-free or use finer-grained locks. For instance, thread context switching is protected by a per-thread lock instead of a global lock, and each processor’s ready list is mostly independent (no global queue to lock, only per-PRCB data). However, some global state (like timers, or when a thread’s priority is boosted or it’s inserted in a wait list that might involve another processor) still requires a small global lock hold. The text from Windows Internals indicates the dispatcher lock still exists but is held only briefly for certain updates.
    
- **Interrupts and IRQL:** The scheduler runs in kernel at IRQL = DISPATCH_LEVEL or above. Typically, a context switch happens either:
    
    - Synchronously by a thread calling a wait or yield (in which case that thread calls into the scheduler, e.g. KiExitDispatcher, which raises IRQL to DISPATCH and selects next thread).
        
    - Or asynchronously via an interrupt (e.g. the clock interrupt or an I/O completion). In that case, the interrupt will trigger a software interrupt or set a need for dispatch. Windows uses a special interrupt, often the DPC interrupt (IRQL 2), to handle scheduling after the IRQ. The routine `KiDispatchInterrupt` is invoked at IRQL 2 to run DPCs and possibly schedule threads. This is basically the “dispatcher” that runs after certain interrupts.
        
    - In either case, while choosing and switching threads, the CPU is at IRQL 2 (no normal thread execution can preempt it). Once the new thread is loaded and everything is set, the IRQL is lowered to the new thread’s previous IRQL (usually 0 for user threads) and execution resumes in the new thread.
        
- **Quantum management:** Each thread (or rather, each priority level) has a time quantum – the time it can run before the scheduler may preempt it to give another thread of equal priority a turn. In Windows 11, quanta are not fixed clock ticks but are measured in CPU cycles or clock tick counts converted to a target cycle count. On a typical system, a quantum might be equivalent to 2 clock ticks (on client) or 12 on server by default. If a thread exhausts its quantum and there is another thread of the _same priority_ ready, the scheduler will switch to that other thread (round-robin within priority). If no equal priority thread is ready, the thread gets to continue running (potentially accumulating more time) – Windows will only enforce a quantum switch if there’s competition at that level. Real-time threads (priority 16–31) are by default given a very large quantum or effectively infinite unless preempted by higher priority, since they shouldn’t be time-sliced with others of same level often.
    
- **Priority Boosting:** Windows dynamically adjusts the priority of threads in some cases (e.g. boosting I/O-bound threads or lowering CPU-bound threads). These boosts can cause a context switch if a boosted thread’s priority now exceeds the running thread. For example, when a thread finishes waiting for I/O, Windows may boost its priority temporarily; that could make it the highest priority ready thread, so the next chance the scheduler gets, it will switch to it. Priority boosts are handled by adding the thread to the ready list at the new boosted priority and then requesting a reschedule if needed.
    
- **Ideal Processor and Load Balancing:** When unblocking a thread, the scheduler tries to schedule it on its **ideal processor** (the CPU recorded in KTHREAD.IdealProcessor). If that processor is idle or running a lower-priority thread, the thread might even preempt whatever is running there. If the ideal CPU is busy with something equal or higher, the thread can go into that CPU’s queue (or if affinity prevents that, it may go to another). Windows also has a **balance set manager** thread that periodically checks for imbalance – however, since Windows 7, the scheduler is more real-time in balancing across processors, so the balance set manager’s role in moving threads is minor. The scheduler will immediately try to find an idle processor for a newly readied thread (this is called **select next processor**).
    
    For instance, if a thread becomes ready on CPU0 but CPU1 is idle, Windows will attempt to schedule that thread on CPU1 (assuming affinity allows). If the ideal processor for that thread is CPU1 and it’s idle, that’s perfect – it will run there. If CPU1 is not idle, but CPU0 (current CPU) is free, it might just run it on CPU0 to avoid delay. There’s logic to decide where to run a thread: it considers ideal processor, last processor, current processor and checks for idle CPUs in the thread’s affinity mask. It also tries to respect NUMA locality: threads have an “ideal node” (NUMA node) and Windows will prefer to schedule them on a processor in that node if possible.
    
- **Idle thread:** If no thread is ready to run on a processor, the scheduler will schedule the per-CPU **Idle thread**. The idle thread’s job is basically to run a HAL routine to halt the processor or to run a minimal loop that checks for work (like delivering DPCs). The idle thread has priority 0 and is never in ready queues; it’s chosen when nothing else can be. Context switching to the idle thread is like any other thread (just very simple state). The idle thread will run until a new real thread is scheduled, at which point a context switch from idle to that thread occurs.
    

**Dispatcher in action (example call flow):**

- If a thread calls `NtDelayExecution` (Sleep), internally the kernel will put the thread into a wait state and call **KiSwapThread** to perform the dispatch. KiSwapThread (or KiExitDispatcher) will raise IRQL to 2, lock the dispatcher if needed, choose next thread, etc., then call **KiSwapContext** to actually swap the register context. KiSwapContext in turn calls the assembly routine **SwapContext**. This routine saves the old thread’s non-preserved registers, switches stacks, and restores the new thread’s registers. When it’s done, the new thread is running and it “returns” from KiSwapContext as that new thread. The old thread remains waiting.
    
- If a hardware timer interrupt fires, the sequence is: HAL timer ISR -> calls KeUpdateSystemTime & KeUpdateRunTime (accounting) -> notices quantum expiration -> queues a DPC or sets a reschedule flag -> exit ISR. Then the DPC routine for scheduler runs (KiTimerExpiration or similar) which will do similar steps of picking next thread and swapping context. This might happen via **KiRetireDpcList** which calls KiQuantumEnd or KiDispatchInterrupt internally.
    
- **KiDispatchInterrupt** is an interesting internal routine: it’s called when a software interrupt at DISPATCH_LEVEL is raised (essentially to run DPCs). It processes DPC queues and also checks for any threads in deferred ready state to promote to ready. After processing, if a reschedule is needed, it will perform the thread selection and call SwapContext directly. The blog snippet indicates KiDispatchInterrupt calls SwapContext directly instead of KiSwapContext for efficiency.
    
- **KiIdleLoop** (the idle loop for a CPU) repeatedly checks for any DPCs or scheduling needs. Part of its loop will check the deferred ready list and dispatch if something is there. The idle loop runs at IRQL 0 but with special handling to not lower IRQL below 0 (since idle thread often runs with scheduling enabled but no work).
    

### Dispatcher Locks and Synchronization

On multiprocessors, two CPUs could theoretically pick the same thread or interfere by both trying to modify a global data structure. Windows avoids that primarily by partitioning data per-CPU. Each CPU handles its own scheduling for the most part. However, certain operations require global coordination:

- When a thread’s priority is boosted or it is made ready from an APC or interrupt on another processor, the kernel might need to lock that thread’s lock or the target CPU’s PRCB to safely insert it. Windows uses inter-processor interrupts (IPIs) to “reschedule” another CPU if needed. For example, if a thread on CPU1 wakes a high-priority thread that is currently queued to run on CPU2 (its ideal CPU), Windows may send an IPI to CPU2 to prompt it to reschedule immediately.
    
- The old **Context Swap lock** (global spinlock in very old Windows) is replaced by a per-thread lock (stored in KTHREAD) for protecting the integrity of that thread’s context during a swap. This means two CPUs won’t try to swap the same thread simultaneously (not really possible logically, but this covers threads migrating).
    
- The **Deferred Ready list** is per-CPU and thus no global lock needed to add to it from an ISR on that same CPU. But if an interrupt on CPU0 makes a thread ready that should run on CPU1, it could either enqueue it to CPU1’s deferred list (which might require atomic ops or an IPI to CPU1 to handle it).
    
- The **Dispatcher database** is a term for all the scheduling queues. Windows still uses a system-wide dispatcher lock at times when manipulating things that could affect any CPU’s scheduling (like certain wait operations that involve multiple threads). But as noted, it’s held briefly. One example might be when a thread is removed from a wait list and goes to ready: if that wait list was global or at least accessible by multiple procs, you lock, remove thread, then at that point you can insert it into a specific CPU’s queue with only that CPU’s lock.
    

### Special Case: Hyper-threading and NUMA

Windows scheduler is topology-aware:

- On hyper-threaded CPUs (multiple logical processors on one physical core sharing resources), the scheduler tries not to schedule threads on two logical siblings if one core has both siblings idle while another core is fully busy. In other words, it favors spreading threads to separate physical cores before using the second logical on a core, to maximize performance.
    
- On NUMA, threads have an ideal NUMA node. The scheduler will first consider processors in that node for running the thread. Only if those are all busy (with equal/higher work) will it consider other nodes. This improves memory locality since threads allocate memory from their local node typically. The KTHREAD has a “IdealNode” and “IdealProcessor” which are used together.
    

## TLB and CR3 Switching (Memory Context Management)

When performing a context switch, handling the memory translation context is one of the most expensive parts due to TLB (Translation Lookaside Buffer) effects. In Windows on x86/x64:

- Each process has a **page directory** (on x86) or **PML4 table** (on x64) that serves as the root of its address translation. The physical base address of this structure is what gets loaded into the **CR3 register** on the CPU to define the current address space.
    
- When the scheduler switches from one process to another, it must load the new process’s CR3. In Windows code, this is typically done in the low-level context switch (part of KiSwapContext or SwapContext). As the Windows Internals book states: _“If the new thread is in a different process, [the dispatcher] loads the address of its page table directory into a special processor register so that its address space is available.”_. That “special register” is CR3 on x86/x64 (on Itanium it was different, but we focus on x64).
    
- Loading CR3 has the side effect of invalidating most of the TLB (since TLB entries are tagged by the address space in x86 context). This means after a process switch, memory accesses will incur page table walks until the TLB is repopulated for the new process’s working set. This is why switching processes can be expensive if done too frequently.
    

**TLB optimizations – PCIDs:** Modern x64 processors (post-Westmere) have a feature called **Process Context Identifiers (PCID)**, which allows the TLB to retain entries for multiple address spaces by tagging them. Windows 10/11 does take advantage of PCIDs when available, especially since the introduction of kernel page-table isolation (KPTI for Meltdown mitigation) which requires switching between user and kernel page tables frequently. Each process can be assigned a PCID (an ID up to 12 bits) such that when CR3 is loaded, if you use the `MOV CR3` with the PCID bits set, the CPU knows it’s a switch but to keep TLB entries for that PCID. Windows uses PCIDs to reduce TLB flush overhead. Typically, PCID 0 might be used for kernel, and user processes get PCIDs 1,2,3,... etc. When switching from one process to another, instead of flushing entire TLB, Windows may use INVPCID instructions to selectively invalidate only the old process’s entries or entries that shouldn’t remain. This retains other processes’ entries in the TLB so that if the CPU goes back to them soon, some translations are still cached.

For example, Windows might assign a PCID equal to the process’s index (just conceptual). When switching:

- Load CR3 with new base + new PCID (invoking minimal flush).
    
- Use INVPCID to invalidate any TLB entries for the new PCID that might be stale (ensuring no leftover translations from a previous use of that PCID, if any).
    
- This way, the TLB entries for the old process (with its PCID) can remain in the TLB (though they won’t be used until that process is scheduled again and CR3 with that PCID is loaded). If the process switches back soon, some memory references may find their translation still in TLB tagged with that PCID.
    

If PCIDs are not available or not used, Windows falls back to the old behavior: loading CR3 flushes the entire TLB. Note that Windows flushes even on thread switch within the same process in some cases if certain events occurred (like if pageable memory was freed or if an explicit flush was needed for other reasons). But generally, if process doesn’t change, it tries not to flush TLB. When threads in the same process context-switch, CR3 is not reloaded (since it’s the same for that process), so TLB remains intact – this is a big performance win for switching between threads of same process (e.g. many server apps use thread pools in one process to benefit from this).

**ASIDs on other architectures:** On ARM64 or Itanium, similar concepts exist (ASIDs, etc.), and Windows uses them to manage TLB flush behavior. The concept is the same: minimize unnecessary TLB invalidations.

**CR3 and hypervisor/VM**: If Windows is running under a hypervisor, the hypervisor might have to intercept CR3 writes or flushes (depending on virtualization extensions). But that’s beyond our scope here. Microsoft Hyper-V does expose PCID to VMs if available, so that Windows 11 in a VM can still use PCIDs.

**Context Switch and Memory Barriers:** When switching processes, Windows must ensure that any memory operations in the old process are completed (store buffers flushed, etc.) before switching to a new address space, to avoid stale memory accesses. Typically, the act of loading CR3 is a serializing operation on x86, which flushes those buffers. Also, going to a higher IRQL (dispatch) on each CPU acts as a barrier because devices/DMAs are handled accordingly.

To see CR3 values in a debugger, one can use the `!process 0 ff` command in WinDbg on a process which will show the DirectoryTableBase. Also, the CPU’s CR3 at runtime can be retrieved with `r cr3` in a live kernel debug (or checking the KPCR for the current process’s Page Directory).

**TLB Shootdowns:** Another aspect: when a process frees or modifies a page mapping, and that process might be running on other CPUs, Windows performs **TLB shootdowns** – cross-processor IPIs to invalidate that mapping on other cores. That’s not exactly context switch, but related. Each context switch on a CPU deals with that CPU’s TLB. Windows keeps track of which processors have a given process’s address space loaded (the process’s bitmask of active processors). If a page is unmapped in a process, the OS sends IPIs to all CPUs that currently have that process loaded to invalidate that page’s TLB entry.

In summary, **Windows’ CR3 switching logic** on context switch is: if new thread’s process != current process, then:

1. Load new CR3 (with PCID if available) to switch address space.
    
2. (Optionally) flush TLB or use PCID optimizations to retain entries.
    
3. Update any CPU registers that track user space (like FS/GS base for user, though usually those are in the thread’s context and restored as part of user mode register state).
    
4. Continue with switch.
    

## Optimizations in Context Switching

Windows has several optimizations to make context switching and scheduling efficient:

- **Lazy FPU/SIMD Saving:** Historically, Windows used _lazy context switching_ for floating-point (FP) and SIMD registers. The idea is not to save and restore the FPU state (MMX/XMM/YMM registers) on every context switch, because not all threads use the FPU frequently. Instead, the OS can defer loading the new FPU state until the thread actually executes an FP instruction, at which point a device not available exception (DNA exception) occurs and the kernel then saves the previous FPU state and loads the new one. This is what is meant by lazy saving – only do the expensive FXSAVE/FXRSTOR when needed. In Windows, a flag in KTHREAD indicates whether the FPU state is “owned” by the thread or not. If not, the first use triggers a save/restore. This optimization can significantly reduce overhead in context switches for code that doesn’t use floating point (like many OS threads or integer-heavy tasks).
    
    However, due to the **Lazy FPU vulnerability (CVE-2018-3665)** on Intel CPUs, Windows (like other OSes) moved to an **eager FPU context switch** by default on affected CPUs or generally, meaning it now saves/restores the FP state on every switch to prevent leaking registers across threads. On modern Windows 11, by default, the lazy approach might be disabled for security. Nonetheless, conceptually it’s an optimization and could still be used on systems not vulnerable or with certain settings. The kernel has code to handle either mode. There’s also an optimization where if two consecutive threads belong to the same process and both use the FPU heavily, it might not need to reload the entire FPU state if it was left as is for that process (though typically it would still load thread-specific values).
    
- **Deferred Ready List:** As described, Windows defers placing threads into the ready queue immediately in certain cases to minimize locking. The **Deferred Ready** state is an intermediate state for threads that have been selected to run on a processor but haven’t been fully readied yet. For example, when a thread’s wait is satisfied at high IRQL (like inside an ISR), the kernel can mark the thread deferred-ready and target a specific CPU for it. Then, when the system is about to lower IRQL or exit the interrupt, it will move that thread to the actual ready list of that CPU. This means the ISR didn’t have to acquire the dispatcher lock, it just added the thread to a per-CPU singly-linked deferred list (which is a quick operation). Later, at a safer time, that list is drained (e.g. in the DPC or when leaving an interrupt). This reduces contention and IRQL hold times, improving scalability on multiprocessors.
    
- **Ideal Processor and Soft Affinity:** The **IdealProcessor** for each thread (stored in KTHREAD) helps keep threads on the same CPU to take advantage of cache warmth. Windows uses a round-robin assignment for ideal processors for new threads to distribute load. It also adjusts ideal processor on the fly: if a thread runs a long time on CPU 3, that might remain its ideal. IdealProcessor is a hint, not a hard binding (unless affinity is set). The scheduler gives preference to scheduling a thread on its ideal CPU, then the last CPU it ran on (if ideal is not available). This optimization aims to maximize cache hits and minimize migration cost.
    
    Additionally, Windows tries to **minimize cross-processor migrations** of threads unless necessary (especially for threads with lower priority – it won’t move them unless idle CPU is starving or priority conditions force it). This reduces the need to reload caches and TLBs on another CPU.
    
- **Thread selection and preemption optimizations:** When a thread becomes ready at a priority higher than the current running thread on a CPU, the kernel will immediately preempt the running one (via IPI if it’s on a different CPU, or just next instruction if on same CPU after IRQL drop). However, if the priority is the same or lower, it will not immediately preempt (it will wait until the running thread’s quantum ends or it blocks). This avoids unnecessary context switches. There’s also a mechanism called **Quantum Stretch** on client systems: if a thread is in the foreground process (with focus), Windows gives it up to 3x longer quantum (if other threads of same priority are ready). This reduces context switches by letting the foreground thread run more before switching out, improving interactive performance.
    
- **Lock splitting:** As mentioned, replacing global locks with per-CPU or per-thread locks (and lock-free operations) is an optimization that was introduced to avoid bottlenecks during context switching. So multiple context switches on different CPUs can happen concurrently without all waiting on one global spinlock.
    
- **Idle optimization:** The idle loop itself is optimized to avoid overhead. When idle, the CPU can enter a low-power state (halt). But the idle thread will also check for pending work (like DPCs) at dispatch level without doing a full context switch. For example, delivering DPCs or rescheduling can happen while in idle without switching context because idle is essentially a “do nothing” thread that can be replaced instantly.
    
- **Large Page Tables and TLB:** Another indirect optimization – Windows uses large pages for kernel (and possibly for some process memory if configured). Large pages (2MB or 1GB pages) mean fewer TLB entries are needed. When switching address spaces, if many entries were 4KB, you flush more entries than if using some large pages (since large page covers more address space per TLB entry). Windows maps the entire kernel in a few large pages (for code/data) which helps that part remain in TLB after a flush (since kernel space is typically global on x64 – actually on x64 without KPTI, kernel memory is shared across processes so not flushed on context switch. With KPTI, they split user/kernel page tables and PCIDs mitigate the impact).
    
- **NUMA aware thread placement:** Ensures minimal NUMA node migrations for threads, which effectively is an optimization to reduce costly remote memory access, akin to reducing context switch “distance” in the system.
    

To illustrate some optimizations, consider a scenario: 10 threads all at same priority run on a single CPU. Without optimizations, you might get a context switch every tick in round-robin. Windows might detect if only one thread is active at that priority and just let it continue (quantum stretching on server OS = fixed long quantum). Or if threads are I/O bound, their priorities will boost when I/O completes and drop when done, so CPU-bound threads might get lowered priority – meaning less frequent preemption by those I/O threads (since they’ll finish quickly). Windows dynamically balances these to avoid rapid oscillations (e.g., it has a concept of ideal interval for context switching).

## Monitoring and Examples (WinDbg and Tools)

Understanding context switching at this level can be reinforced by using tools and debuggers:

- **Performance Monitor (perfmon.exe):** Under “System” counters, you have “Context Switches/sec” which shows the rate of context switches across the whole system. Also “Thread(_Total)\Context Switches/sec” can show similar broken down by thread, and you can monitor specific processes by summing their threads. A consistently high rate (e.g. many tens of thousands per second on a single CPU) might indicate too many threads contending or lots of interrupt activity.
    
- **Process Explorer:** In Process Explorer (from Sysinternals), if you enable View → “Select Columns” → “Thread” tab, you can add columns for **Context Switches** and **Context Switch Delta** for threads. The **Context Switches**column shows the total number of context switches each thread has done since it started, and **Context Switch Delta**shows how many switches occurred in the last update interval (e.g. last 1 second, if refresh rate is 1 second). By sorting threads by context switch delta, you can see which threads are getting switched the most. Often, the System process’s “System Idle Process” threads have high switch counts (that’s just the idle thread being scheduled and unscheduled frequently). You may also see high context switches for interrupt-intensive threads or if some threads are yielding a lot.
    
    Process Explorer also displays each thread’s current **State** (Running, Wait:xxx, Ready, etc.) and CPU time. A thread that is Ready but not running means it’s waiting its turn on a CPU – if you see many ready threads of high priority, the scheduler may be very busy switching among them.
    
- **WinDbg (Kernel Debugging):** If you break into a live kernel debug or analyze a crash dump, you can use these commands:
    
    - `!thread` – This command, when given a thread address or the current thread, will dump the ETHREAD and KTHREAD information. It includes fields like priority, state, **Context Switch Count**, Wait reason, kernel stack pointer, etc. For example:
        
        ```
        kd> !thread 8652b580  
        THREAD 8652b580  Cid 1234.5678  Teb: 7ffdf000 Win32Thread: 00000000 RUNNING on processor 0  
            Priority 10  PriorityDecrement 0  
            Context Switches: 15764  (some large number)  
            UserTime 00:00:00.0156  KernelTime 00:00:00.0312  
            Wait Start TickCount 1234567  
        ...
        ```
        
        This shows the thread is RUNNING on CPU0 at priority 10 and has done 15,764 context switches so far. (This is an example, actual output varies by Windows version and symbols). If the thread was waiting, it would show a wait reason.
        
    - `!process 0 1` – Dumps a list of processes with their threads. It will show for each thread: its TID, priority, state, and sometimes the wait reason or last error. This can help identify how many threads are ready vs waiting.
        
    - `!ready` – Lists all threads that are in Ready state, sorted by priority. This essentially reflects the ready queues. It will show which threads are ready to run, their priority and which processor they were last on or last ran.
        
    - `!pcr` or `!prcb` – On a particular processor context, `!pcr` will show the KPCR and PRCB data: CurrentThread, NextThread, IdleThread addresses, current IRQL, and some scheduling flags【29†】. For example `!pcr 0` shows for CPU0. `!prcb <n>` (if extension available) might dump the PRCB structure in detail including the ready lists (not sure if that’s built-in; sometimes you have to manually dt).
        
    - `!running` (from Debugger extension MEX) – This can show all running threads on each CPU, but the default `!pcr` is usually enough: it tells you which thread is on which CPU (CurrentThread for each).
        
    - **Call stacks:** In a kernel debug, you might inspect the stack of a thread doing a context switch. For instance, if a thread is in the middle of switching, a stack trace might show functions like `KiSwapContext -> KiSwapThread -> KiExitDispatcher`. A blocked thread might show `KeWaitForSingleObject -> ... -> KiSwapThread -> KiSwapContext -> nt!SwapContext`. The new thread would have a stack that picks up from after SwapContext (since SwapContext doesn’t return in the old thread, it returns in the new thread context).
        
- **Xperf/WPA (Windows Performance Analyzer):** For a deep dive, one can use ETW (Event Tracing for Windows) with the “Context Switch” tracing. Windows has an event for thread context switches (recording old thread, new thread, reason, CPU, etc.). By capturing a trace with the “CPU Scheduling” provider, you can get a timeline of context switches and visualize which threads ran when, and for how long. This is very useful to analyze thread behavior, latency, etc., but requires more setup.
    
- **Process Hacker:** Similar to Process Explorer, it can show per-thread context switch counts. It also directly shows the number of context switches a process has incurred (sum of its threads) and can highlight intense switching.
    

**Pseudo-code example of context switch:**  
To tie it together, here is a pseudo-code approximation of what happens in a simplified way when a thread voluntarily yields the CPU (like waiting on an event):

```c
// Pseudo-code of a simplified dispatcher scenario in kernel (very abstract)
KeWaitForSingleObject(object):
    IRQL = KeRaiseIrql(DISPATCH_LEVEL);         // raise IRQL to block preemption
    object.WaitList.Add(CurrentThread);         // put current thread in object’s wait list
    CurrentThread->State = WAITING;
    // Find next thread to run on this processor:
    next = SchedulerFindNextReadyThread(CurrentProcessor);
    if (next == NULL) {
        next = CurrentProcessor->IdleThread;
    }
    // Set up for context switch
    CurrentProcessor->NextThread = next;
    // (CurrentThread still points to old thread here)
    // Remove 'next' from ready list, mark it as running
    if (next != CurrentProcessor->IdleThread) {
        RemoveFromReadyQueue(next);
        next->State = RUNNING;
    }
    // Perform the low-level context switch
    KiSwapContext(CurrentThread, next);
    // -- At this point, we are in the context of 'next' thread --
    CurrentProcessor->CurrentThread = next;
    CurrentProcessor->NextThread = NULL;
    KeLowerIrql(IRQL); // lower to previous (usually 0 for user threads)
    return STATUS_WAIT_0; // (in the case of a wait, the new thread will eventually return something)
```

And KiSwapContext (very abstractly):

```c
KiSwapContext(oldThread, newThread):
    // Save oldThread context (simulate pushing registers)
    oldThread->SavedState.Rip = GetInstructionPointer();
    oldThread->SavedState.Rsp = GetStackPointer();
    oldThread->SavedState.Rflags = GetFlags();
    // Save non-volatile registers
    oldThread->SavedState.Rbx = Rbx;
    oldThread->SavedState.Rbp = Rbp;
    ... (etc)
    // If lazy FPU, handle FPU save if needed
    if (FPUsedBy(oldThread)) SaveFPUState(oldThread);
    // Switch address space if needed
    if (oldThread->Process != newThread->Process) {
        LoadCR3(newThread->Process->DirectoryTableBase);
    }
    // Switch stack
    SetStackPointer(newThread->SavedState.Rsp);
    // Load new thread’s non-volatile regs
    Rbx = newThread->SavedState.Rbx;
    Rbp = newThread->SavedState.Rbp;
    ... 
    // Restore flags, instruction pointer (this might be via ret/iret in assembly)
    SetFlags(newThread->SavedState.Rflags);
    JumpTo(newThread->SavedState.Rip);  // actually this happens by a RET popping RIP from stack
```

In reality, SwapContext is implemented in assembly for performance, doing these steps with perhaps some differences in order and using the fact that some registers are already on stack (pushed by caller). Also, Windows keeps the old thread’s `KernelStack` pointer in KTHREAD and uses that for stack swap rather than directly reading CPU registers.

**Diagrams (ASCII art) illustrating a context switch and memory layout:**

Below is an ASCII diagram of a context switch between two threads on a single CPU, highlighting stacks and register save/restore:

```
CPU (Processor 0) executing Thread A (Process X):
------------------------------------------------
Registers: [ RIP=A_pc, RSP=Kstack_A_top, RAX,... ]   CR3=PageDir(X)
Kernel Stack (Thread A):
    | ...           |  <- top (current RSP)
    | Saved regs?   |      (running in kernel, some regs in use)
    |-------------- |
    | Trap frame    |  <- e.g. saved user mode RIP, etc. if came from user
    |-------------- |
    | ...           |  <- base (InitialStack)
User Stack (Thread A):
    |--------------|
    | (not in use now, thread is in kernel) 
    |--------------|

Ready Thread B (Process Y) is about to run (its context is saved):
------------------------------------------------
Registers: [ (not currently on CPU) ] 
Kernel Stack (Thread B):
    | ...           | <- top of stack (InitialStack, not actively used while thread not running)
    | Saved RIP = B_pc   | <- last saved instruction pointer
    | Saved RFlags       |
    | Saved RBX, RBP, etc (non-volatile regs) |
    | (maybe trap frame if B was preempted in kernel) |
    |--------------|
    | ...           | <- base of stack (StackLimit)
User Stack (Thread B):
    |--------------|
    | ... (if B was in user mode, trap frame above saved its user SP, etc.) 

Context Switch from A to B:
------------------------------------------------
1. Save A’s context:
   - Push A.RIP, RFlags onto A’s kernel stack.
   - Save A’s RSP (stack pointer) into A.KTHREAD.KernelStack.
   - Save non-volatile registers (RBX, RSI, etc.) into A’s stack or KTHREAD.
2. If switching to new process Y:
   - Load CR3 with Y’s page directory base (flushes TLB):contentReference[oaicite:115]{index=115}.
3. Switch stack pointer:
   - Set RSP = B.KTHREAD.KernelStack (saved pointer to B’s last stack position).
4. Restore B’s context:
   - Pop registers from B’s stack (or load from KTHREAD) – e.g. RBX, RSI, RFlags, then finally RIP.
   - CPU jumps to B.RIP. Thread B is now running.
------------------------------------------------

After switch:
Thread A KTHREAD.State = Ready/Wait, KernelStack = saved pointer
Thread B KTHREAD.State = Running, KernelStack = active (RSP)
CR3 = PageDir(Y) (address space of Process Y loaded, TLB reflects Y’s mappings)
```

And a diagram of stacks per CPU and how interrupt and DPC stacks relate:

```
Memory Layout per CPU (Kernel Mode):
---------------------------------------------
[ Per-CPU Region (KPCR) ] 
    contains: CurrentThread, PRCB, pointers to stacks, etc.
[ KPRCB (in KPCR or separate) ]
    contains: Ready queues, DpcStack pointer, etc.
[ Interrupt Stack(s) ]  (one or more, per CPU)
    CPU0_InterruptStack: dedicated for ISRs (size ~4KB or more)
[ DPC Stack ] (per CPU, maybe same as InterruptStack on x64 if unified, or separate on x86)
    CPU0_DPCStack: used for running DPCs at IRQL 2
[ Idle Thread Stack ] (one per CPU)
    CPU0_IdleStack: used when idle thread runs
---------------------------------------------
For each thread that can run on CPU0:
    [ Kernel Stack for Thread X ] (in system space, typically 12-24KB)
    [ Kernel Stack for Thread Y ]
    ...
 (These are not contiguous; each is allocated separately in nonpaged pool or kernel stack segment)

User space (per process):
    [ Thread X User Stack ] (in process X address space, typically 1MB)
    [ Thread Y User Stack ] (in process Y, etc.)
```

This shows that the CPU has special stacks and each thread has its own kernel stack. The interrupt stack is used only transiently during interrupts, the DPC stack during DPC execution. The current thread’s kernel stack pointer is switched out when those are in use, but the OS keeps track so it can resume properly.

Finally, note that **Windows 11’s scheduler** includes features like **Thread Director** on newer Intel CPUs (if exposed, OS can bias scheduling based on efficiency cores vs performance cores), but that’s beyond core context switching – it influences which CPU is chosen as ideal for a thread but the fundamental context switch mechanism remains as described.

**Summary:** Context switching in Windows 11 involves intricate cooperation between hardware and the kernel. The OS saves CPU state of threads in the KTHREAD (and stack), switches page tables if necessary (updating CR3 and flushing TLB with optimizations like PCID), and uses per-CPU scheduling queues to decide which thread to run next. Optimizations such as lazy FPU save, deferred ready processing, ideal processor selection, and careful lock management ensure that this heavy task is as efficient as possible. Tools like Process Explorer, performance counters, and WinDbg can reveal the behavior of the scheduler – for example, high context switch counts might be observed for certain threads or interrupts, indicating a lot of activity. With this deep understanding, one can appreciate the complexity that allows Windows to juggle thousands of threads across multiple processors seamlessly.

**Sources:** Windows Internals 7th Edition for thread scheduling and context switch details, Windows kernel debugging & research blogs for internal data structures, and Microsoft documentation for tools and APIs.