Below are additional, deeply detailed Obsidian notes that delve further into the Windows internals underlying memory-mapped files (sections) and shared memory. These notes assume you are already familiar with the basic usage of `CreateFileMapping`, `OpenFileMapping`, `MapViewOfFile`, and the general concept of section objects. The focus here is on understanding the internal mechanisms, data structures, and system calls that enable these capabilities, as well as considerations for advanced scenarios.

**Audience:**  
These notes are for veteran Windows engineers and researchers who want deep insights into how memory mapped files and sections operate at the OS internals level. Understanding these details may help with troubleshooting unusual issues, optimizing performance, or working with low-level system components.

---

## Kernel Objects and Section Manager Internals

A **section object** (also known as a file mapping) is managed by the Windows **Memory Manager** and the **Section Manager** component within the kernel. Key points:

1. **Section Objects:**  
   - Created via `NtCreateSection` (the native system call underlying `CreateFileMapping`).
   - A section object references either a file object (if file-backed) or the system’s paging file (if created with `INVALID_HANDLE_VALUE`).
   - Named sections are placed in the **Object Manager** namespace, typically under `\BaseNamedObjects`.  
     Internally, this means you can see these objects (if you had direct kernel-level visibility) in the object directory structure.

2. **Object Reference Counting:**  
   - Each `CreateFileMapping` or `OpenFileMapping` call increments reference counts on the section object.
   - The section remains alive as long as at least one handle or mapping reference exists.
   - When all references are closed, the memory is reclaimed, and if file-backed, pending dirty pages are flushed to the backing file.

3. **Backing Store (File-Backed vs. Pagefile-Backed):**  
   - **File-Backed:** Changes to mapped pages are eventually written to the file. The `Modified Page Writer` system thread handles background flushes.
   - **Pagefile-Backed (Anonymous):** The memory resides in the page file or simply in physical memory as demand paging units. No persistent storage outside of the system’s paging mechanism. Once all references close, the data disappears.

---

## Virtual Addressing and Page Table Entries

When you call `MapViewOfFile`, the kernel:

1. **Allocates Virtual Address Range:**  
   - `NtMapViewOfSection` (the native call underlying `MapViewOfFile`) uses the process’s virtual address space manager to find a contiguous range of free virtual addresses.
   - It updates the process’s **Virtual Address Descriptor (VAD)** structures to mark this range as referencing a portion of the section object.

2. **On-Demand Paging:**
   - Pages are initially not all committed in RAM. They use the **demand paging** mechanism.
   - When the process accesses a page for the first time, a page fault triggers the memory manager to bring the page from the backing store (file or pagefile) into physical memory.
   - Subsequent accesses to the page do not cause faults, as long as it remains in memory and isn’t evicted.

3. **Shared Pages Among Processes:**
   - If multiple processes map the same section (and the same offsets), they share the same physical pages if those pages are already loaded into memory.
   - The system uses reference counting for shared pages, ensuring changes by one process are instantly visible to others because they access the exact same page frames in RAM.
   - Page table entries in each process’s page tables point to the same physical frames. The difference is that each process’s page tables and VADs are updated so they see the mapped region at different virtual addresses if necessary.

---

## Synchronization of Modifications

1. **Immediate Visibility:**
   - Writes by one process to a page in the shared memory are immediately visible to all other processes that have that page mapped, because they are physically the same page of RAM.
   - There is no separate “commit” step. Memory writes propagate at the speed of a normal store instruction.

2. **Coherency and CPU Caches:**
   - On multicore systems, hardware cache coherence ensures that updates by one CPU core are visible to others.
   - No special flush is required unless you’re dealing with memory-mapped I/O or special non-coherent memory areas (rarely the case in normal user-mode code).

3. **User-Mode Synchronization Still Needed:**
   - Although data is physically shared and coherent, logically you must still use synchronization (mutexes, events, interlocked operations) if multiple processes read and write concurrently to prevent data races.
   - The kernel does not provide automatic locking for changes to shared memory. It only ensures that what is written is what you read.

---

## Named Objects and the Object Manager

1. **Name Resolution:**
   - When you specify a name in `CreateFileMapping`, the kernel creates a named section object in the Object Manager namespace.
   - `OpenFileMapping` calls `NtOpenSection` internally, which uses the Object Manager’s namespace lookups to find the existing named section.
   - Names typically reside under the `\BaseNamedObjects` directory unless specified otherwise, and `Local\` or `Global\` prefixes determine session isolation.

2. **Security Descriptors:**
   - Section objects have associated security descriptors.
   - If a process attempts `OpenFileMapping` without appropriate `SECURITY_DESCRIPTOR` permissions, it will fail with `ACCESS_DENIED`.
   - Use standard Win32 security functions or `SECURITY_ATTRIBUTES` when creating the mapping to control access.

---

## Large Mappings and Advanced Features

1. **Large Files and Huge Memory Mappings:**
   - Memory-mapped files can map extremely large files (multi-GB or even TB on 64-bit systems).
   - You can map partial views, only bringing portions of the file into memory as needed.
   - The system uses a tree of VADs to manage large address spaces efficiently.

2. **NUMA and Performance Considerations:**
   - On NUMA (Non-Uniform Memory Access) machines, the memory manager tries to allocate physical pages local to the process’s node to reduce memory latency.
   - For shared mappings across multiple NUMA nodes, the placement of pages and the order processes fault them in may affect performance.

3. **Write-Through and Page Protection:**
   - By default, modifications are lazy: they occur in memory and flush later if file-backed.
   - You can combine `FILE_FLAG_WRITE_THROUGH` or use flush operations (`FlushViewOfFile`, `FlushFileBuffers`) to ensure data hits stable storage quickly.
   - Protections like `PAGE_READONLY`, `PAGE_WRITECOPY`, or `PAGE_EXECUTE` influence what operations processes can perform on the shared memory. Write-copy maps create a private copy of pages on write, allowing controlled sharing.

---

## Debugging and Diagnostics

1. **Tools:**
   - **Process Explorer:** Shows section objects and their names under process handles.
   - **RAMMap (Sysinternals):** Can show which pages belong to which file mappings.
   - **Windbg/KD:** At kernel-level, you can inspect VADs, section objects, and page table entries.
   - **ETW (Event Tracing for Windows):** The Memory Manager providers can help trace page faults related to memory-mapped files.

2. **Common Issues:**
   - **ACCESS_DENIED:** Check security descriptors or that the name and permission match on `CreateFileMapping` and `OpenFileMapping`.
   - **Name Conflicts:** If a section name is already in use but with different parameters, you’ll get unexpected errors. Use unique naming or handle existing objects carefully.
   - **Memory Pressure:** Large memory mappings consume address space and may stress the system’s paging. If too large, you can run into `STATUS_INSUFFICIENT_RESOURCES` or paging contention.

---

## Integration with Other IPC Mechanisms

1. **Named Mutexes and Semaphores for Coordination:**
   - Often combined with named synchronization objects (mutexes, events) to coordinate data access in the shared memory.
   - These synchronization objects are also kernel objects named in the same object namespace, making it easy for multiple processes to acquire and release locks around the shared data.

2. **Combining with RPC or Pipes:**
   - Share bulk data via memory mapping and control messages via a lightweight IPC (like named pipes or RPC).
   - The small messages can tell the other process to “read now” or “data updated”, reducing overhead compared to copying large amounts of data through pipe or RPC directly.

3. **Advanced Protocols:**
   - Some applications implement ring buffers or circular queues in shared memory, enabling high-throughput, low-latency communication.
   - Producer-consumer patterns are common: one process writes data sequentially, another reads it as it comes in, with pointers stored in the shared memory itself to indicate read/write offsets.

---

## Summary and Conclusion

Memory-mapped files (section objects) at the Windows internals level:

- Leverage the powerful Virtual Memory Manager (VMM) and Section Manager subsystems.
- Provide a direct path to share pages of RAM among processes, backed by file or pagefile.
- Integrate seamlessly with the object namespace, security model, paging system, and synchronization primitives.
- Offer high performance and flexible IPC patterns.

Armed with these internal details, you can design robust, high-performance shared memory systems on Windows that scale from small utility apps to large, complex, multi-process architectures. Understanding the underlying kernel mechanics helps with performance tuning, debugging subtle issues, and ensuring correct and secure usage of shared memory mechanisms.