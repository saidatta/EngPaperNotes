https://www.youtube.com/watch?v=zdZdtg1f9lA

**Audience:**  
These notes are for experienced Windows engineers who already understand process internals, kernel objects, and memory management. In this guide, we focus on creating and using memory-mapped files (section objects) to share memory between unrelated processes.

---
## Overview
**Memory-Mapped Files** (MMFs), known internally as **section objects**, allow you to map the contents of a file (or the system’s page file) into a process’s virtual address space. By doing so, multiple processes can share the same underlying data. If you name the section object, different processes can open that object by name and thus share memory without being related (no inheritance or code-sharing tricks needed).

**Key Idea:**  
- Create a section object (memory-mapped file) with a unique name.
- Map a view of the section into each process’s address space.
- Write to the shared region in one process, read it in another.

Unlike the simple sharing method that required the same executable or DLL, memory-mapped files allow truly independent processes to share memory as long as they agree on the section object’s name.

---
## Creating a Memory-Mapped File
Use `CreateFileMapping` to create a named or unnamed memory section:
- `CreateFileMapping` can map:
  - A real disk file, making its contents accessible via memory.
  - The system’s paging file (by specifying `INVALID_HANDLE_VALUE`), creating a pure memory block with no persistent backing store.
**Example:**
```cpp
HANDLE hMapFile = CreateFileMappingW(
    INVALID_HANDLE_VALUE,   // Use pagefile as backing store
    NULL,                   // Default security
    PAGE_READWRITE,         // Read/write access
    0,                      // High-order size (0 for <4GB)
    1 << 20,                // Low-order size: 1 MB
    L"MySharedMemory"       // Name of the section object
);

if (!hMapFile) {
    wprintf(L"CreateFileMapping failed: %lu\n", GetLastError());
    return;
}
```

- The above snippet creates a 1 MB shared memory block backed by the system page file.
- The name `L"MySharedMemory"` identifies this section. Another process can open this same memory by calling `OpenFileMapping`.

---
## Mapping a View of the File
After creating or opening a file mapping, use `MapViewOfFile` to map a portion (or the entire size) into the process’s address space.
**Example:**
```cpp
LPVOID pView = MapViewOfFile(
    hMapFile,
    FILE_MAP_READ | FILE_MAP_WRITE, // Desired access
    0, 0,                           // Offset high/low: start at 0
    64 * 1024                       // View size: 64 KB
);

if (!pView) {
    wprintf(L"MapViewOfFile failed: %lu\n", GetLastError());
    CloseHandle(hMapFile);
    return;
}
```
- `pView` now points to a 64KB portion of the 1MB shared memory segment.
- You can read and write to `pView` as if it were a normal pointer to memory.

When done, call `UnmapViewOfFile(pView)` to release the mapping, and `CloseHandle(hMapFile)` when you’re finished with the section object.

---
## Reading and Writing Shared Memory
Reading and writing are straightforward: just treat `pView` like a pointer to an array of bytes, a string buffer, or a custom struct. Any changes are visible to other processes that have mapped the same portion of the same section object.

**Example Write:**
```cpp
// Suppose we have a wide-char string buffer "text"
wcscpy((wchar_t*)pView, L"Hello from Process A!");
```

**Example Read (in another process):**
```cpp
wchar_t buffer[256];
wcscpy(buffer, (wchar_t*)pView);
wprintf(L"Read from shared memory: %s\n", buffer);
```

Any process that has opened `MySharedMemory` and mapped it at the same offset gets the same data.

---
## Multiple Processes and Naming

**Named Sections:**  
If you provide a name to `CreateFileMapping`, another process can use `OpenFileMapping` with the same name to access the same shared memory:

```cpp
HANDLE hMapFile = OpenFileMappingW(
    FILE_MAP_READ | FILE_MAP_WRITE,
    FALSE,
    L"MySharedMemory"
);

LPVOID pView = MapViewOfFile(hMapFile, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 64 * 1024);
```

If successful, both processes now share the same 64KB window into the 1MB section.

**Unnamed Sections:**  
If you omit a name or pass NULL as the name parameter, you create a unique object that no other process can access unless you share it by duplicating the handle or letting the target process inherit it. Without a name or handle-sharing mechanism, no automatic sharing occurs.

---
## Synchronization
When multiple processes write to the shared memory concurrently, you need synchronization to avoid data corruption. Use standard synchronization primitives (mutexes, semaphores, events) to coordinate reads and writes.
**Example:**
- Create a named mutex (using `CreateMutex`) with a known name.
- Before writing to the shared memory block, a process calls `WaitForSingleObject(hMutex, INFINITE)`.
- After writing, it calls `ReleaseMutex(hMutex)`.
- Reading processes do the same if they need to ensure consistent reads.

---
## Persisting Data and Other Options
- **File-Backed Mapping:** Instead of `INVALID_HANDLE_VALUE`, provide a real file handle to `CreateFileMapping`. Changes to the memory region are eventually flushed to disk, allowing data persistence after all handles are closed.
- **Access Flags and Permissions:** You can create read-only mappings or even execute-in-place mappings. Adjust `PAGE_READONLY`, `PAGE_WRITECOPY`, or `PAGE_EXECUTE_READWRITE` accordingly.
- **Partial Views and Large Files:** Memory mapping allows partial views into a large file. You can map different parts of a huge file at different times, potentially improving performance for large datasets.

---
## Troubleshooting and Observations

- If `CreateFileMapping` fails with `ERROR_ACCESS_DENIED` or similar, ensure you have proper privileges and no conflicting object already exists with incompatible parameters.
- If `MapViewOfFile` fails, check that your requested size and offsets are valid and that the protection flags match the original `CreateFileMapping` call.
- Tools like **Process Explorer** show section objects in the handle view. Observing handle counts can help verify if multiple processes share the same named section.

---

## Example: Simple Text-Sharing Application

Consider two independent GUI applications:
1. **Process A (Writer):**  
   - Creates a named section "MySharedMemory".
   - Maps a view, writes a string from an edit box.
   - Unmaps the view.
   
2. **Process B (Reader):**  
   - Opens "MySharedMemory".
   - Maps the same portion of memory.
   - Reads the string into its edit box.

Result: Both see the same data. Changes made by one appear in the other after re-reading.

---

## Summary

Memory-mapped files (section objects) provide a flexible, general-purpose method to share memory between independent processes. By naming the section, any process can open and map it, enabling shared memory without requiring code sharing or process relationships. Just remember to handle synchronization, choose appropriate memory protections, and consider whether the data should persist using a file-backed mapping or remain temporary with a pagefile-backed mapping.

With this knowledge, you can implement sophisticated IPC (Inter-Process Communication) mechanisms that leverage fast, memory-like access patterns for shared data.

---
Below are additional, deeply detailed Obsidian notes that delve further into the Windows internals underlying memory-mapped files (sections) and shared memory. These notes assume you are already familiar with the basic usage of `CreateFileMapping`, `OpenFileMapping`, `MapViewOfFile`, and the general concept of section objects. The focus here is on understanding the internal mechanisms, data structures, and system calls that enable these capabilities, as well as considerations for advanced scenarios.

----
**PART II**
**Audience:**  
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
   - The small messages can tell the other process to “read now” or “data updated”, reducing overhead compared to copying large amounts of data through pipe or RPC directly
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