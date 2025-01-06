**Audience:**  
These notes are intended for experienced Windows engineers with an understanding of virtual memory, page allocation, and process memory management at both user and kernel levels. We will discuss the conceptual and technical differences between **reserved** and **committed** memory, show code examples illustrating their use, and dive into how Windows manages these states internally.

---
## Overview
In Windows, virtual memory for a process can exist in three main states:
1. **Free:** The virtual address range is not allocated or mapped to anything. Accessing it results in an immediate access violation.
2. **Reserved:** The address range is reserved for a future purpose but not yet backed by physical storage (RAM or pagefile). It's a placeholder ensuring that no other allocations overlap this range. Accessing reserved memory without committing it results in an access violation.
3. **Committed:** The address range is backed by physical storage (RAM or pagefile). The system guarantees these pages are available (though they can be paged out and later paged in) when the process tries to access them. Accessing committed memory works without exceptions, as the OS is "committed" to providing those bytes.
**Key Point:**  
- **Reserved Memory**: Occupies only virtual address space and some kernel bookkeeping structures, but not actual RAM or paging file space at allocation time.  
- **Committed Memory**: Requires real resources—either RAM or space in the pagefile to guarantee availability.

---
## Why Have Reserved Memory?
Imagine you need a large, contiguous memory region for a data structure (e.g., a huge matrix). If you commit it all upfront, you consume a lot of committed memory, pressuring the system even if you don't use most of it.

By **reserving** a large range first, you:
- Guarantee a large, contiguous block of addresses.
- Postpone actually "paying" (committing) memory until you need it.
- Commit smaller chunks (pages) on-demand as the program accesses them.

This approach is crucial for large, sparse data structures or stacks that grow dynamically, such as thread stacks and big application buffers.

---
## Internals: How Windows Manages These States
- **Page Tables and VADs:**  
  Windows uses **Virtual Address Descriptors (VADs)** to track reserved regions. For reserved memory, no page table entries map to physical pages. For committed memory, the page table entries point to actual RAM or to placeholders in the pagefile.

- **Kernel Structures:**  
  The kernel maintains metadata about each region. A "reserved" region has an entry in the process's VAD tree, indicating the address range is taken but has no physical backing yet.  
  When you commit a page within a reserved region, Windows updates page tables and ensures space in RAM or pagefile.

- **On-Demand Paging:**  
  Even committed memory can be paged out. But at least the system ensures a slot in pagefile, guaranteeing that on access, the page can be brought back.

---
## Memory States in Practice

**States from a CPU Perspective:**
- **Free:** CPU translating virtual address finds no mapping → immediate invalid.
- **Reserved:** Still no physical mapping. If accessed, CPU triggers page fault → OS sees no committed page → access violation.
- **Committed:** CPU can find or load a page. If in RAM, immediate success. If paged out, page fault leads the memory manager to bring it back from pagefile.

---
## Code Examples

### Committing Memory Directly

```c
#include <windows.h>
#include <stdio.h>

int main() {
    SIZE_T size = 10ULL * 1024 * 1024 * 1024; // 10 GB
    void* p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) {
        printf("Failed to allocate 10GB committed memory.\n");
        return 1;
    }

    // Accessing the memory is safe, it’s guaranteed.
    char* ptr = (char*)p;
    // Write something small
    const char* text = "Hello, memory";
    strcpy(ptr, text);

    // Check Task Manager or VMMap: commit will have increased by ~10GB.
    getchar(); // Wait before exiting to inspect memory usage

    VirtualFree(p, 0, MEM_RELEASE);
    return 0;
}
```

**What happens internally?**  
- Immediately on allocation, Windows commits 10GB. Task Manager's "Commit" counter jumps by ~10GB.
- Memory is zero-initialized lazily on first access. If you inspect memory in a debugger, it appears as zeros even before writing.

### Reserving First, Committing Later

```c
#include <windows.h>
#include <stdio.h>
#include <excpt.h>

static LONG FixMemoryFilter(LPEXCEPTION_POINTERS ep, void* cellAddr, SIZE_T cellSize) {
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        // Commit page containing cellAddr
        void* pageBase = (void*)((ULONG_PTR)cellAddr & ~(4095ULL)); // round down to page boundary
        if (!VirtualAlloc(pageBase, cellSize, MEM_COMMIT, PAGE_READWRITE)) {
            return EXCEPTION_CONTINUE_SEARCH; // can't fix
        }
        return EXCEPTION_CONTINUE_EXECUTION; 
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

int main() {
    SIZE_T totalSize = 10ULL * 1024 * 1024 * 1024; // 10GB reserved
    void* bigBlock = VirtualAlloc(NULL, totalSize, MEM_RESERVE, PAGE_NOACCESS);
    if (!bigBlock) {
        printf("Failed to reserve 10GB.\n");
        return 1;
    }

    // No commit yet, no commit charge. If we inspect with VMMap, we see a huge reserved region but no commit increase.

    // On first access, we handle the fault:
    char* cell = (char*)bigBlock; // first cell
    __try {
        // Causes access violation (not committed yet)
        strcpy(cell, "Hello, memory");
    } __except(FixMemoryFilter(GetExceptionInformation(), cell, 4096)) {
        // If we reach here, something went really wrong
        printf("Could not fix memory.\n");
    }

    // After the exception handling, cell is now committed
    // Next accesses won't fault:
    strcpy(cell, "Hello again!");

    getchar(); // Wait to inspect memory usage in VMMap or Task Manager

    VirtualFree(bigBlock, 0, MEM_RELEASE);
    return 0;
}
```

**What happens here?**  
- Initially reserves 10GB: no increase in commit.
- First access triggers access violation.
- Exception handler commits the page containing the cell.
- Now that page is committed, further accesses are fine.
- Only a single 4KB page committed out of the 10GB reserved region.

---

## Observing in VMMap or Task Manager

- **VMMap (Sysinternals):**  
  Displays regions as reserved (yellow or different colors) and committed (green sections). Initially, large reserved region appears with almost zero commit. After fault handling, a small commit segment appears inside the large reserved block.

- **Task Manager (Performance tab):**
  - **Commit** counter: Increases significantly if you commit large memory. For reserved-only allocations, commit doesn't grow.
  - **Memory in Use (RAM):** Might not change until you actually write to a committed page, causing page zeroing and eventual real RAM usage.

---

## When to Use Each Approach

- **Direct Commit:**  
  Simpler. If you know you need all the memory immediately, commit at once. Useful for smaller allocations or guaranteed usage.

- **Reserve Then Commit On-Demand:**  
  Complex but efficient. For large, sparse data or unknown future usage patterns, reserve a big range upfront and commit pages on-demand (often via exception handling or explicit VirtualAlloc calls).

- **Hybrid Strategies:**  
  Set a baseline (e.g., initial stack size = commit a small portion) and reserve a large potential area. Expand as needed.

---

## Internals Summary

- **Reserved = Address Space Promise:** The kernel sets aside address ranges by updating the VAD tree for the process. No physical memory or pagefile usage occurs at this point, just a record that these addresses are off-limits for other allocations.

- **Committed = Resource Allocation:**  
  The memory manager allocates pagefile space or RAM to ensure that if the page is touched, it can deliver. Page tables and PFNs (Page Frame Numbers) are set up, ensuring a stable backing store for that memory.

- **Changing States:**  
  `VirtualAlloc` with `MEM_COMMIT` upgrades reserved memory to committed. `VirtualFree` with `MEM_DECOMMIT` returns committed pages to a reserved state, freeing actual physical resources but preserving the reservation. `MEM_RELEASE` frees it all, reverting to free.

---

## Summary

- **Reserved Memory:**  
  Just a promise; the address range is earmarked but not backed by RAM or pagefile. It's a flexible way to secure large contiguous address spaces without paying for them until needed.

- **Committed Memory:**  
  Fully backed. Costly in terms of system commit limits. Always available to the process without exceptions.

By understanding the nuances, developers can design memory management strategies that offer large data structures, efficient lazy allocations, and dynamic growth without straining system resources unnecessarily.
```