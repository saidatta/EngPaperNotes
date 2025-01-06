https://www.youtube.com/watch?v=Sy9CTb7JX7k

**Audience:**  
These notes target experienced Windows engineers familiar with process memory, 32-bit vs. 64-bit architectures, and internal aspects of the Windows memory manager. We will deeply discuss how the size of a process’s address space is determined, the role of the “large address aware” flag, differences on 32-bit vs. 64-bit systems, and provide examples and internal reasoning.

---

## Overview

A process’s address space is the set of virtual addresses it can use. In Windows, the size of this address space depends on:

1. **Process Architecture:**  
   - **32-bit (x86)** processes typically have a 2 GB user-mode address space by default.
   - **64-bit (x64 or AArch64)** processes have a much larger user-mode address space, commonly 128 TB (terabytes) in modern Windows versions.

2. **Large Address Aware Flag (LAA):**  
   For 32-bit processes on a 64-bit system, enabling the “large address aware” flag increases the user-mode space from 2 GB to 4 GB. For 64-bit processes, this flag is typically set by default, allowing them to use the full 128 TB without limitations.

3. **System and Kernel Constraints:**  
   The kernel-mode address space is separate and mapped in high ranges. User mode only sees a portion reserved for user space. On 64-bit Windows, the kernel uses a similar 128 TB region for system space, leaving a large unmapped gap between user and kernel addresses.

---

## Address Space Layout by Architecture

### 64-bit Processes

- **Default Address Space:**
  - User-mode: ~128 TB (terabytes)
  - Kernel-mode: ~128 TB  
  Total potential virtual address space is huge, but hardware and OS choose not to utilize all 64 bits of virtual addresses. Current Intel/AMD CPUs commonly support 48 or 57 bits of virtual addressing. Windows currently uses a 48-bit virtual address space scheme (128 TB) for user and kernel each.

- **Internals:**
  The Windows memory manager (ntoskrnl + win32k) arranges top half of the address space for kernel, bottom half for user. Since the user space is huge, running out of virtual address space is rare.

- **LAA in 64-bit:**
  Typically always enabled by the linker for 64-bit builds. Not having it would artificially restrict the process to just 2 GB, which defeats the point of a 64-bit process.

### 32-bit Processes

- **No Large Address Aware (Default Case):**
  - User-mode: 2 GB 
  - Kernel-mode: 2 GB (on a 32-bit system)
  
  On a 64-bit system, the kernel does not share address space with user mode in the same way as 32-bit Windows did, but the 32-bit process emulates the traditional layout. Without LAA, a 32-bit process, even on a 64-bit OS, is still limited to 2 GB of user space.

- **With Large Address Aware:**
  - User-mode: 4 GB (on a 64-bit OS)  
  This doubles the available user space. On a native 32-bit Windows system with special boot options (/3GB switch), LAA could yield 3 GB of user space, but on a 64-bit system, LAA for a 32-bit process gives a full 4 GB.

- **Reason for the Flag:**
  Historically, some 32-bit applications assumed the top bit of addresses (bit 31) would never be set (since 2 GB < 2^31). They used that bit as a flag. The LAA flag signals that the application is prepared to handle addresses >2 GB, thus no assumptions about top bits.

---

## Observing Address Space with Tools

**VMMap (from Sysinternals):**
- Launch a process, then run `VMMap`.
- Select the process and observe the memory layout.
  
For a 64-bit process:
- VMMap shows a massive free region indicating ~128 TB user space.
  
For a 32-bit process without LAA:
- VMMap shows only ~2 GB user space.
  
For a 32-bit process with LAA (on x64 Windows):
- VMMap shows a ~4 GB user space.

**Code Example:**

```c
// Minimal example: just a process staying alive so we can observe it in VMMap.

// Compile as 32-bit without LAA:
// cl /EHsc /O2 /Zi /MD winapp.cpp /link /subsystem:windows

// Compile as 32-bit with LAA:
// Add "/LARGEADDRESSAWARE" to link command or enable in Visual Studio:
// cl /EHsc /O2 /Zi /MD winapp.cpp /link /subsystem:windows /LARGEADDRESSAWARE

// For 64-bit (which sets LAA by default):
// cl /EHsc /O2 /Zi /MD winapp.cpp /link /subsystem:windows /MACHINE:X64

#include <windows.h>

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR cmd, int show) {
    // Just sleep forever so we can inspect address space:
    Sleep(INFINITE);
    return 0;
}
```

Run the compiled executable and then open VMMap to analyze.

---

## Internals and Memory Manager Behavior

- **Memory Manager and Virtual Memory:**
  The Windows memory manager sets aside a large region for user-mode. When an application allocates memory (VirtualAlloc), the memory manager picks a free region from the user space.

- **Hardware Address Bit Limits:**
  While 64-bit pointers can theoretically address 16 exabytes (2^64), current architectures limit physical and virtual address sizes. Windows chooses a stable subset (48 bits = 256 TB total, split ~128 TB user, 128 TB kernel).

- **Kernel-Mode Separation:**
  On 64-bit systems, user and kernel are separated more clearly than on 32-bit systems. On 32-bit, kernel and user shared a 4 GB space, typically 2 GB user, 2 GB kernel by default. With /3GB switch, it might be 3 GB user, 1 GB kernel. On 64-bit, user and kernel have vast separate spaces, eliminating much of that complexity.

---

## Advanced Scenarios

- **Changing Large Address Aware Post-Build:**
  You can use tools like `Editbin /LARGEADDRESSAWARE` on a PE file after compilation to enable/disable this flag.  
  Example:
  ```bash
  editbin /LARGEADDRESSAWARE myapp.exe
  ```
  
- **Memory Pressure and Limitations:**
  Large address space does not guarantee unlimited memory. You still need physical RAM or pagefile-backed commits. The OS can run out of commit even if lots of virtual address space is free.

- **Future Expansions:**
  As CPU and OS evolve, Windows may adopt larger virtual address spaces by enabling more bits (e.g., 57-bit addressing for 128 PB). The concept remains: LAA for 32-bit processes signals that addresses might use more bits.

---

## Summary

- **64-bit processes:** ~128 TB user-space address space by default (LAA is on by default).
- **32-bit processes (no LAA):** 2 GB of user-space on a 64-bit OS.
- **32-bit processes (with LAA):** 4 GB of user-space on a 64-bit OS.

The large address aware flag is crucial for 32-bit applications to fully utilize the 4 GB space on a 64-bit system. Without it, they remain stuck at 2 GB for compatibility reasons.

By inspecting running processes with tools like VMMap, you can observe these differences in action. Internally, the memory manager and kernel decide the layout based on CPU architecture, OS design choices, and the LAA flag set in the PE file.

Thus, the address space size is a function of architecture (32 vs. 64-bit), OS specifics, and optional flags like LAA.
```