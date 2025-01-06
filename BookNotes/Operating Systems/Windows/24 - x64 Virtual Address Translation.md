https://www.youtube.com/watch?v=KumivbtVeUU

**Audience:**  
These notes are intended for experienced Windows kernel engineers, systems programmers, and low-level software developers who want a deep technical understanding of how x64 processors translate virtual addresses into physical addresses. We will cover the concept, the structures involved (page tables), large page variants, how to inspect and verify translations using WinDbg, and discuss the CPU’s internal mechanisms like the TLB.

---
## Introduction
Modern operating systems, including 64-bit Windows, use virtual memory:
- Each process sees a **virtual address space** that may differ from the machine’s physical memory layout.
- The CPU and OS cooperate to map virtual addresses to physical memory via **page tables**.

Key motivations:
- Isolate processes from each other.
- Provide flexible memory management (paging, dynamic allocation, etc.).
- Enhance security (no direct physical addressing by user mode).

On x64 Windows:
- Processes have a huge virtual address range.
- For a 64-bit process on modern Windows: **128 TB** user space and **128 TB** kernel space (total 256 TB).
- For 32-bit processes on a 64-bit OS, address spaces differ (2 GB or 4 GB if large address aware is set).

**Important**: All CPU memory accesses use virtual addresses, even in kernel mode. The CPU must translate these virtual addresses into physical addresses on every memory operation.

---
## Basic Concept of Translation
When the CPU needs to access a virtual address, it must:
1. Split the address into components: page offsets and page directory indexes.
2. Traverse hierarchical page tables to find the corresponding physical page number.
3. Combine that physical page number with the offset to get the final physical address.
If at any stage the translation fails (e.g., a page is not present), the CPU triggers a **page fault**. The OS then either brings the page in from the page file or signals an access violation.
![[Screenshot 2024-12-13 at 2.19.08 PM.png]]
**Addressing Bits:**
- Modern 64-bit CPUs do not use all 64 bits of the address. Windows currently uses 48 bits for addressing (with high bits sign-extended).
- This yields **128 TB** per mode (user/kernel).
**Processor Registers:**
- `cr3` register contains the physical address of the top-level table (PML4) for the current address space.
---
## The Page Table Hierarchy
On x64, the translation involves up to 4 levels of tables:
1. **PML4 (Page Map Level 4) Table:**  
   - `cr3` points to this 4 KB structure.
   - It has 512 entries (indexed by 9 bits from the VA).
2. **PDPT (Page Directory Pointer Table):**  
   - Selected by the next 9 bits of the VA.
   - Each entry points to a Page Directory or a huge page.
3. **PD (Page Directory):**  
   - Another 4 KB table with 512 entries.
   - Each entry references a Page Table or a 2 MB large page.
4. **PT (Page Table):**  
   - Another 4 KB table with 512 entries.
   - Each entry points to a 4 KB physical page.
![[Screenshot 2024-12-13 at 2.42.09 PM.png]]
Finally, the last 12 bits of the virtual address serve as the offset within the final 4 KB page.
![[Screenshot 2024-12-13 at 2.45.43 PM.png]]
**Summary of Bits:**
- Virtual Address (48 bits used):
  - `[PML4 index:9 bits] [PDPT index:9 bits] [PD index:9 bits] [PT index:9 bits] [Offset:12 bits]`
**Larger Pages:**
- 2 MB pages skip the PT level (no PT needed).
- 1 GB pages skip both PT and PD levels (less commonly used in Windows by default).
---
## Translation Cache: The TLB
The **Translation Lookaside Buffer (TLB)** is a small cache in the CPU that stores recent virtual-to-physical page mappings.
- If a translation is found in TLB, no page table walk is needed.
- If not found (TLB miss), CPU performs the full 4-level lookup.
Performance heavily depends on TLB efficiency. Large pages reduce TLB misses by covering more memory with fewer entries.
---
## Handling Page Faults
If a page table entry’s "present" bit is not set:
- The CPU raises a page fault exception.
- OS’s memory manager handles it:
  - If the page is on disk (paged out), OS fetches it into memory, updates the page table, and resumes execution.
  - If address is invalid, process receives an access violation.

---
## Viewing and Debugging Translations

**Using WinDbg:**
- `!process` to see process details, including `DirectoryTableBase` (CR3).
- `!pte <Address>` to show page table entries for a given VA (though sometimes can be finicky with symbol issues).
- `!vtop <CR3> <VA>` or `vtop`/`v2p` extension to manually translate a virtual address to physical if you have CR3 or directory base.

**Example:**
```none
0: kd> !process 0 0 System
...
DirBase = 0x1873c000
...
```
`DirBase` gives the top-level PML4 physical address.

```none
0: kd> !vtop 0x1873c000 fffff80068f08000
```
Translates `fffff80068f08000` (kernel VA) to physical.

If successful, shows the corresponding physical address and page attributes.
![[Screenshot 2024-12-13 at 3.01.21 PM.png]]
**Manual Calculation:**
- Extract indexes from VA’s bits.
- Fetch entries from PML4, PDPT, PD, PT by adding `(index * 8)` to the base address of each table.
- Verify present bits and large page bits.

---
## Code and Example
While user-mode code doesn’t directly manipulate page tables, kernel code or low-level debugging can reveal them.
**Pseudo-Code to Emulate Translation (Hypothetical):**
```c
// Assume we have cr3 (the PML4 physical address), and a 48-bit VA
uint64_t TranslateVAtoPA(uint64_t cr3, uint64_t va) {
    uint64_t pml4_index = (va >> 39) & 0x1ff;
    uint64_t pdpt_index = (va >> 30) & 0x1ff;
    uint64_t pd_index   = (va >> 21) & 0x1ff;
    uint64_t pt_index   = (va >> 12) & 0x1ff;
    uint64_t offset     = va & 0xfff;

    // Read PML4 entry from cr3
    uint64_t pml4e = ReadPhysQword((cr3 & ~0xfffULL) + pml4_index * 8);
    if (!(pml4e & 1)) RaisePageFault();

    uint64_t pdpt = (pml4e & ~0xfffULL);
    uint64_t pdpte = ReadPhysQword(pdpt + pdpt_index * 8);
    if (!(pdpte & 1)) RaisePageFault();

    if (pdpte & (1ULL << 7)) {
        // large page (1GB)
        return ((pdpte & ~0x3fffffffULL) + (va & 0x3fffffffULL));
    }

    uint64_t pd = (pdpte & ~0xfffULL);
    uint64_t pde = ReadPhysQword(pd + pd_index * 8);
    if (!(pde & 1)) RaisePageFault();

    if (pde & (1ULL << 7)) {
        // large page (2MB)
        return ((pde & ~0x1fffffULL) + (va & 0x1fffffULL));
    }

    uint64_t pt = (pde & ~0xfffULL);
    uint64_t pte = ReadPhysQword(pt + pt_index * 8);
    if (!(pte & 1)) RaisePageFault();

    uint64_t page = (pte & ~0xfffULL);
    return page + offset;
}
```

This pseudo-code simulates the CPU’s logic for 4 KB and large page sizes.

---
## Internals and Limits
- The 48-bit addressing used today maps to `2^48 = 256 TB` total, half user, half kernel.
- Future expansions (57-bit virtual addressing) would allow up to `128 PB` per side if OS supported it.
- If a process tries to allocate huge amounts of memory, it can run out of commit limit, but address space fragmentation is rarely a problem on x64 due to vast space.

---
## Summary
- **x64 Virtual Address Translation** uses a 4-level page table hierarchy (PML4 → PDPT → PD → PT).
- Virtual addresses are split into indexes for each table plus a final offset.
- The CPU uses `cr3` to find PML4 and translates down through PDPT, PD, PT.
- Large pages skip levels for simpler/faster translations but are optional.
- Tools like WinDbg `!vtop`, `!pte` and manual calculation allow us to see how translation works.
- Today’s Windows uses 48-bit virtual addresses, leading to a 128 TB user + 128 TB kernel space model.
- Translation complexity is hidden by TLB caches for performance.