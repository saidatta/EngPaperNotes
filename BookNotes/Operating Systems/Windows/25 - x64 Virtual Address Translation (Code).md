**Context:**
We’re examining code that directly reads and writes the memory of a process by translating virtual addresses to physical addresses without attaching a debugger or using standard OS services like `ReadProcessMemory`. Instead, it uses kernel-mode functions and page table traversal to perform the memory access. This approach relies on x64 virtual address translation concepts discussed before:
- Virtual addresses must be translated to physical addresses via page tables.
- Once you have a physical address, you can use `MmCopyMemory` or `MmMapIoSpaceEx` to read/write physical memory directly.
**High-Level Goal:**
The code obtains the process’s CR3 (or DirectoryTableBase) to manually translate a given virtual address in that process’s context into a physical address. Then it reads or writes the target process's memory using that physical address. This bypasses normal OS-level protections and requires kernel privileges.

---
## Key Functions and Their Roles

### `PsGetProcessSectionBaseAddress(PEPROCESS Process)`
- Returns the base address of the main executable section in a given process.  
- In user-mode, `GetProcessBaseAddress()` uses this to find where the executable is loaded in the virtual address space of the target process.  
- Useful as a starting point to read memory from that process’s image.

**Snippet:**
```c
PVOID GetProcessBaseAddress(int pid) 
{
    PEPROCESS pProcess = NULL;
    if (pid == 0) return STATUS_UNSUCCESSFUL;

    NTSTATUS NtRet = PsLookupProcessByProcessId(pid, &pProcess);
    if (NtRet != STATUS_SUCCESS) return NtRet;

    PVOID Base = PsGetProcessSectionBaseAddress(pProcess);
    ObDereferenceObject(pProcess);
    return Base;
}
```
**Meaning:**  
- Looks up the `PEPROCESS` for the given PID.
- Retrieves the base address of the process (like `LoadLibrary` base address).
- Returns that address to the caller.

---
### `GetUserDirectoryTableBaseOffset()`
- Different Windows builds store the user directory table base (used for virtual-to-physical translations of user address ranges) at different offsets within the EPROCESS structure.
- This function returns the offset into the EPROCESS for `UserDirectoryTableBase`, depending on the OS build number.
  
**Meaning:**  
- For each Windows build, a different offset is used to find the user mode CR3 equivalent (UserDirectoryTableBase).  
- Ensures code can adapt to different Windows versions at runtime, a form of dynamic versioning.

---

### `GetProcessCr3(PEPROCESS pProcess)`
- Reads the process’s `DirectoryTableBase` (the CR3 value on x64).
- If the normal directory base (at offset 0x28) is zero, it tries to get `UserDirectoryTableBase` using the previously found offset from `GetUserDirectoryTableBaseOffset`.
  
**Meaning:**
- Every process has a page table hierarchy represented by CR3 or directory base.
- If the main dirbase is zero (some special processes or configurations), fallback to user directory table base.

Once you have CR3 (the directory table base), you can perform translation of virtual addresses to physical addresses.

---

### `ReadVirtual` and `WriteVirtual`
```c
NTSTATUS ReadVirtual(uint64_t dirbase, uint64_t address, uint8_t* buffer, SIZE_T size, SIZE_T *read)
{
    uint64_t paddress = TranslateLinearAddress(dirbase, address);
    return ReadPhysicalAddress(paddress, buffer, size, read);
}
```
**Meaning:**
- Translates a given virtual address (`address`) using `TranslateLinearAddress` and the process’s `dirbase`.
- Once we have a physical address (`paddress`), calls `ReadPhysicalAddress` to read from physical memory into `buffer`.

`WriteVirtual` similarly translates then writes at the physical address.

---

### `ReadPhysicalAddress` and `WritePhysicalAddress`
- `ReadPhysicalAddress` uses `MmCopyMemory` with `MM_COPY_MEMORY_PHYSICAL` to read from a physical address directly into a buffer. This does not go through the process’s virtual address translation normally used by user-mode API. It’s a raw physical memory read.
  
```c
NTSTATUS ReadPhysicalAddress(PVOID TargetAddress, PVOID lpBuffer, SIZE_T Size, SIZE_T *BytesRead)
{
    MM_COPY_ADDRESS AddrToRead = {0};
    AddrToRead.PhysicalAddress.QuadPart = (LONGLONG)TargetAddress;
    return MmCopyMemory(lpBuffer, AddrToRead, Size, MM_COPY_MEMORY_PHYSICAL, BytesRead);
}
```

- `WritePhysicalAddress` uses `MmMapIoSpaceEx` to map a physical range into kernel virtual space and then `memcpy` to that mapped range:
  
```c
NTSTATUS WritePhysicalAddress(PVOID TargetAddress, PVOID lpBuffer, SIZE_T Size, SIZE_T* BytesWritten)
{
    PHYSICAL_ADDRESS AddrToWrite = {0};
    AddrToWrite.QuadPart = (LONGLONG)TargetAddress;

    PVOID pmapped_mem = MmMapIoSpaceEx(AddrToWrite, Size, PAGE_READWRITE);
    if (!pmapped_mem)
        return STATUS_UNSUCCESSFUL;

    memcpy(pmapped_mem, lpBuffer, Size);
    *BytesWritten = Size;
    MmUnmapIoSpace(pmapped_mem, Size);
    return STATUS_SUCCESS;
}
```

**Meaning:**
- `ReadPhysicalAddress` and `WritePhysicalAddress` work with raw physical addresses.
- `WritePhysicalAddress` maps the physical memory range into a temporary kernel space pointer, copies data, then unmaps it.

---

### `TranslateLinearAddress`
This function performs the manual 4-level page table walk described in the introduction:
- Extract `pml4`, `pdpt`, `pd`, and `pt` indexes from the given virtual address.
- Reads each table entry from physical memory, checks the present bit.
- If large pages are encountered at PD or PDPT level, handle accordingly.
- Finally, returns the physical address of the page plus offset.

**Meaning:**
- Emulates the CPU’s page table lookup logic in software.
- Uses `ReadPhysicalAddress` to read page table entries from memory.
- If at any step an entry is not present, returns 0 indicating no translation (a would-be page fault scenario).

---

### `ReadProcessMemory` and `WriteProcessMemory`
- Similar to kernel-mode implementations of `ReadProcessMemory` known from user-mode.  
- For given PID, obtains `PEPROCESS` and from it the process CR3 (dirbase).
- Splits the reading/writing over page boundaries:
  1. Translate the base virtual address
  2. Read/Write up to the end of that page’s available range
  3. Move to the next page and repeat until all requested bytes are done.

**Meaning:**
- By using `TranslateLinearAddress` per page, these functions effectively read/write memory from any process, without attaching or calling standard OS routines, bypassing normal access checks.

---

## Example Usage Provided
```c
char buf[64] ={0};
SIZE_T read;
ULONG_PTR Base = GetProcessBaseAddress(4321);
ReadProcessMemory(4321, Base , &buf, 64, &read);
```

**Meaning:**
- For process with PID = 4321:
  - Gets its base address (using `PsGetProcessSectionBaseAddress`), presumably the start of the main module.
  - Reads 64 bytes from that base address directly into `buf`.
- This reveals the first bytes of the process’s main image. The output (e.g., `MZ`) might show the DOS header of a PE file.

---

## Logical Summary

This code:
1. Identifies a process by PID and retrieves its `PEPROCESS`.
2. Extracts the process’s paging structure base (CR3 or directory table base).
3. Converts a target virtual address (like process’s base address) into a physical address using a manual page table walk (`TranslateLinearAddress`).
4. Accesses physical memory directly with `MmCopyMemory` or `MmMapIoSpaceEx`.
5. Implements `ReadProcessMemory` and `WriteProcessMemory` analogs at a kernel level, bypassing normal OS checks and attachment methods.

**In simpler terms:**
- It's a kernel method to read/write another process’s memory by:
  - Getting the process’s CR3.
  - Doing the page table translation manually.
  - Reading/writing the resolved physical page(s) directly.

This is a low-level, invasive technique that overrides the normal OS-provided abstractions, used typically for advanced debugging, memory forensics, or certain sophisticated kernel tools. It relies heavily on the x64 virtual address translation mechanism we studied: splitting addresses into indexes, navigating PML4/PDPT/PD/PT tables, and dealing with large page shortcuts.

All these steps align perfectly with the theory explained in the x64 virtual address translation concept, demonstrating a practical application of these theoretical underpinnings.
```