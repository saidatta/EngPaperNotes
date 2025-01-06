**Audience:**  
These notes target experienced Windows engineers familiar with system internals, the Windows NT native API, and working with undocumented or partially documented functions.

---
## Overview

In the previous part, we focused on official, documented Windows APIs:
- **Toolhelp API**  
- **WTS (Terminal Services) API**  
- **PSAPI (EnumProcesses)**

These APIs are straightforward and supported, but each has limitations. To gain more comprehensive or low-level details, you can resort to the native NT API, specifically `NtQuerySystemInformation`. This function provides extensive information about processes, threads, and other system objects, often more than the documented APIs. However, it is:
- **Undocumented** (or only partially documented) and subject to change.
- Requires careful usage and compatibility testing.
- Returns complex data structures that vary between Windows versions.

**Use Cases:**
- Gathering extensive data for advanced tooling (like deep system monitoring, specialized debugging, or forensic analysis).
- Accessing fields not exposed by documented APIs (e.g., unique kernel identifiers, detailed memory stats, process mitigations).

---

## NtQuerySystemInformation Basics

**Function Prototype (from winternl.h):**
```c
NTSTATUS NTAPI NtQuerySystemInformation(
    SYSTEM_INFORMATION_CLASS SystemInformationClass,
    PVOID SystemInformation,
    ULONG SystemInformationLength,
    PULONG ReturnLength
);
```

- `SystemInformationClass` specifies what info to retrieve.  
  For process enumeration, we use `SystemProcessInformation` (0x05).
- `SystemInformation` is a buffer that will receive a chain of `SYSTEM_PROCESS_INFORMATION` structures.
- `SystemInformationLength` is the size of the buffer in bytes.
- `ReturnLength` optionally receives the required size or the used size.

**Key Structure: `SYSTEM_PROCESS_INFORMATION`**  
This structure provides a wealth of data:
- NextEntryOffset (to get to the next process record)
- NumberOfThreads
- CreateTime, UserTime, KernelTime
- ProcessId, ParentProcessId
- HandleCount, SessionId
- WorkingSetSize, PeakWorkingSetSize, VirtualSize
- ImageName (UNICODE_STRING)
- Priority, BasePriority
- And many more fields…

**Accessing `NtQuerySystemInformation`:**
- Declared in `<winternl.h>`.
- Linked via `ntdll.dll`.
- No import library for `ntdll.dll` is provided by default in older SDKs, but modern SDKs may have `ntdll.lib`. If not, you must use `LoadLibrary("ntdll.dll")` and `GetProcAddress()`.

---
## Example Code for NtQuerySystemInformation

Below is an example of enumerating processes using `NtQuerySystemInformation`. Keep in mind this involves undocumented fields and may change between Windows versions.

```cpp
#include <windows.h>
#include <winternl.h> // for NtQuerySystemInformation, SYSTEM_PROCESS_INFORMATION
#include <stdio.h>

// Link with ntdll if available. Otherwise, use GetProcAddress dynamically.
// #pragma comment(lib, "ntdll.lib")

// If not defined in winternl.h for your environment:
typedef NTSTATUS (NTAPI *pfnNtQuerySystemInformation)(
    SYSTEM_INFORMATION_CLASS, PVOID, ULONG, PULONG
);

void EnumProc_Native() {
    // Dynamically get NtQuerySystemInformation if not linked
    HMODULE hNtdll = LoadLibraryA("ntdll.dll");
    if (!hNtdll) {
        printf("Error loading ntdll.dll\n");
        return;
    }

    pfnNtQuerySystemInformation NtQuerySystemInformation = 
        (pfnNtQuerySystemInformation)GetProcAddress(hNtdll, "NtQuerySystemInformation");

    if (!NtQuerySystemInformation) {
        printf("Error finding NtQuerySystemInformation\n");
        FreeLibrary(hNtdll);
        return;
    }

    ULONG size = 0x10000; // initial guess
    PBYTE buffer = (PBYTE)malloc(size);
    if (!buffer) return;

    NTSTATUS status;
    for (;;) {
        status = NtQuerySystemInformation(SystemProcessInformation, buffer, size, &size);
        if (status == STATUS_INFO_LENGTH_MISMATCH) {
            // Need a bigger buffer
            free(buffer);
            buffer = (PBYTE)malloc(size);
            if (!buffer) return;
            continue;
        }
        break;
    }

    if (!NT_SUCCESS(status)) {
        printf("NtQuerySystemInformation failed: 0x%08X\n", status);
        free(buffer);
        FreeLibrary(hNtdll);
        return;
    }

    // Parse the returned buffer
    PSYSTEM_PROCESS_INFORMATION spi = (PSYSTEM_PROCESS_INFORMATION)buffer;

    for (;;) {
        // Process Id
        DWORD pid = (DWORD)(ULONG_PTR)spi->UniqueProcessId;
        // Process name (if present)
        WCHAR nameBuffer[MAX_PATH];
        int nameLen = 0;
        if (spi->ImageName.Length > 0 && spi->ImageName.Buffer != NULL) {
            nameLen = min((int)(spi->ImageName.Length / sizeof(WCHAR)), MAX_PATH - 1);
            wcsncpy_s(nameBuffer, spi->ImageName.Buffer, nameLen);
            nameBuffer[nameLen] = L'\0';
        } else {
            wcscpy_s(nameBuffer, L"System Process");
        }

        // Display some info
        wprintf(L"PID: %5u  Threads: %3u  Handles: %5u  Name: %s  WS: %u KB\n",
                pid,
                spi->NumberOfThreads,
                spi->HandleCount,
                nameBuffer,
                (UINT)(spi->WorkingSetSize / 1024));

        // Move to next process
        if (spi->NextEntryOffset == 0) break;
        spi = (PSYSTEM_PROCESS_INFORMATION)((PBYTE)spi + spi->NextEntryOffset);
    }

    free(buffer);
    FreeLibrary(hNtdll);
}
```

**What This Example Shows:**
- It retrieves a dynamic list of processes using `NtQuerySystemInformation`.
- Iterates over a linked list of `SYSTEM_PROCESS_INFORMATION` entries.
- Extracts PID, handle count, thread count, working set, and image name.
- Prints them out similarly to other APIs, but with more detail possible.
**Pros:**
- Rich data and more fields than other APIs.
- Allows deep insights: CPU times, memory usage, I/O counters, etc.
- No need for multiple calls or handle openings for basic data.
**Cons:**
- Undocumented and not guaranteed stable between Windows releases.
- Requires careful memory allocation and re-allocation.
- Potentially breaks with future Windows updates.

---
## Considerations for Using the Native API
1. **Compatibility:**  
   The structure layouts and field meanings can change. You should test across different Windows versions and maintain backward compatibility if your tool targets multiple versions of Windows.
2. **Error Handling:**  
   Handle `STATUS_INFO_LENGTH_MISMATCH` by resizing the buffer. Consider other potential NTSTATUS error codes.
3. **Security and Privileges:**  
   While `NtQuerySystemInformation(SystemProcessInformation)` doesn’t usually require special privileges, some fields or related calls might. Always check access rights if you plan to augment with other system calls.
4. **Performance:**  
   Directly retrieving all process info in one call can be efficient. However, parsing large structures is more complex.
5. **Updates & Documentation:**  
   Keep an eye on unofficial documentation sources like **ReactOS** source code, **Undocumented NT Internals** websites, and community forums. Official Microsoft documentation may be limited.
---
## Summary
The native NT API via `NtQuerySystemInformation` provides the most comprehensive and low-level data about processes. While powerful, it’s undocumented and may break between OS updates. For stable, long-term solutions, prefer documented APIs. For advanced tooling, debugging, or forensics in controlled environments, using `NtQuerySystemInformation` can unlock capabilities that other APIs cannot match.