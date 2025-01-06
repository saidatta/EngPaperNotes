https://www.youtube.com/watch?v=HEKcrGUnu-c
**Audience:**  
This note is for experienced Windows engineers who have already explored the previously discussed enumeration methods (Toolhelp, WTS, PSAPI, and `NtQuerySystemInformation`). In this part, we delve deeper into the native API approach by leveraging the **PHNT** header files to make using the native API more convenient and explore additional information classes.

---
## Introduction
In the previous parts, we covered multiple methods of enumerating processes on Windows:
- **Part 1:** Documented Windows APIs (Toolhelp, WTS, PSAPI)
- **Part 2:** The native NT API (`NtQuerySystemInformation`) directly from `ntdll.dll`
Now we refine the native approach further by introducing **PHNT**—a community-maintained set of header files that define Windows native API functions, structures, and enums not fully documented by Microsoft. With PHNT, you can work with `NtQuerySystemInformation` more easily and access additional system information classes that expose more data about processes.

---
## PHNT: What Is It and Why Use It?
**PHNT** is a collection of native API header files extracted and organized from publicly available sources (e.g., ReactOS, leaked symbols, etc.) to provide developers with a clean and consistent interface to the Windows native API. It gives you:
- Comprehensive definitions of structures like `SYSTEM_PROCESS_INFORMATION` and its variants.
- Enumerations of `SYSTEM_INFORMATION_CLASS` values.
- Declarations of native functions like `NtQuerySystemInformation`.

By using PHNT, you avoid guessing structure layouts or relying on outdated or partial definitions.

---
## Installing PHNT

**Option 1: Manual Download**  
You can clone or download PHNT from its GitHub repository. Once obtained, copy the relevant headers into your project’s include directory.

**Option 2: Using vcpkg (Recommended)**  
`vcpkg` is Microsoft’s C++ package manager, making it easy to integrate PHNT into your build. For example:

```bash
# Enable vcpkg integration with Visual Studio
vcpkg integrate install

# Install PHNT
vcpkg install phnt
```

After installation, Visual Studio or your chosen environment can automatically find and use PHNT headers.

---

## Including PHNT in Your Code

Typically, you must include two headers in this order:

```cpp
#include <phnt_windows.h>
#include <phnt.h>
```

- `phnt_windows.h` provides minimal Windows definitions so `phnt.h` can safely include native structures without conflict.
- `phnt.h` then provides definitions for NT native calls, enums, and structures.

---

## Using NtQuerySystemInformation with PHNT

### Structures and Information Classes

PHNT defines several `SYSTEM_INFORMATION_CLASS` values and associated structures, including:

- `SystemProcessInformation` → `SYSTEM_PROCESS_INFORMATION`
- `SystemExtendedProcessInformation` → `SYSTEM_PROCESS_INFORMATION` with extended fields
- `SystemFullProcessInformation` → Even more detailed info (in newer Windows versions)

These variants provide increasing levels of detail about processes and their threads.

### Example Code

Below is a more refined example than before, now using PHNT headers. This example:

1. Queries `SystemProcessInformation` for a list of processes.
2. Prints details such as PID, PPID, thread count, handle count, session ID, and image name.
3. Uses PHNT definitions to avoid manual structure definitions.

```cpp
#include <windows.h>
#include <phnt_windows.h>
#include <phnt.h>
#include <stdio.h>

// Link with ntdll (provided by Windows Kits or add via pragma)
// #pragma comment(lib, "ntdll.lib")

int main() {
    NTSTATUS status;
    ULONG size = 0;
    PVOID buffer = nullptr;

    // First call with zero-size buffer to get required size
    status = NtQuerySystemInformation(SystemProcessInformation, NULL, 0, &size);
    if (status != STATUS_INFO_LENGTH_MISMATCH) {
        printf("Unexpected status: 0x%08X\n", status);
        return 1;
    }

    // Allocate a buffer large enough to hold process list
    // Add a bit more to reduce likelihood of needing to requery immediately
    size += 4096;
    buffer = malloc(size);
    if (!buffer) return 1;

    status = NtQuerySystemInformation(SystemProcessInformation, buffer, size, &size);
    if (!NT_SUCCESS(status)) {
        printf("NtQuerySystemInformation failed: 0x%08X\n", status);
        free(buffer);
        return 1;
    }

    // Parse the returned process list
    PSYSTEM_PROCESS_INFORMATION spi = (PSYSTEM_PROCESS_INFORMATION)buffer;
    for (;;) {
        // Convert handles representing IDs to integers
        DWORD pid = HandleToULong(spi->UniqueProcessId);
        DWORD ppid = HandleToULong(spi->InheritedFromUniqueProcessId);

        // ImageName can be empty for system processes
        PWSTR imageName = (spi->ImageName.Length > 0) ? spi->ImageName.Buffer : L"(no name)";

        wprintf(L"PID: %5u  PPID: %5u  Threads: %3u  Handles: %5u  Session: %u  Name: %s\n",
            pid,
            ppid,
            spi->NumberOfThreads,
            spi->HandleCount,
            spi->SessionId,
            imageName
        );

        if (spi->NextEntryOffset == 0) break;
        spi = (PSYSTEM_PROCESS_INFORMATION)((PBYTE)spi + spi->NextEntryOffset);
    }

    free(buffer);
    return 0;
}
```

**Notes:**

- We use `HandleToULong` to safely convert `HANDLE` values to `ULONG`.
- `spi->ImageName` is a `UNICODE_STRING`, which may or may not be null-terminated. Usually, it is properly null-terminated for processes. If unsure, manually ensure termination or handle the string length carefully.
- `SessionId`, `HandleCount`, and many other fields are directly available without additional calls.

---

## Enhanced Information Classes

PHNT also exposes additional info classes:

- `SystemExtendedProcessInformation` provides a `SYSTEM_PROCESS_INFORMATION` structure with extra fields, such as enhanced thread information or additional memory counters.
- `SystemFullProcessInformation` (on newer Windows versions) may yield even more data, including process mitigations and extended flags.

To use these, simply change the `SystemInformationClass` argument and the corresponding structures, if defined. Ensure that your PHNT headers and OS version support these classes.

---

## Handling Command Lines and Other Details

Even with PHNT and the native API, you won’t get certain details directly (e.g., the command line arguments of the process). You must still:

1. Open a handle to the process with `OpenProcess()`.
2. Use documented APIs (like `NtQueryInformationProcess` with `ProcessCommandLineInformation` if available) or read PEB memory directly if you’re going into undocumented territory.

As always, more advanced data often requires additional steps and possibly elevated privileges.

---

## Considerations

1. **Compatibility:**  
   Structures and fields can differ by Windows version. PHNT tries to keep up, but always test your code on the Windows versions you target.

2. **Forward Compatibility:**  
   Future Windows updates may change internal layouts. Keep PHNT updated and consider graceful fallback paths if critical structures change.

3. **Performance:**  
   Using the native API is fast and retrieves all processes in one go. Parsing may be slightly more complex due to variable-length entries and Unicode strings.

4. **Security and Privilege:**  
   Typically, enumerating processes with `NtQuerySystemInformation` does not require elevated privileges. However, accessing certain details or performing additional queries (like opening a process to get its command line) might require admin rights or special privileges.

---

## Summary

**Part 3** builds on the native API approach by integrating PHNT. With PHNT:

- You get a stable and updated set of definitions for the native API.
- It becomes easier to use `NtQuerySystemInformation` and related structures.
- You can access richer sets of data, including extended and full process info classes, without guesswork.

This approach is ideal for advanced tooling, system internals exploration, and detailed forensics. For production code that must remain stable across OS versions, consider fallback methods or extensive testing.

By now, you have a full spectrum of tools for enumerating processes on Windows, from simple, documented APIs to the powerful but undocumented native interfaces with PHNT.
```