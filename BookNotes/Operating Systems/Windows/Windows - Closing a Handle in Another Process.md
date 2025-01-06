https://www.youtube.com/watch?v=yz9u7qA-XyM
	
## Overview

By default, `CloseHandle()` only works on handles within the current process. However, sometimes you need to close a handle owned by a different process. While there’s no direct API to forcefully close a foreign handle, Windows provides a way via the `DuplicateHandle()` function combined with the `DUPLICATE_CLOSE_SOURCE` flag.

**Key Concept:**  
- **`CloseHandle()`**: Closes a handle in the current process only.
- **`DuplicateHandle()` with `DUPLICATE_CLOSE_SOURCE`**:  
  Allows you to "duplicate" a handle from a target process into yours or simply close it in the target process without obtaining it locally.

**Requirements:**
- You must have sufficient privileges and rights (`PROCESS_DUP_HANDLE`) on the target process.
- You need the numeric value of the handle you wish to close in the target process.

---
## Two Main Approaches
1. **Code Injection (Harder)**:  
   Inject code (e.g., via `CreateRemoteThread()` or `QueueUserAPC()`) into the target process and call `CloseHandle()` inside it.  
   - **Drawback**: Requires powerful access rights and complex steps.
2. **DuplicateHandle with DUPLICATE_CLOSE_SOURCE (Easier)**:  
   Use `DuplicateHandle()` from your process with special flags that instruct the system to close the source handle in the target process.

**We focus on the second, simpler approach.**

---
## Detailed Steps
1. **Obtain a Handle to the Target Process:**
   - Use `OpenProcess(PROCESS_DUP_HANDLE, FALSE, targetPid);`
   - This gives you a handle with enough rights to duplicate handles from that process.
2. **Call `DuplicateHandle()` with `DUPLICATE_CLOSE_SOURCE`:**
   - Pass the target process handle as the source.
   - Pass the handle value (from the target process) that you want to close.
   - Use `NULL` or your own process handle as the destination process handle.
   - Specify `NULL` for the target handle variable if you only want to close the handle, not duplicate it.
   - Include the `DUPLICATE_CLOSE_SOURCE` flag to tell Windows to close that handle in the target process.
3. **Check Results:**
   - If `DuplicateHandle()` succeeds, the handle in the target process is closed.
   - If it fails, check `GetLastError()` for details.
**No code injection needed!**

---
## Example Code

```cpp
#include <windows.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: CloseHandleInOtherProcess <PID> <HandleValue>\n");
        return 1;
    }

    // Parse arguments
    DWORD pid = strtoul(argv[1], NULL, 0);
    HANDLE handleToClose = (HANDLE)(ULONG_PTR)strtoul(argv[2], NULL, 0);

    // Open target process with duplicate handle rights
    HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, pid);
    if (!hProcess) {
        printf("Error: Could not open process %lu. GetLastError=%lu\n", pid, GetLastError());
        return 1;
    }

    // Use DuplicateHandle with DUPLICATE_CLOSE_SOURCE
    BOOL success = DuplicateHandle(
        hProcess,         // Source process handle
        handleToClose,    // Source handle to close
        NULL,             // Target process handle (NULL means we don’t actually need the duplicated handle)
        NULL,             // Out parameter for new handle (not needed here)
        0,                // Desired access (not needed since we’re closing)
        FALSE,            // Inheritance
        DUPLICATE_CLOSE_SOURCE // Important: closes the source handle in the target process
    );

    if (!success) {
        printf("Error duplicating/closing handle. GetLastError=%lu\n", GetLastError());
    } else {
        printf("Success: Closed handle %p in process %lu.\n", handleToClose, pid);
    }

    CloseHandle(hProcess);
    return 0;
}
```

---
## Use Cases
- **Terminating Single-Instance Locks:**  
  Some applications create a named mutex or event to ensure only one instance runs. By closing that handle in the original instance’s process, you can trick the application into allowing a second instance.
- **Debugging / Administration:**  
  System administrators or debugging tools might need to remove stale handles in malfunctioning processes to free resources.

---
## Practical Considerations

- **Permissions:**  
  You must have the right to call `OpenProcess(PROCESS_DUP_HANDLE)`. If the target process is protected (e.g., protected process, high-integrity), it may fail.
- **Handle Values:**  
  You need the exact handle value (a number like `0x188`) from the target process. Tools like Process Explorer show handle values.
- **No Guarantee of Object Lifetime:**  
  If other handles reference the same object, the object will remain until all handles are closed. Closing one handle may not always destroy the object, but it reduces references.
- **Caution:**  
  Arbitrarily closing handles in other processes can cause unpredictable behavior and instability if that handle was critical to the application’s operation. Use this technique responsibly, typically for debugging or controlled scenarios.

---
## Summary

- **Normal Behavior:** `CloseHandle()` affects only handles in the current process.
- **Solution:** `DuplicateHandle()` with `DUPLICATE_CLOSE_SOURCE` allows you to close handles in another process.
- **Outcome:** Streamlined, no injection required, provided you have the right access and know the handle value.

This powerful technique can be part of advanced debugging, recovery, or administrative tasks on Windows systems.
```