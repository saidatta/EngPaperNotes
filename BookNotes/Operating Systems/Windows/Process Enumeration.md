## Overview

Enumerating processes means listing all currently running processes, potentially to gather their attributes (e.g., PID, name, threads, memory usage). Common reasons include:

- **Diagnostic Tools:** List and examine running processes (like Task Manager).
- **Forensics & Security:** Detect suspicious processes or confirm environment conditions (e.g., running inside a container).
- **Automation & Monitoring:** Track process lifecycles, resources, or implement custom logic when certain processes appear.

Windows provides several APIs for this task, each with different levels of detail and complexity.

**APIs Covered:**
1. **Toolhelp API (CreateToolhelp32Snapshot)**
2. **WTS (Windows Terminal Services) API (WTSEnumerateProcessesEx)**
3. **PSAPI (EnumProcesses)**
4. (A native, undocumented API `NtQuerySystemInformation` will be discussed in a later part, not here.)

---
## Method 1: Toolhelp API
**Header & Library:**
- Include: `<windows.h>`, `<tlhelp32.h>`
- Library: Linked by default with `kernel32.lib`.
**Key Functions:**
- `CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)`  
  Creates a snapshot of the system’s process list.
- `Process32First()`, `Process32Next()`  
  Iterate over the snapshot to list `PROCESSENTRY32` structures.
**Data Provided:**
- `PROCESSENTRY32`:
  - `th32ProcessID`: Process ID
  - `th32ParentProcessID`: Parent PID
  - `cntThreads`: Thread count
  - `szExeFile`: Base name of the executable

**Example Code Snippet:**
```cpp
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>

void EnumProc_Toolhelp() {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) return;

    PROCESSENTRY32 pe = { sizeof(pe) };
    if (Process32First(hSnapshot, &pe)) {
        do {
            printf("PID: %5u  PPID: %5u  Threads: %3u  Name: %s\n",
                   pe.th32ProcessID, pe.th32ParentProcessID,
                   pe.cntThreads, pe.szExeFile);
        } while (Process32Next(hSnapshot, &pe));
    }
    CloseHandle(hSnapshot);
}
```
**Pros:**
- Easy to use.
- Provides basic info (PID, PPID, thread count, executable name).
**Cons:**
- Limited detail (no handle count, memory usage, or user info).
- For more info, you must open the process handle and query additional data.

---
## Method 2: WTS (Windows Terminal Services) API
**Header & Library:**
- Include: `<windows.h>`, `<WtsApi32.h>`, and `<sddl.h>` if converting SIDs.
- Link with: `Wtsapi32.lib`
**Key Functions:**
- `WTSEnumerateProcessesEx()`:
  - Retrieves a rich set of information: session ID, process ID, number of threads, handle count, memory usage, process name, user SID, etc.
- `WTSFreeMemory()` to free the returned process list.
**Data Provided (`WTS_PROCESS_INFO_EX`):**
- `ProcessId`
- `SessionId`
- `NumberOfThreads`
- `HandleCount`
- `WorkingSetSize`
- `pProcessName`
- `pUserSid`
- and more (PagefileUsage, KernelTime, UserTime, etc.)

**Example Code Snippet:**
```cpp
#include <windows.h>
#include <WtsApi32.h>
#include <sddl.h>     // For ConvertSidToStringSid
#include <stdio.h>
#include <string>

std::string SidToString(PSID pSid) {
    if (!pSid) return "";
    LPSTR strSid = nullptr;
    if (ConvertSidToStringSidA(pSid, &strSid)) {
        std::string sidStr(strSid);
        LocalFree(strSid);
        return sidStr;
    }
    return "";
}

void EnumProc_WTS() {
    PWTS_PROCESS_INFO_EX pInfo = NULL;
    DWORD count = 0;
    // Level=1 (documented), SessionId=0 for all sessions.
    if (!WTSEnumerateProcessesExA(WTS_CURRENT_SERVER_HANDLE, &count, 1, (LPSTR*)&pInfo, &count)) {
        return;
    }

    for (DWORD i = 0; i < count; i++) {
        auto &proc = pInfo[i];
        std::string sidStr = SidToString(proc.pUserSid);

        printf("PID: %5u  Session: %u  Threads: %3u  Handles: %5u  Name: %s  WS: %u MB  SID: %s\n",
               proc.ProcessId,
               proc.SessionId,
               proc.NumberOfThreads,
               proc.HandleCount,
               proc.pProcessName,
               (UINT)(proc.WorkingSetSize >> 20), // convert bytes to MB
               sidStr.c_str());
    }

    WTSFreeMemory(pInfo);
}
```

**Pros:**
- Rich detail: memory usage, handle count, session ID, user SID, etc.
- Can query processes per session or remotely (using server handles).

**Cons:**
- More complex than Toolhelp.
- Requires `WTSFreeMemory` to clean up.
- May need elevated privileges for some data.

---
## Method 3: PSAPI (EnumProcesses)
**Header & Library:**
- Include: `<windows.h>`, `<psapi.h>`
- Link with: `Psapi.lib`
**Key Functions:**
- `EnumProcesses()`:
  - Returns an array of PIDs.
- To get more info, open each process and call `QueryFullProcessImageName()`.

**Example Code Snippet:**
```cpp
#include <windows.h>
#include <psapi.h>
#include <stdio.h>

void EnumProc_PSAPI() {
    DWORD pids[1024], needed = 0;
    if (!EnumProcesses(pids, sizeof(pids), &needed)) return;

    DWORD count = needed / sizeof(DWORD);
    for (DWORD i = 0; i < count; i++) {
        DWORD pid = pids[i];
        HANDLE hProc = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        if (hProc) {
            char path[MAX_PATH];
            DWORD size = MAX_PATH;
            if (QueryFullProcessImageNameA(hProc, 0, path, &size)) {
                printf("PID: %5u  Name: %s\n", pid, path);
            } else {
                // Could be protected system process
                printf("PID: %5u  [No access]\n", pid);
            }
            CloseHandle(hProc);
        } else {
            printf("PID: %5u  [Cannot open]\n", pid);
        }
    }
}
```
**Pros:**
- Simple if you only need PIDs.
- Fast enumeration of PIDs.
**Cons:**
- Provides no additional info directly; must open each process for more details.
- Access to system or protected processes often fails without elevation.

---
## Comparison of the Three Methods

| API            | Info Detail | Ease of Use | Additional Data | Remote Session Support | Best For                                     |
|----------------|-------------|-------------|----------------|------------------------|-----------------------------------------------|
| Toolhelp       | Basic (PID, PPID, threads, exe) | Easy | Minimal | No | Quick basic enumeration, simple tools          |
| WTS            | Rich (session, handles, memory, SID) | Moderate | Extensive | Yes | Comprehensive process info, session-aware      |
| PSAPI (EnumProc)| Just PIDs by default | Moderate | Must open process for details | No | Fast enumeration, good if you need only PIDs   |

---

## Notes and Best Practices

- **Elevation and Access Rights:**  
  Some processes (e.g., System processes, protected processes) require admin privileges or special rights to open and query. Without these, you may fail to retrieve names or details.
  
- **Memory Management:**  
  - Toolhelp: Just close the snapshot handle.
  - WTS: Must call `WTSFreeMemory()` to free returned data.
  - PSAPI: Must manage your own arrays and handle lifecycles.

- **Performance Considerations:**  
  - `EnumProcesses` is very fast but minimal. To get full details, multiple `OpenProcess()` calls can add overhead.
  - `WTSEnumerateProcessesEx` gives comprehensive info in one call, potentially saving time but at the cost of complexity.
  - Toolhelp snapshot is easy to use and reasonably fast for basic data.

---

## Summary

Enumerating processes on Windows can be done via multiple APIs, each with its own trade-offs. For quick, basic enumeration, **Toolhelp** is simple. For rich details (e.g., handle counts, memory, sessions), **WTS** is better. For a minimal and fast approach, **PSAPI**'s `EnumProcesses()` suffices but requires additional steps to retrieve details.

In the next part, we’ll explore the **native API (NtQuerySystemInformation)** for even more detailed and low-level data, at the cost of using undocumented or semi-documented functions.
```