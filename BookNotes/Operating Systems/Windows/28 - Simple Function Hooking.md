https://www.youtube.com/watch?v=TxBGBz7FRyk
## Introduction
Function hooking is a technique that allows you to intercept and modify behavior of existing functions (APIs) at runtime. Instead of calling a function and getting its normal behavior, you patch or redirect it so that your custom code (the "hook") executes first. You can choose whether to call the original function after doing some work, or prevent it from being called entirely. 
This is used in many scenarios:
- **Monitoring**: Log calls, parameters, and return values of critical functions.
- **Modifying Behavior**: Alter arguments or results to change how software behaves.
- **Debugging and Tracing**: Understand internal calls without modifying the original source code.
In these notes, we’ll focus on a simple hooking scenario on Windows using the **Detours** library by Microsoft. Detours automates much of the complexity of inline hooking. We will also cover the internal aspects of hooking.

---
## Core Concepts
### Inline Hooking
**Inline hooking** modifies the instructions at the start of a function. Typically, the start of a function begins with a few instructions. By overwriting these instructions with a jump (`jmp`) to your custom code, you gain control whenever someone calls this function. Your shell code or hook code can then call the original instructions (if needed) and return to the function’s original flow.

**Key Points:**
- Inline hooking does not rely on IAT (Import Address Table) patching.
- It affects all calls to that function: static, dynamic, `GetProcAddress` calls – everything.
- You need careful code to back up and restore original instructions (the Detours library does this for you).

### Detours Library
The Detours library by Microsoft simplifies inline hooking by:
- Starting and committing "transactions".
- Providing `DetourAttach` and `DetourDetach` functions to attach/detach hooks.
- Handling saving/restoring original bytes and instructions.
- Works on x86 and x64.

### The Process
1. **Identify Target Function**: Choose what function you want to hook, e.g., `Ellipse` from GDI32 or `HeapAlloc` from Kernel32.
2. **Define Original Function Pointer**: A global or static pointer that will store the original function’s address after hooking.
3. **Define Hook Function**: A function with the same prototype as the original. In the hook function, you can:
   - Inspect parameters.
   - Call the original function via the saved original pointer.
   - Modify return values or arguments.
4. **Begin Detour Transaction**: `DetourTransactionBegin()`.
5. **Update Transaction**: `DetourUpdateThread()` and `DetourAttach()`.
6. **Commit the Transaction**: `DetourTransactionCommit()` to apply the hook.
7. **Call your program as normal**: Now calls to the targeted function will go through your hook.

---

## Example: Hooking `Ellipse` with Detours

### Setup
Imagine we have a program drawing an ellipse:
```cpp
#include <windows.h>
#include <detours.h> // from NuGet or installed separately

// original pointer
decltype(&Ellipse) OriginalEllipse = Ellipse; 

// our hook
BOOL WINAPI HookEllipse(HDC hdc, int left, int top, int right, int bottom) {
    // Before calling the original, we can log or modify parameters
    ATLTRACE("Ellipse called: (%d, %d, %d, %d)\n", left, top, right, bottom);

    // For demonstration, let's draw a smaller ellipse
    left += 30; top += 30;
    right -= 30; bottom -= 30;

    // maybe change the pen color by selecting a custom pen
    HPEN hPen = CreatePen(PS_SOLID, 6, RGB(255,0,0)); 
    HPEN hOldPen = (HPEN)SelectObject(hdc, hPen);

    // call the original ellipse
    BOOL ret = OriginalEllipse(hdc, left, top, right, bottom);

    // restore old pen
    SelectObject(hdc, hOldPen);
    DeleteObject(hPen);

    return ret;
}

void InstallHooks()
{
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&)OriginalEllipse, HookEllipse);
    DetourTransactionCommit();
}

// In WinMain or main
int APIENTRY wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int)
{
    InstallHooks();
    // ... rest of your code that draws ellipses ...
}
```

**Explanation:**
- `OriginalEllipse` is initially set to `Ellipse` (the original API).
- `DetourAttach` will replace `Ellipse` with `HookEllipse` and modify `OriginalEllipse` to point to the original code.
- Every call to `Ellipse` now goes to `HookEllipse`.
- Inside `HookEllipse`, we log parameters, adjust them, and call `OriginalEllipse` to actually draw the ellipse.

---

## Internals of Detours

Without Detours, you would:
- Read the first few bytes of `Ellipse`.
- Save them somewhere.
- Overwrite them with a jump to `HookEllipse`.
- In `HookEllipse`, to call the original function, you:
  - Create a "trampoline" that has these saved bytes + a jump back to `Ellipse+X` where `X` is number of bytes stolen.
- Detours automates this complexity:
  - It analyzes the target function start.
  - Allocates a hidden region (trampoline) to replicate original instructions.
  - Writes a jump from the original function to your hook.
  - Saves pointer to the trampoline for calling original code from hook.

### Memory Layout
- **Original Function (e.g. Ellipse)**:
  - [jmp to HookEllipse] replaced at start.
- **Trampoline** (allocated by Detours):
  - Original instructions bytes
  - jump back to `Ellipse + offset`

### Thread Safety
- Detours tries to handle hooking in a thread-safe manner:
  - Must call `DetourUpdateThread` for threads that might be executing the function you are hooking. This ensures no conflicts.
  - Typically, calling `DetourUpdateThread(GetCurrentThread())` suffices if your hooking is done from initialization code.

---

## Handling Multiple Hooks

You can hook multiple functions:
```cpp
decltype(&HeapAlloc) OriginalHeapAlloc = HeapAlloc;

PVOID WINAPI HookHeapAlloc(HANDLE hHeap, DWORD dwFlags, SIZE_T dwBytes)
{
    ATLTRACE("HeapAlloc called: size=%zu\n", dwBytes);
    return OriginalHeapAlloc(hHeap, dwFlags, dwBytes);
}

void InstallMultipleHooks()
{
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    DetourAttach(&(PVOID&)OriginalEllipse, HookEllipse);
    DetourAttach(&(PVOID&)OriginalHeapAlloc, HookHeapAlloc);

    DetourTransactionCommit();
}
```

**Key Point:**
All attachments must happen between `DetourTransactionBegin()` and `DetourTransactionCommit()`.

---

## Unhooking (Restoring Original Functions)

Later, you may want to remove hooks:
```cpp
void RemoveHooks()
{
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourDetach(&(PVOID&)OriginalEllipse, HookEllipse);
    DetourDetach(&(PVOID&)OriginalHeapAlloc, HookHeapAlloc);
    DetourTransactionCommit();
}
```

When detach occurs:
- Detours rewrites the original instructions back into the target function.
- Freed any allocated trampoline if possible.

---

## Debugging and Troubleshooting

1. **Crashes after hooking**:
   - Usually due to stack misalignment or incorrect prototypes. On x64, the stack pointer (RSP) must be 16-byte aligned before `call` instructions. Detours ensures alignment for you, but ensure your hook prototypes match exactly.

2. **Ensure correct prototypes**:
   - If original function is `BOOL (WINAPI *Ellipse)(HDC,int,int,int,int)`, ensure hook is `BOOL WINAPI HookEllipse(HDC,int,int,int,int)`.

3. **Check presence of calling conventions**:
   - Most Windows APIs use `WINAPI` (which is `__stdcall` on x86, but on x64 calling convention differences are universal).
   - On x64 Windows, `__stdcall`, `__cdecl`, `__fastcall` are all unified into a single calling convention. Just ensure parameters and return types match exactly.

4. **Logging**:
   - Print statements (ATLTRACE or OutputDebugString) help verify the hook is actually called.
   - If hooking a function called very frequently (like `HeapAlloc`), expect large logs.

5. **Make sure you commit changes**:
   - If you forget `DetourTransactionCommit()`, no hook is applied.

---

## Internals and Advanced Considerations

- Detours works by placing a short jump (5 bytes on x86, often more complex on x64 due to RIP-relative instructions) at the start of the target function.
- On x64, often a 14-byte patch is placed, because short jumps and far jumps differ. Detours uses a special arrangement to ensure all offsets are correct.
- On heavily protected processes (e.g., anti-tampering), hooking might fail. Detours tries normal `WriteProcessMemory`/`VirtualProtect` steps. If these fail, hooking fails.

**Performance Impact**:
- A small overhead occurs because each call now jumps to your hook and possibly out again. 
- Usually negligible for most applications.

**Compatibility**:
- Works on all versions of Windows supported by detours.
- If function is hot-patched by OS or already hooked by something else, you might have conflicts. The order of hooking multiple hooking frameworks matter.

---

## Conclusion

Hooking a function with Detours is straightforward: just start a transaction, attach your hook, and commit. Your hook gets full control over calls to that function. You can alter parameters, call original, log, or skip calls.

**In summary**:
- Identify function & create global original pointer.
- Write a hook function with same signature.
- Begin detour transaction, `DetourAttach`, `DetourTransactionCommit`.
- On usage, your hook intercepts calls. Confirm by logging or changing behavior.