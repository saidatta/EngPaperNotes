https://www.youtube.com/watch?v=SmFi1cj6gMg
**Audience:**  
These notes target experienced Windows engineers and low-level developers who want to deeply understand a more covert technique to inject a DLL into a target process using shell code. We will comprehensively cover the logic, code, and the internal mechanics of this approach, including potential pitfalls and stealth techniques.

---

## Introduction

DLL injection has several classic methods:
- **CreateRemoteThread**: Very detectable; creates a new remote thread directly.
- **QueueUserAPC**: Stealthier, but requires the target thread to enter an alertable wait state.

**Shell Code Injection** offers another avenue:
- Write arbitrary machine code (shell code) into the target process.
- Hijack an existing thread (no new threads created).
- Set that thread’s instruction pointer to your shell code.
  
**Advantage:**
- Can be more stealthy and bypass some detection heuristics.
- Does not rely on the target process calling standard APIs in a normal manner.

**Challenge:**
- Shell code has no direct API calls (like `LoadLibrary` or `GetProcAddress`), at least not by default.
- Due to ASLR and other modern security features, you cannot simply guess DLL function addresses.
- Your shell code must be self-contained and must not rely on known function addresses unless your injector passes them in.

---

## Concepts and Steps

1. **Identify Target Thread**:  
   Find a suitable thread within the target process. Any thread works, but a thread with known or stable conditions is ideal.

2. **Suspend Thread & Get Context**:
   - Use `SuspendThread` to stop execution.
   - `GetThreadContext` to retrieve the thread’s current register state (especially `RIP`, the instruction pointer).

3. **Allocate Memory in Target Process**:
   - `VirtualAllocEx` to allocate memory (preferably `PAGE_READWRITE` first).
   - Write both the DLL path and shell code into this allocated region.

4. **Patch Shell Code with Runtime Values**:
   - Insert the address of `LoadLibraryA` or desired function into the shell code bytes.
   - Insert the address of the DLL path (just allocated) into the shell code’s call parameter.
   - Insert the original `RIP` of the thread so shell code can jump back after completion.

5. **Change Memory Protection**:
   - After writing shell code, use `VirtualProtectEx` to change the memory to `PAGE_EXECUTE_READ`.
   - Avoid `PAGE_EXECUTE_READWRITE` to reduce detection risk.

6. **Set the New Thread Context**:
   - Update `RIP` in the `CONTEXT` structure to point to shell code’s start.
   - `SetThreadContext` to apply the changes.

7. **Resume the Thread**:
   - `ResumeThread` to let the thread run shell code.
   - Shell code executes `LoadLibraryA`, loading your DLL into the target process.
   - On completion, shell code restores state and jumps back to original `RIP`.

---

## Shell Code Internals

**No Direct API Calls:**
- Because addresses are randomized, the injector must supply the correct addresses:
  - For `LoadLibraryA`: The injector can resolve it using `GetProcAddress(GetModuleHandle("kernel32.dll"), "LoadLibraryA")`.
- Shell code must:
  - Move the `LoadLibraryA` address into a register (e.g., `R11`).
  - Move the DLL path address into `RCX` (1st param of the Windows x64 calling convention).
  - Allocate stack space for calling conventions.
  - `CALL R11` to invoke `LoadLibraryA(RCX)`.
  - After `LoadLibraryA` returns, restore registers and jump back to original code.

**Typical x64 Calling Convention:**
- RCX: 1st argument
- RDX: 2nd argument
- R8: 3rd
- R9: 4th
- Additional arguments on stack.
- Need 32 bytes of shadow space on stack for the called function.

**Large Page, TLB, etc.**
- Not directly relevant here, but understanding VA translation helps if debugging memory issues.

---

## Code Example: High-Level Steps

### Pseudocode:

```c++
int main(int argc, char** argv) {
    int pid = atoi(argv[1]);
    char* dllPath = argv[2];

    HANDLE hProcess = OpenProcess(PROCESS_VM_OPERATION | PROCESS_VM_WRITE | ... , pid);
    if(!hProcess) return GetLastError();

    DWORD tid = GetThreadInProcess(pid);
    if(!tid) return -1;

    HANDLE hThread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, tid);
    if(!hThread) return GetLastError();

    // Suspend the target thread
    SuspendThread(hThread);

    // Prepare memory in target
    PVOID region = VirtualAllocEx(hProcess, NULL, 0x1000, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);

    // Write DLL path at region start
    WriteProcessMemory(hProcess, region, dllPath, strlen(dllPath)+1, NULL);

    // Our shell code offset at region+0x800 (2KB offset)
    uint8_t shellCode[] = { ... }; // Shell code from assembler
    // Patch loadLibrary address, dll path address, original RIP

    // VirtualProtectEx to PAGE_EXECUTE_READ
    DWORD oldProt;
    VirtualProtectEx(hProcess, region, 0x1000, PAGE_EXECUTE_READ, &oldProt);

    // Get Context
    CONTEXT ctx = {0};
    ctx.ContextFlags = CONTEXT_CONTROL;
    GetThreadContext(hThread, &ctx);

    // Patch shell code with ctx.Rip
    // Set ctx.Rip to shellcode start

    SetThreadContext(hThread, &ctx);

    // Resume thread to run shell code
    ResumeThread(hThread);

    CloseHandle(hThread);
    CloseHandle(hProcess);
}
```

---

## Detailed Steps with Formatting

- **Step 1: Open Target Process**
  ```c++
  HANDLE hProcess = OpenProcess(PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_QUERY_INFORMATION, FALSE, pid);
  ```

- **Step 2: Find a Thread**
  ```c++
  DWORD tid = GetThreadInProcess(pid); // enumerates threads, returns first thread ID
  HANDLE hThread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, tid);
  ```

- **Step 3: Suspend and Get Context**
  ```c++
  SuspendThread(hThread);
  CONTEXT ctx = {0};
  ctx.ContextFlags = CONTEXT_CONTROL;
  GetThreadContext(hThread, &ctx);
  ```

- **Step 4: Allocate Memory in Target**
  ```c++
  PVOID base = VirtualAllocEx(hProcess, NULL, 0x1000, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
  WriteProcessMemory(hProcess, base, dllPath, strlen(dllPath)+1, NULL);
  ```

- **Step 5: Shell Code Assembly**
  - Prepared external using an assembler tool.
  - Shell code placeholders (`AA`, `BB`, `CC`) replaced with real addresses.
  ```c++
// Example pseudo code for patching:
memcpy(shellCode+3, &LoadLibraryAddr, 8);
memcpy(shellCode+0x13, &base, 8);
memcpy(shellCode+0x23, &ctx.Rip, 8);
  ```

- **Step 6: Write Shell Code and Change Protections**
  ```c++
  WriteProcessMemory(hProcess, (char*)base+0x800, shellCode, sizeof(shellCode), NULL);
  DWORD oldProt;
  VirtualProtectEx(hProcess, base, 0x1000, PAGE_EXECUTE_READ, &oldProt);
  ```

- **Step 7: Set Thread Context and Resume**
  ```c++
  ctx.Rip = (ULONG64)base+0x800; // shell code start
  SetThreadContext(hThread, &ctx);
  ResumeThread(hThread);
  ```

---

## Internals & Potential Pitfalls

- **ASLR & GetProcAddress**:  
  Real shell code might have to dynamically find `LoadLibraryA` by parsing the PEB, walking the module list, and scanning export tables, all in assembly. We simplified by passing addresses from the injector.
  
- **Stack Stability**:  
  Shell code must follow x64 calling conventions. Provide shadow space (32 bytes) before calling the API.

- **No RWX Memory**:  
  Allocating as `PAGE_READWRITE` then `VirtualProtectEx` to `PAGE_EXECUTE_READ` after writing reduces detection by AV/EDR tools.

- **Thread Context Restoration**:  
  The shell code calls `LoadLibraryA`. After returning, it must restore registers and jump back to the original `RIP`. If not done, the target thread might crash or behave incorrectly.

- **Error Handling**:
  Real code should handle errors gracefully. For a proof-of-concept, minimal checks are shown.

---

## Result

The final result:
- A thread in the target process wakes up at the shell code, calls `LoadLibraryA` with the given DLL path.
- DLL is loaded into the target process.
- Shell code then restores state and continues execution as if nothing happened.

This technique is stealthier:
- No new threads created.
- No alertable waits required.
- API calls minimized or replaced with manual methods.

**Caveat**:  
Difficult to write and maintain shell code. Complex scenario if you must find addresses dynamically. Proper testing and debugging are crucial.

---

## Conclusion

Shell code-based DLL injection on x64:
- Demonstrates advanced injection techniques beyond standard APIs.
- Requires deep understanding of memory, calling conventions, and CPU state.
- Offers higher stealth at the cost of complexity.

For those who master this technique, it’s a powerful tool in the arsenal of low-level Windows manipulation.
```