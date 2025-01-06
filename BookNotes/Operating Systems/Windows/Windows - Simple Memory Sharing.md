https://www.youtube.com/watch?v=KMq-5kGufYI
## Overview
By default, each Windows process has its own isolated virtual address space. Even if multiple processes run from the same executable, their global variables do not affect each other. However, in certain scenarios, it’s beneficial to have multiple instances of the same program share a few variables in memory without creating separate shared memory files or using other IPC mechanisms.
**Key Concept:**  
- You can mark a data section in a PE (Portable Executable) image as **shared**, causing all processes that load that image to share the variables in that section.
- This approach is simpler than memory-mapped files for small amounts of shared data. However, it’s less flexible and suitable mainly for fixed-size global data.

**Note:** Synchronization is essential if multiple processes write to the shared data. Without synchronization (e.g., a mutex), data corruption or inconsistent states may occur.

---
## Prerequisites and Limitations
- This technique works if all processes use the same executable or DLL image.
- The data to be shared must be in a custom named section with special attributes set at link time.
- Shared data is typically placed in a global variable, initialized at compile time (i.e., you must provide an initializer for the variable).
- Changes to shared variables are immediately visible to all processes that have loaded the image.

---

## Step-by-Step Implementation

1. **Define a Custom Data Section:**
   Use `#pragma data_seg` to define a section and place your global variables there:
   ```cpp
   #pragma data_seg(".shared")
   int g_sharedCounter = 0;  // Must have an initializer
   #pragma data_seg()
   ```

   - The `.shared` name is arbitrary but must be ≤8 characters.
   - Initialize your variable; uninitialized variables might end up in a different section.

2. **Set the Section Attributes:**
   Instruct the linker to mark this section as readable, writable, and shared. Add a comment pragma to pass a linker directive:
   ```cpp
   #pragma comment(linker, "/SECTION:.shared,RWS")
   ```
   
   **R** = Read, **W** = Write, **S** = Shared.  
   The `S` is what ensures multiple processes share the same physical memory for this section.

3. **Use the Shared Variables:**
   In your code:
   ```cpp
   #include <windows.h>
   #include <stdio.h>

   // Shared variable definition above

   int main() {
       for (;;) {
           printf("g_sharedCounter = %d\n", g_sharedCounter);
           g_sharedCounter++;
           Sleep(1000);  // 1-second pause
       }
       return 0;
   }
   ```

   Start one instance of the program, and `g_sharedCounter` increments every second. Start a second instance, and now each instance increments the same global counter. The combined effect will be that the counter increases by 2 each second if two processes are running, by 3 for three processes, and so on.

---

## Example Behavior

- **Single Instance:**
  Starts at zero and increments by 1:  
  ```
  g_sharedCounter = 0
  g_sharedCounter = 1
  g_sharedCounter = 2
  ...
  ```

- **Second Instance Launched:**
  Both processes share the same `g_sharedCounter`. Each increments it by 1 every second. Now you’ll see values jumping by 2 each cycle because both processes increment the single shared value:
  ```
  Process 1: sees counter incrementing by 2 steps each time it prints.
  Process 2: similarly sees the same shared increments.
  ```

- **Third Instance:**
  With three instances, the counter increments by 3 each cycle.

---

## Verifying the Section

You can inspect the PE file using tools like **PE Viewer** or **Dumpbin** to see the `.shared` section:

- The `.shared` section will appear as a custom section in the executable.
- The `RWS` attributes confirm it’s readable, writable, and shared.

**Example (using dumpbin):**
```bash
dumpbin /headers SimpleSharing.exe
```
You’ll find a `.shared` section with the `IMAGE_SCN_MEM_SHARED` attribute set.

---

## Advantages and Drawbacks

**Advantages:**
- Simple for small amounts of data.
- No need to explicitly create memory-mapped files or other IPC structures.
- Ideal for quick global state sharing between multiple instances of the same program.

**Drawbacks:**
- Inflexible: Data is fixed at build time.
- Limited to the executables and DLLs that have the shared sections defined.
- No built-in synchronization: You must manually handle concurrent writes using mutexes or critical sections.
- Not suitable for large or dynamically sized data.

---

## Synchronization Example

If multiple processes write to the shared variable, consider using a named mutex:

```cpp
HANDLE hMutex = CreateMutexA(NULL, FALSE, "Global\\MySharedMutex");
while (true) {
    WaitForSingleObject(hMutex, INFINITE);
    g_sharedCounter++;
    ReleaseMutex(hMutex);

    printf("g_sharedCounter = %d\n", g_sharedCounter);
    Sleep(1000);
}
```

This ensures that increments are atomic and no data corruption occurs.

---

## Summary

By using a custom `.shared` section and a linker directive, you can create global variables that are shared across all processes executing the same image. This method offers a very simple form of inter-process shared memory for small, static data, but requires careful synchronization and should be used judiciously.
```