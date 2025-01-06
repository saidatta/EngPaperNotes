https://www.youtube.com/watch?v=j53HxU6BRiw

## Background
- **Handles and the Process Handle Table:**  
  Every Windows process maintains a handle table, which maps handles (user-mode integers) to kernel objects (e.g., mutexes, events, job objects). These handles are reference points to kernel objects stored in system memory.
- **What is a Handle?**  
  A handle is essentially an index (or reference) to a kernel object entry. This entry stores:
  - **Object Address (Pointer):** Points to the kernel object in system space.
  - **Access Mask:** Defines the permissions granted for operations on the object.
  - **Flags:** Such as inheritance settings, protection from close, and audit-on-close features.
  Internally, objects are aligned on a 16-byte boundary, and Windows currently uses a 48-bit address space for kernel addresses. This alignment effectively reduces to requiring about 44 bits for the object address within the handle entry.
![[Screenshot 2024-12-09 at 2.49.12 PM.png]]
---
## Theoretical Limits
- On modern Windows systems, you can often create tens of millions of handles in a single process. In practice, the limit is not a simple fixed number—it's influenced by:
  - Available system resources (e.g., memory).
  - Page pool and non-paged pool availability.
  - The size and type of objects you create.
### Memory Considerations
- **Handle Table Memory:**  
  Each handle entry consumes some kernel memory (in paged pool or non-paged pool depending on the object type and system specifics). A commonly cited figure is around 16 bytes per handle entry. However, this only accounts for the handle entry itself, not the object it points to.
- **Object Memory:**  
  The object that the handle references also consumes memory. For small objects like mutexes, the overhead per object is relatively low. For larger objects (e.g., job objects), the memory usage can grow substantially.
- **Non-Page Pool vs. Page Pool:**  
  Many kernel objects (like job objects, mutexes) come from *non-paged pool* (memory always resides in RAM), which is more limited and always resident in RAM. Creating millions of large objects can rapidly deplete non-paged pool and stress the system.

---
## Practical Experiments

### Example 1: Creating Millions of Mutexes

**Approach:**  
Continuously call `CreateMutex` until it fails. Each `CreateMutex` returns a unique handle and a new kernel mutex object.

**Code Snippet:**

```cpp
#include <windows.h>
#include <stdio.h>

int main() {
    DWORD count = 0;
    for (;;) {
        HANDLE h = CreateMutex(NULL, FALSE, NULL);
        if (!h) {
            printf("Failed at count: %lu, error: %lu\n", count, GetLastError());
            break;
        }
        count++;
    }

    printf("Total handles: %lu\n", count);
    Sleep(INFINITE); // Keep process alive to inspect via Task Manager
    return 0;
}
```

**Observations:**
- On a system with sufficient resources, you might reach ~16 million handles.
- Each handle consumes around 16 bytes in the handle table → ~256 MB total overhead for handles alone.
- The mutex objects themselves consume additional non-paged pool memory. With millions of mutex objects, non-paged pool can grow by multiple gigabytes.
**Task Manager Insight:**
- Add the "Handles" column to Task Manager to see the count climb.
- Add the "Page pool" cost of kernel memory associated with process and "Non-paged pool" columns to observe memory usage growth.
- The exact maximum number varies by system memory and configuration.

---
### Example 2: One Object, Many Handles (DuplicateHandle)
Instead of creating distinct objects, create one object and duplicate its handle many times. This tests the limit of handles without incurring large object creation costs.
**Code Snippet:**
```cpp
#include <windows.h>
#include <stdio.h>

int main() {
    // Create one mutex
    HANDLE hOriginal = CreateMutex(NULL, FALSE, NULL);
    if (!hOriginal) {
        printf("Failed to create mutex: %lu\n", GetLastError());
        return 1;
    }

    DWORD count = 1; // We already have one handle
    for (;;) {
        HANDLE hDup;
        BOOL ok = DuplicateHandle(
            GetCurrentProcess(), hOriginal,
            GetCurrentProcess(), &hDup,
            0, FALSE, DUPLICATE_SAME_ACCESS);
        if (!ok) {
            printf("Failed at count: %lu, error: %lu\n", count, GetLastError());
            break;
        }
        count++;
    }

    printf("Total handles: %lu\n", count);
    Sleep(INFINITE); // Keep alive for inspection
    return 0;
}
```

**Observations:**

- You may again reach millions of handles.
- This time, memory usage in non-paged pool doesn’t skyrocket because you have only one mutex object.
- The handle table still consumes about the same 16 bytes per handle, but there's only one actual object, so object-related memory overhead is minimal.

---

### Example 3: Large Objects (Job Objects)

Job objects are more memory-intensive. Replacing `CreateMutex` with `CreateJobObject`:

```cpp
#include <windows.h>
#include <stdio.h>

int main() {
    DWORD count = 0;
    for (;;) {
        HANDLE h = CreateJobObject(NULL, NULL);
        if (!h) {
            printf("Failed at count: %lu, error: %lu\n", count, GetLastError());
            break;
        }
        count++;
    }

    printf("Total handles: %lu\n", count);
    Sleep(INFINITE);
    return 0;
}
```

**Observations:**

- Job objects are much larger in kernel memory.
- Non-paged pool and overall system commit might explode into the gigabytes.
- The system may struggle, potentially hitting memory limits, and the process might fail earlier.

---

## Impact and Limitations

- **System Stability:**  
  Creating millions of handles and objects can degrade system performance, potentially leading to thrashing, memory pressure, or instability.

- **Practical Limits:**  
  Although you can theoretically create tens of millions of handles, practical engineering constraints (memory, system responsiveness, application needs) typically limit handle usage to much lower numbers.

- **Cleanup Overhead:**  
  Once the process holding millions of handles exits, Windows must free all those objects and handle entries. This can be a lengthy operation. Expect temporary performance lag after closing a handle-intensive process.

---

## Conclusions

- **Maximum Handle Count:**  
  Achieving tens of millions of handles is possible but comes at a high resource cost. The exact maximum is not fixed; it depends on available memory and the size of the objects.

- **Memory Matters:**  
  A single object duplicated many times inflates handle count with minimal object overhead, while creating distinct objects for each handle can consume large amounts of non-paged pool.

- **Practical Advice:**  
  - Avoid unnecessary handle creation loops.
  - Ensure you close handles promptly.
  - Monitor system memory usage if designing for a handle-heavy application.
  - Use resource pools and caching strategies rather than creating vast numbers of kernel objects.

---

## Additional References

- **Windows Internals Book:** Detailed coverage of the object manager, handle tables, and kernel memory usage.
- **Sysinternals Tools:**
  - **Process Explorer & Object Explorer:** View handle counts, object counts, and kernel memory usage.
- **Microsoft Docs:** Kernel object reference, memory management, and handle internals.

---

**In summary**, while Windows can support an enormous number of handles per process, practical limits are imposed by memory availability and system performance. Engineers should be mindful of resource consumption and cleanup overhead when working with large numbers of kernel objects and handles.
```