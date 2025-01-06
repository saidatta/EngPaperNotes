**Audience:**  
These notes target experienced Windows engineers, kernel developers, or advanced reverse-engineers who are continuing from "Drivers and Devices (Part 1)." This second part dives deeper into how user-mode code accesses kernel devices, introduces the concept of file objects in the I/O system, and explores how to open handles to devices—both with and without symbolic links. We'll also discuss the relevant internals and demonstrate code examples.

---
## Introduction
Previously (in Part 1), we established the relationship between driver objects and device objects:
- **Driver Object:** Represents the driver’s code and capabilities.
- **Device Objects:** Represent endpoints or communication channels, often named, that the driver exposes. 

In **Part 2**, we focus on how user-mode applications open device objects to communicate with drivers. 

**Key Points:**

- We **open handles to devices**, not drivers.
- Opening a handle to a device returns a **file object** in kernel mode, bridging user and kernel spaces.
- Symbolic links and native APIs offer multiple methods to access a device.

---

## File Objects

**What is a File Object?**  
When a user-mode application calls `CreateFile` on something like `\\.\DeviceName`, the I/O manager creates a **file object** in kernel mode. This file object:

- Represents the open instance of that device to the calling process.
- Contains pointers to the device object and other state (like access mode, position, synchronization mode).
- Ties user-mode handles to kernel-mode objects.

**Flow:**

1. User calls `CreateFile` with a path like `\\.\Beep`.
2. The I/O manager resolves `\\.\Beep` to a device object (possibly via a symbolic link).
3. The I/O manager creates a file object for that open instance.
4. The user-mode function returns a handle, referencing that file object.
5. Subsequent `ReadFile`, `WriteFile`, `DeviceIoControl` use the handle → file object → device object → driver dispatch routines.

---

## Accessing Devices from User Mode

### Using Symbolic Links

**Most Common Case:**
- Drivers often create a symbolic link under `\??` (or `\\.\` in user-mode namespace) to their device, making it accessible like `\\.\MyDevice`.
- `CreateFile(L"\\\\.\\MyDevice", ...)` opens a handle to that device.
- Example (from code in previous video):
    ```c++
    HANDLE hDevice = CreateFile(
        L"\\\\.\\ProcExp152", // symbolic link to device 
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING, 0, NULL
    );
    if (hDevice == INVALID_HANDLE_VALUE) {
        wprintf(L"Error: %lu\n", GetLastError());
    } else {
        wprintf(L"Got handle 0x%p\n", hDevice);
        // Use DeviceIoControl, etc.
        CloseHandle(hDevice);
    }
    ```

**Internals:**
- The I/O manager looks up `\??\ProcExp152` in the object manager namespace.
- Finds a symbolic link to `\Device\ProcExp152`.
- Resolves to a device object associated with some driver.
- Creates file object → returns handle.

### Using GlobalRoot Path

**If No Symbolic Link Exists:**
- You can open a device by using a "rooted" path like `\\?\GlobalRoot\Device\Beep`.
- `GlobalRoot` (or `\\?\GlobalRoot`) points to the root of the object manager namespace.  
- Example:
    ```c++
    HANDLE hBeep = CreateFile(
        L"\\\\?\\GlobalRoot\\Device\\Beep",
        GENERIC_WRITE,
        FILE_SHARE_READ|FILE_SHARE_WRITE, 
        NULL, OPEN_EXISTING, 0, NULL
    );
    ```
- This works even if there's no user-friendly symbolic link. 

**Internals:**
- This bypasses the symbolic link layer, directly specifying the object manager path.
- The I/O manager parses the entire namespace path: `Device\Beep` to find the device object.

### Using Native APIs (NtOpenFile / NtCreateFile or Zw calls)

**For Advanced/Undocumented Scenarios:**
- `NtOpenFile` (or the `Ex/Nt` variant) can open device objects by specifying a full object manager path as a `UNICODE_STRING`.
- This approach requires native API usage and is typically not recommended unless you know exactly what you are doing.

**Example:**
```c
#include <winternl.h>

// Prepare UNICODE_STRING and OBJECT_ATTRIBUTES for "\Device\Beep"
UNICODE_STRING devName;
RtlInitUnicodeString(&devName, L"\\Device\\Beep");

OBJECT_ATTRIBUTES attr;
InitializeObjectAttributes(&attr, &devName, OBJ_CASE_INSENSITIVE, NULL, NULL);

IO_STATUS_BLOCK ioStatus;
HANDLE hBeepNative;
NTSTATUS status = NtOpenFile(
    &hBeepNative,
    GENERIC_WRITE,
    &attr,
    &ioStatus,
    FILE_SHARE_READ | FILE_SHARE_WRITE,
    FILE_NON_DIRECTORY_FILE
);

if (NT_SUCCESS(status)) {
    // hBeepNative now is a handle to the beep device via native API
    // Use NtDeviceIoControlFile, etc.
    NtClose(hBeepNative);
}
```

---

## Interacting with the Device Once Opened

Once you have a handle, you can:

- `ReadFile`/`WriteFile`: If the device supports these IRP major functions.
- `DeviceIoControl`: The most common way to send custom commands to devices. Usually, drivers define IOCTL codes and associated input/output data structures.
- `SetFilePointerEx`, `SetEndOfFile`, etc.: If the device supports these operations (many do not).

**Example (Beep Device):**
- The beep device accepts an IOCTL to set frequency and duration.
- We define a structure:
    ```c
    typedef struct _BEEP_SET_PARAMETERS {
        ULONG Frequency;
        ULONG Duration; // in ms
    } BEEP_SET_PARAMETERS;
    ```
- The IOCTL is defined in `ntddbeep.h`.
    ```c
    #define IOCTL_BEEP_SET CTL_CODE(FILE_DEVICE_BEEP, 0, METHOD_BUFFERED, FILE_ANY_ACCESS)
    ```
- Code:
    ```c++
    BEEP_SET_PARAMETERS note = { 1000, 2000 }; // 1KHz for 2s
    DWORD bytesReturned;
    DeviceIoControl(
        hBeep,
        IOCTL_BEEP_SET,
        &note, sizeof(note),
        NULL, 0,
        &bytesReturned,
        NULL
    );
    ```

This will play a beep sound asynchronously if allowed.

---

## Internals: File Object Creation

- On `CreateFile`:
  1. User mode calls `CreateFile` → NTDLL → `NtCreateFile`.
  2. Object manager resolves name → finds device object.
  3. I/O manager calls the driver’s `IRP_MJ_CREATE` dispatch routine.
  4. If successful, I/O manager creates a file object (a kernel object of type `File`), linking it to the device object.
  5. A handle referencing this file object is returned to the caller.

- The file object holds state like access mode, synchronization flags, pointer to `DeviceObject`, and a `FsContext` pointer used by file system drivers to store context.

**Deeper:**
- You can use the kernel debugger to look at file objects (`!fileobj`), see the device object they're linked to, and the driver that controls them.

---

## Summary

- **File objects** are how user-mode interacts with devices. No direct handles to drivers, always devices.
- Most common method: `CreateFile("\\\\.\\DeviceName")` using a symbolic link.
- If no symbolic link: Use `GlobalRoot` path or native APIs (`NtOpenFile`) to directly access the device namespace path.
- Once opened, you have a handle that references a file object in kernel mode that ultimately references a device object.
- This handle can be used with `ReadFile`, `WriteFile`, or `DeviceIoControl` to talk to the driver through the device object.

**In short:**
- Drivers expose devices.
- User-mode opens handles to device objects (through various paths).
- File objects bridge user handles and device objects.
- Armed with this knowledge, you can navigate complex driver/device ecosystems and debug them effectively.