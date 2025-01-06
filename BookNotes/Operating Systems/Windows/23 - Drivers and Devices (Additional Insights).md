In the previous notes (Part 2), we concluded with a summary of how user-mode code opens handles to device objects and communicates with drivers via file objects and IRPs. Let's delve deeper into the journey of a user-mode request (such as a `ReadFile` or `DeviceIoControl`) through the kernel, focusing on IRPs, stack locations, and the dispatch routines in drivers.

---
## The Life of an I/O Request

### 1. User Request Origin

When a user-mode application calls a high-level I/O function (e.g., `ReadFile(hDevice, ...)`):

1. **User-mode to Kernel Transition:**  
   The call transitions into kernel mode via `ntdll.dll` and the system call interface.
   
2. **Identification of File Object and Device:**  
   The kernel obtains the file object associated with `hDevice`.  
   - This file object points to a device object.
   - The kernel knows which driver stack is responsible for handling this request based on the device object.

### 2. IRP Creation and Major Function Codes

3. **IRP Creation:**  
   The I/O manager allocates an IRP (I/O Request Packet). The IRP contains:
   - MajorFunction code: e.g., `IRP_MJ_READ`, `IRP_MJ_WRITE`, `IRP_MJ_DEVICE_CONTROL`.
   - MinorFunction (if any).
   - Pointer to the file object.
   - Buffers for input/output data.
   
4. **I/O Stack Locations:**  
   Each driver in the stack gets its own "stack location" in the IRP, specifying what it needs to do. For example, a storage stack might have a file system driver, a volume manager, a disk driver, and a port driver. The IRP is passed down layer by layer.

**Code Sketch (Kernel Perspective):**

```c
// Pseudocode from I/O manager's perspective
PIRP Irp = IoAllocateIrp(NumberOfStackLocations, FALSE);
Irp->MajorFunction = IRP_MJ_READ; // for a read request
Irp->FileObject = FileObject; // from user handle
Irp->AssociatedIrp.SystemBuffer = UserBuffer; 
```

### 3. Dispatching the IRP

5. **Top-Level Driver Dispatch:**  
   The IRP is first given to the top-most driver in the device stack.  
   - The I/O manager calls `DriverObject->MajorFunction[IRP_MJ_READ]` or whatever major function is relevant.

6. **Driver Processing:**  
   The top-level driver (e.g., a file system or filter driver) looks at the IRP, does what it can:
   - If it can satisfy the request immediately, it might complete the IRP right there.
   - Otherwise, it may modify the IRP’s parameters, mark it pending, and pass it down to the next driver by calling `IoCallDriver` with the IRP.

7. **Passing Down the Stack:**  
   `IoCallDriver` moves the IRP to the next lower driver stack location. That driver gets the IRP via its dispatch routine, and the process repeats until the lowest-level driver that can actually perform the I/O is reached.

**Example Scenario:**
- A `ReadFile` on a handle to a disk file:
  - IRP first hits the file system driver (e.g., NTFS).
  - NTFS might translate the read request into volume/disk operations and `IoCallDriver` to send the IRP to the volume manager.
  - The volume manager may pass it further down to a disk class driver, which may pass it down to a port driver, and eventually a miniport or bus driver that interacts with the hardware.

### 4. IRP Completion

8. **Completing the IRP:**
   Eventually, the lowest-level driver either finishes the request or fails it:
   - It sets `Irp->IoStatus.Status` and `Irp->IoStatus.Information`.
   - Calls `IoCompleteRequest(Irp, IO_NO_INCREMENT)` to notify the I/O manager that the IRP is done.

9. **Propagating Back Up:**
   As the IRP unwinds back up the stack, upper drivers can see the completion results, possibly adjust them, or log information.
   
10. **Returning to User Mode:**
    The I/O manager finally returns control to user mode with the result of the operation (e.g., number of bytes read, error codes, etc.).

---

## Internals and Considerations

### Layered Drivers and Filters

- Complex device stacks often involve multiple drivers. Each driver views a portion of the IRP stack called the "I/O Stack Location" (IO_STACK_LOCATION).
- Each stack location includes parameters relevant to that driver’s layer, like offsets for a read or particular IOCTL codes.

### Using Tools to Inspect IRPs

- **WinDbg Commands:**
  - `!irp <Address>`: Inspect an IRP in detail, see major function, stack locations, associated file object, current driver, etc.
  - `!devobj <Address>`: Shows device object details (we covered this in Part 1).
  - `!fileobj <Address>`: Show details about a file object (handles, device pointer).

**Example (in WinDbg):**
```none
!irp ffffa68699c0b010
```
Displays IRP major function, minor function, and layered drivers it passed through.

### Performance and Memory

- IRPs are allocated from non-paged pool (or special lookaside lists).  
- Minimizing unnecessary IRP allocations is key for performance in I/O-heavy drivers.

### Security and Access Control

- When a user calls `CreateFile` on a device, the system enforces security checks:
  - DACLs on the device object.
  - The requestor’s credentials.
- If the driver fails to handle IRP_MJ_CREATE properly, unauthorized users might open the device handle.

### IOCTLs (DeviceIoControl)

- `DeviceIoControl` user-mode API triggers `IRP_MJ_DEVICE_CONTROL` requests with an IOCTL code and input/output buffers.
- Drivers define their own IOCTL codes, often in a header shared with user-mode clients, ensuring both sides agree on the "protocol."

**IOCTL Example:**
```c++
// In user mode:
MY_IOCTL_PARAMS params = { /* ... */ };
DWORD bytesReturned;
if (!DeviceIoControl(hDevice, MY_IOCTL_CODE, &params, sizeof(params), NULL, 0, &bytesReturned, NULL)) {
    // Error handling
}
```
Driver sees `IRP_MJ_DEVICE_CONTROL` with `Irp->Parameters.DeviceIoControl.IoControlCode = MY_IOCTL_CODE`.

---

## Putting It All Together

**Scenario: Reading from a Virtual Device:**

1. User mode:
   ```c++
   HANDLE h = CreateFile(L"\\\\.\\MyVirtualDevice", GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
   // h is now a handle referencing a file object which references the device

   BYTE buffer[100];
   DWORD bytesRead;
   BOOL ok = ReadFile(h, buffer, sizeof(buffer), &bytesRead, NULL);
   ```

2. Kernel steps:
   - `ReadFile` → `NTDLL` → `NtReadFile` → I/O manager → creates IRP with `IRP_MJ_READ`.
   - IRP flows down the driver stack.  
   - Final driver handles the read (maybe from a buffer or memory it manages) and calls `IoCompleteRequest`.
   - Control returns to user-mode with `bytesRead` indicating how many bytes received.

3. The complexity hidden:
   - Each layer possibly modifies the IRP stack location, may pend the IRP if data not ready, and re-queues completion.
   - Eventually, IRP completion sets `Irp->IoStatus.Status = STATUS_SUCCESS; Irp->IoStatus.Information = actual length`.

---

## Summary

- **File Objects:** Are the kernel representation of a user-mode handle to a device object.
- **IRPs:** Convey all I/O requests in kernel mode, traveling through driver stacks.
- **User-Mode Access:** Typically through `CreateFile` + symbolic link (`\\.\MyDevice`), or using `GlobalRoot` or native APIs for more direct paths.
- **Dispatch Routines and IRP Major Functions:** Drivers implement certain IRP major functions (e.g., `IRP_MJ_READ`, `IRP_MJ_WRITE`, `IRP_MJ_DEVICE_CONTROL`) to handle operations.
- **DeviceIoControl:** Sends custom commands (IOCTLs) to devices, letting drivers implement arbitrary operations.

With this knowledge, you understand how applications interact with drivers at runtime, how IRPs flow through driver stacks, and how to leverage tools and native APIs for special cases. This forms a comprehensive view of the Windows driver I/O model.
```