https://www.youtube.com/watch?v=sSZ8jnpUCi0&t=559s

**Audience:**  
These notes target experienced Windows engineers familiar with kernel concepts, driver development, and internal kernel debugging. This is the first part of a two-part series on understanding drivers, devices, and their relationships in the Windows kernel.

---
## Introduction
In Windows, **drivers** and **device objects** are often mentioned together, but they represent different concepts:
- **Driver:** Think of a driver as a piece of code (logic) that knows how to handle particular types of devices or functionalities.
- **Device Object:** Represents a specific device (physical or virtual) as a kernel object. Each device object corresponds to an endpoint through which I/O operations (like read, write, control requests) can be performed.
**Key Distinction:**
- A single **driver** can manage multiple **device objects**.
- User-mode applications open handles to **devices**, not to drivers directly.
---
## Driver Objects vs. Device Objects
1. **Driver Object:**
   - Created by the kernel when the driver is loaded.
   - Identifies the driver’s code, entry points, and dispatch routines.
   - Example: A keyboard class driver (`kbdclass`) might handle multiple keyboards. There’s a single driver object representing that driver.

2. **Device Object:**
   - Created by the driver when a new device (physical or virtual) appears.
   - Each device object represents one communication endpoint.
   - Users open a handle to a device (e.g., `\\.\Beep` device, `\\.\C:` representing a volume controlled by `ntfs.sys` driver).
   - The OS never directly uses a driver object for I/O; it always uses device objects.

### Multiple Devices per Driver

- Example: A serial port driver could manage `COM1`, `COM2`, `COM3`, and `COM4`.
- One driver object (e.g., `serial.sys`) and multiple device objects (`\Device\Serial0`, `\Device\Serial1`, etc.).

---

## Relationship and Data Structures

**Pointer Links:**
- **Driver object** has a pointer to the first device object it controls (`DriverObject->DeviceObject`).
- Each **device object** points back to its driver (`DeviceObject->DriverObject`) and may have a `NextDevice` pointer, forming a singly-linked list of devices managed by the same driver.

**A Typical Scenario:**
```
          +--------------+
          | DriverObject |
          +--------------+
               |
           DeviceObject -> DeviceObject -> ... -> DeviceObject (last)
```

**APIs and Internals:**
- Drivers create device objects typically in `DriverEntry` or in `AddDevice` routines for Plug and Play drivers using `IoCreateDevice`.
- Once created, these device objects are arranged in a linked list off the driver object.
---
## Layering of Devices
- Drivers can “stack” their device objects, creating a layered or filter stack.
- For example, a keyboard class driver sits on top of a keyboard port driver. Both create device objects forming a layered chain:
  - User I/O → Top-Level Device Object (class driver) → Lower Device Object (port driver) → actual hardware interface.

**Note:** While we say "layered drivers," it’s actually their **device objects** that form the layered stack.

---
## Tools and Debugging
### WOBJ (WOBj) Tool (CIS Internals)
- `wobj.exe` can browse the object manager namespace.
- **Driver objects** reside in `\Driver` directory.
- **Device objects** often in `\Device` directory.
  
**Example:**  
- `\Driver\Beep` corresponds to the beep driver object.
- `\Device\Beep` corresponds to the beep device object that user-mode can open (e.g., `CreateFile("\\.\Beep", ...)`).

### Kernel Debugger (WinDbg)
- Use `!drvobj <Name> [Flags]` to inspect a driver object and its device objects.
  
**Examples:**
```none
!drvobj beep f
```
- Shows the `beep` driver’s details:
  - One device object
  - Its dispatch routines (like `BeepOpen`, `BeepClose`)
  - Supported operations (create, close, device control)  
  Unimplemented operations might point to internal "Invalid device request" handlers.

```none
!drvobj kbdclass f
```
- The keyboard class driver shows multiple device objects (for multiple keyboards).
- Also shows IRP major function array and which operations are implemented.

### Data Structures in Debugger

- `DT` commands on driver object and device object structures:
  ```none
  dt nt!_DRIVER_OBJECT <Address>
  dt nt!_DEVICE_OBJECT <Address>
  ```
- Inspect fields like `DriverName`, `MajorFunction[]` array, `DeviceObject`, `NextDevice`.

**Real Example Output (Beep driver):**
- Driver name: `\Driver\Beep`
- `MajorFunction[IRP_MJ_CREATE] = BeepOpen`
- `MajorFunction[IRP_MJ_CLOSE] = BeepClose`
- All others point to `IopInvalidDeviceRequest` meaning unsupported.

**Real Example Output (Keyboard Class driver):**
- Multiple device objects listed.
- More IRP major functions implemented: read, device control, cleanup.
- Possibly layered devices visible (top-level and lower-level filter devices).

---

## Code Example (Pseudo-code)

**DriverEntry** snippet:
```c
NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath
    )
{
    // Set dispatch routines
    DriverObject->MajorFunction[IRP_MJ_CREATE] = MyCreate;
    DriverObject->MajorFunction[IRP_MJ_CLOSE] = MyClose;
    DriverObject->DriverUnload = MyUnload;

    // Create a device object
    PDEVICE_OBJECT DeviceObject = NULL;
    NTSTATUS status = IoCreateDevice(
        DriverObject,
        0,
        &MyDeviceName, // e.g., L"\\Device\\MyDevice"
        FILE_DEVICE_UNKNOWN,
        0,
        FALSE,
        &DeviceObject
    );
    if (!NT_SUCCESS(status)) return status;

    // DeviceObject->DriverObject = DriverObject automatically set by IoCreateDevice
    // Now DeviceObject is linked via DriverObject->DeviceObject
    // and can be opened by user mode if we create a symbolic link.

    return STATUS_SUCCESS;
}
```

**User-mode (Open Device):**
```c
HANDLE hDevice = CreateFile(L"\\\\.\\MyDevice", 
                            GENERIC_READ | GENERIC_WRITE,
                            0, NULL, OPEN_EXISTING, 0, NULL);

// This opens a handle to the device, not to the driver.
```

---

## Summary

- A **driver object** represents the driver code itself. There’s typically one driver object per driver binary.
- **Device objects** represent devices (physical or virtual) that the driver manages. One driver, many devices.
- Users interact with devices, not drivers. All system I/O requests flow through these device endpoints.
- Tools like `wobj`, `!drvobj` help visualize the relationship.
- Complex driver stacks form layers of device objects, each driver providing a piece of functionality.

In short, understanding this relationship is crucial for driver development, debugging, and analyzing system behavior at the kernel level. This sets the stage for the next part, where we’ll discuss how user-mode requests reach these devices through IRPs and dispatch routines.
```