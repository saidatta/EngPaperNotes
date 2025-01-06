https://www.youtube.com/watch?v=qAmwNuRbyeI
## Overview

**What are Windows Containers (Server Silos)?**  
A **server silo**, often referred to as a Windows container, is a specialized form of a job object that provides isolation at the OS level, similar in concept to Linux containers. Rather than spinning up a full VM, containers share the host kernel but isolate processes’ views of the system’s resources:

- **Namespaces & Isolation:** Each container (silo) provides a private namespace for objects like files, registry keys, named kernel objects, and even a private copy of certain system directories.
- **Resource & Security Isolation:** Containers limit what processes inside the silo can see: a restricted set of processes, a private registry hive, and a restricted filesystem view.
- **Lighter-Weight than VMs:** Processes in a silo share the host OS kernel, reducing overhead compared to full virtualization.

Under the hood, Windows containers build upon **job objects**, extending them into “silos” that add the necessary isolation layers. This is exposed in Windows 10+ and Windows Server.

---
## Relation to Job Objects
1. **Job as a Foundation:**  
   Every silo starts as a job object. A special flag or internal call “converts” this job into a silo, enabling additional isolation beyond normal job limits.
2. **Silo Extensions:**  
   Unlike normal jobs that primarily manage resource limits (CPU, memory, I/O), a silo adds:
   - **Private FileSystem & Registry Roots:** Processes see a different C:\ drive, a minimal OS image, and a container-specific registry hive.
   - **Private Namespaces:** Each silo has a unique Object Manager namespace directory (e.g., `\Sessions\...\Silos\...`).
   - **Session & Security:** Containers run in their own isolated session, further distancing them from host processes.
3. **Multiple Processes within a Silo:**  
   All processes within the container are associated with the silo job. Child processes inherit silo membership, just as with normal jobs.
---
## Example: Docker with Windows Containers
**Docker Workflow:**
1. **Switch to Windows Containers:**  
   Docker Desktop supports Windows containers mode. When active, `docker run --isolation=process` launches a Windows container.
2. **Launching a Container:**  
   ```bash
   docker run --isolation=process -it mcr.microsoft.com/windows/nanoserver:latest cmd
   ```
   Inside the container:
   - The `C:\` drive is minimal and isolated.
   - The registry, process list, and namespaces are restricted.
3. **Host View vs. Container View:**
   - Host sees many processes, multiple sessions, and a large set of system objects.
   - Container sees a limited set of processes, often just a handful of system services and the launched application, and a stripped-down filesystem.

---
## Internal Mechanics and NT API Details
1. **Silo Creation (Native API):**  
   A silo is created by a specialized call (undocumented for general use) within the kernel that extends a normal job object into a silo. Internally:
   - `NtCreateJobObject` creates a job.
   - A specialized call or parameter (used by container orchestration tools) transforms this job into a silo, initializing silo-specific structures in the kernel’s `EJOB` and associated `ESILO` structures.
2. **Private Namespaces & Object Manager:**
   - The Object Manager is extended to handle silo-specific namespaces. When a silo is created, the kernel creates a private object directory structure for that silo.
   - These namespaces ensure that named kernel objects (events, mutexes, semaphores) and symbolic links do not clash with the host or other containers.
3. **Registry Isolation:**
   - The registry is virtualized for silos. Under the hood, the kernel and user-mode infrastructure (the “registry silo code”) creates per-silo registry hives.
   - APIs like `RegOpenKeyEx` inside a silo resolve keys against a silo-specific root, ensuring processes see a containerized registry view.
4. **Filesystems & Volumes:**
   - For process isolation, Windows containers mount specific volumes for the silo’s root filesystem. The kernel’s volume mounting and filter manager create ephemeral volumes for the container.
   - Tools like `fltmc volumes` on the host show these additional volumes appear/disappear as containers start/stop.
5. **Process and Session Isolation:**
   - Each silo typically runs in its own Session (e.g., Session 3 for a new container), separate from the host’s Session 1 or Session 0.
   - The `NtQuerySystemInformation(SystemProcessInformation)` inside the container returns only the container’s processes.
   - Kernel debugger commands like `!silo` can list all server silos and their associated processes at the kernel level.

---
## Observing Containers with Tools
1. **Host Tools:**
   - **Process Explorer:** Will show container processes as normal processes but in a special job (the silo job).  
   - **ObjExplore (if available):** Can reveal per-silo object directories.
   - **Total Registry (or custom tools):** Can show silo-specific registry keys hidden from normal `regedit`.
2. **Inside the Container:**
   - Running `proclist` or `tasklist` inside the container shows a minimal set of processes.
   - Filesystem is minimal (e.g., a trimmed `C:\Windows\System32`), and no host applications or drives are visible.
   - The registry and named objects are restricted.
3. **Kernel Debugger:**
   - `!silo` command in WinDbg lists all silos.
   - `!job` on the silo’s job object reveals it’s a silo and shows containerized processes.
   - Inspecting `\Sessions\...\Silos\...` object directories reveals isolated namespaces.

---
## Code Snippets and Conceptual Steps

**(Note: Creating a silo directly via NT APIs is undocumented and not supported in normal application code. Usually, container engines like Docker do this through host processes with special privileges. The examples below are conceptual only.)**

```c
// Pseudocode, as direct silo creation is not documented for public use.
// In reality, container engines call internal APIs or use special system calls.

HANDLE hJob = CreateJobObject(NULL, L"MyContainerSilo");
// Transform job into a silo (undocumented step).
// Set various silo parameters for file system root, registry hive, etc.
// Start a minimal init-like process inside the silo job.

SetInformationJobObject(hJob, JobObjectSiloRootDirectoryInformation, ...);
SetInformationJobObject(hJob, JobObjectSiloRegistryInformation, ...);
SetInformationJobObject(hJob, JobObjectSiloObjectRootInformation, ...);

// Launch processes with CreateProcess passing ExtendedProcessParameters 
// indicating silo membership, or rely on inherited membership from the initial process.
```

---

## Advanced Features

1. **Silo Hierarchies:**  
   Possible future or advanced scenarios may allow hierarchical silos, just like nested jobs, providing layered isolation.

2. **Resource Governance:**  
   Combine silo (container) isolation with job-based resource limits (CPU, memory) for granular control.

3. **Containerized Services and GUI:**  
   Currently, Windows containers focus on headless workloads. GUI isolation is limited. Future improvements might allow partial GUI siloing.

4. **Integration with Hyper-V:**  
   For stronger isolation, Windows can run containers under Hyper-V isolation. This uses VMs but still presents a container-like experience. Pure process isolation (server silo) is lighter but less isolated than a full VM.

---

## Troubleshooting and Diagnostics

1. **Namespace Leaks:**
   - If some process or handle remains open, volumes or directories may remain mapped after the container’s demise.
   - Close all references (e.g., close Explorer windows if you mounted the container’s volume outside).

2. **Access Denied:**
   - Creating or modifying silos typically requires administrative privileges or system-level code (like Docker’s host service).

3. **ETW and Logging:**
   - Use ETW (Event Tracing for Windows) providers like “Microsoft-Windows-Container” for low-level container debugging.
   - Sysinternals tools can help reveal mismatches in namespaces or leftover mounts.

---
## Summary
Windows containers (server silos) leverage the flexibility of job objects extended into silo objects to provide lightweight OS-level virtualization. By creating private object namespaces, registry hives, and filesystem views, they isolate processes and resources much like Linux containers, but built upon Windows’ existing kernel infrastructure (jobs, sessions, object directories).

For engineers working close to the metal:
- Silos show up as specialized jobs in kernel debugging.
- Container engines like Docker orchestrate the creation and tearing down of these silos.
- Tools like `!silo` in WinDbg and volume/namespace inspection reveal the internal complexity of these lightweight isolation boundaries.

Windows containers represent an interplay between user-mode container engines, NT kernel native APIs, and advanced object namespace management, enabling a modern approach to sandboxing and microservice deployment on Windows.
```