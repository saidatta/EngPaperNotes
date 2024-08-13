https://medium.com/@vikas.singh_67409/data-consistency-in-face-of-crashes-2df94392ecdf
---
### Overview
Data consistency during crashes is a critical concern in software engineering, particularly for systems relying on file storage, such as databases and distributed systems like Zookeeper. This note explores the complexities of maintaining data consistency across application, OS, and hardware crashes, examines how file systems handle write operations, and discusses the atomicity and ordering assumptions made by applications. We will dive into specific challenges, including the nuances of POSIX file system calls, the implications of write buffering, and potential data corruption scenarios.
### Key Concepts
- **Atomicity**: The guarantee that operations (like writes) are completed entirely or not at all.
- **Write Ordering**: The order in which write operations are executed and persisted on disk.
- **Durability**: Ensuring that once a write operation is confirmed, the data remains intact even after a crash.
- **Crash Consistency**: The state of data after a crash, ensuring that no partial writes or corrupt data persist.

---
### The Challenge of Data Consistency
#### Types of Crashes

1. **Application-Level Crashes**: A crash within the running application, potentially leaving data in an inconsistent state.
2. **Operating System Crashes**: OS-level crashes can interrupt active write operations, leading to partial writes.
3. **Hardware Crashes**: Sudden power loss or disk failures that occur during write operations can result in incomplete or corrupt data.

### POSIX File System Semantics
POSIX (Portable Operating System Interface) defines standards for file system calls, but its specifications primarily focus on in-memory operations. The actual persistence of these operations to disk is left to the file system's discretion, which can lead to varied behavior across different systems.
#### Write Buffering and Reordering
- **Buffering**: File systems often buffer writes to improve performance. This buffering, however, introduces the risk of reordering writes or losing data during a crash.
- **Reordering**: Buffered writes may not be flushed to disk in the order they were issued, potentially leading to data inconsistencies if a crash occurs before all writes are committed.
#### File System Calls and Durability
- **`fsync` and `fdatasync`**: These system calls are used to ensure that data is written to disk. However, their effectiveness varies based on what object they are called on and how the underlying file system handles them.
  - **`fsync`**: Flushes all buffered modifications for a file, including metadata (such as file size and modification time).
  - **`fdatasync`**: Similar to `fsync`, but only flushes the data and minimal metadata necessary for consistency (e.g., file size).
### Issues Arising from Crashes
#### Atomicity and Write Operations
- **Atomic Writes**: Applications assume that write operations are atomic (all or nothing). However, this is not always the case, particularly with larger writes:
  - **Small Writes**: Typically, sector-sized writes (e.g., 512 bytes) are atomic.
  - **Large Writes**: Writes larger than a sector (e.g., 4KB blocks) may not be atomic, leading to partial writes during a crash.
- **Partial Write Scenario**:
  - **Example**: A file write operation of 4KB is issued. If a crash occurs after only 2KB is written, the file may end up with a mix of new and old data, leading to corruption.
#### Ordering of Operations
- **Implicit Ordering Assumptions**: Applications often assume that writes will be persisted in the order they are issued. However, this assumption does not always hold:
  - **Write Coalescing**: File systems may reorder writes for efficiency, potentially violating application expectations.
  - **Example**: A Git operation assumes that appending data to a file occurs before renaming the file. If these operations are reordered due to buffering, a crash could leave the repository in an inconsistent state.
### Case Study: Zookeeper Data Corruption
In distributed systems like Zookeeper, which relies on the underlying file system for data storage, ensuring crash consistency is crucial. The reported data corruption in Zookeeper was traced back to crashes that occurred during critical file operations.
- **Zookeeper**: A distributed coordination service that uses a file system for storing configuration data. If the file system does not guarantee atomicity and correct ordering, Zookeeper's data can become corrupted during crashes.
### Techniques for Handling Crash Consistency
#### Hardware-Level Solutions
- **Battery-Backed Write Caches**: These caches ensure that writes in the buffer are committed to disk even during a power failure.
- **Write-Ahead Logging (WAL)**: Used by databases to log changes before applying them, ensuring that operations can be rolled back or reapplied after a crash.
#### Software-Level Solutions
- **Journaling File Systems**: File systems like ext4 or XFS use journaling to record changes before applying them, ensuring that after a crash, the system can recover to a consistent state.
- **Copy-on-Write (CoW)**: Techniques used by file systems like ZFS and Btrfs, where new data is written to a new location before pointers are updated, ensuring that crashes do not affect the original data.
#### Application-Level Techniques
- **Transaction Mechanisms**: Applications can implement their own transaction mechanisms to ensure atomicity, such as using WAL or staging writes to temporary files before renaming them to the target location.
### Example: Ensuring Atomicity and Ordering
Consider an application that needs to ensure both atomicity and proper ordering of writes:
#### Scenario: Atomic Append with Ordered Writes

```c
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

void write_atomic_ordered(const char* filepath, const char* data) {
    int fd = open(filepath, O_WRONLY | O_APPEND | O_SYNC);
    if (fd == -1) {
        perror("open");
        return;
    }

    if (write(fd, data, strlen(data)) == -1) {
        perror("write");
    }

    if (fsync(fd) == -1) {
        perror("fsync");
    }

    close(fd);
}

int main() {
    const char* filepath = "/path/to/file";
    write_atomic_ordered(filepath, "Important data\n");
    return 0;
}
```

- **`O_SYNC`**: Ensures that the write operation is immediately committed to disk.
- **`fsync(fd)`**: Ensures that all data, including metadata, is flushed to disk.

### Equations and Formulas

- **Calculating Safe Write Buffer Size**:
  - Given the block size \( B \) and sector size \( S \), the safe write buffer size \( W \) should ensure atomicity:
	  - ![[Screenshot 2024-08-13 at 3.05.59 PM.png]]
  - For a 4KB block size and a 512-byte sector size:
    ![[Screenshot 2024-08-13 at 3.06.15 PM.png]]
  - This ensures that writes are atomic within a single block.

---
### Conclusion

Ensuring data consistency in the face of crashes is a complex challenge involving careful consideration of atomicity, write ordering, and durability. The combination of hardware solutions (like battery-backed caches), software techniques (like journaling and CoW), and application-level strategies (like transaction mechanisms) can help mitigate these risks. For critical systems, especially those relying on file systems for storage, understanding these nuances is essential to prevent data corruption and ensure robust crash recovery.

### Further Reading and References

- **Papers**:
  - "All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications" - An in-depth exploration of crash consistency issues across file systems.
- **Tools**:
  - **BOB and ALICE**: Tools developed to automatically detect crash-related bugs in applications and file systems.
- **Journaling File Systems**:
  - **ext4**, **XFS**, **ZFS**: Examples of file systems implementing journaling or CoW techniques to enhance crash recovery.

These notes provide a comprehensive technical overview of data consistency challenges in the face of crashes, offering detailed insights, code examples, and practical solutions for building resilient systems.