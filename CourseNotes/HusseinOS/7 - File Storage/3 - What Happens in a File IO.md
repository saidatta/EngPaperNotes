Below is a set of **Obsidian**-formatted notes titled “OS: What Happens in a File I/O.” These notes walk through the journey of a **read** request (and briefly writing) from an application’s user space down to the physical disk, covering **POSIX** concepts, block/page translations, caching, and more. Feel free to copy/paste them into your Obsidian vault or any other Markdown editor.

## 1. Introduction

When an application **reads** or **writes** a file in a modern operating system (like Linux), there are multiple layers involved:

1. **User Space**: The application’s code and memory (buffers).
2. **System Call Interface**: The OS boundary (POSIX or Win32 calls).
3. **Kernel Space**: Where the file system logic, page cache, and device drivers live.
4. **Block Device / Storage Hardware**: HDD or SSD, which finally services the I/O.

This note explores that path in detail, focusing on **Linux/POSIX** (though the concepts are somewhat similar in other OSes).

---

## 2. POSIX and File Descriptors

### 2.1 POSIX vs. Win32
- **POSIX**: A standard set of Unix-like APIs for I/O. Common calls:
  - `read(fd, buffer, size)`
  - `write(fd, buffer, size)`
  - `fsync(fd)`
- **Windows** uses **Win32 APIs** (e.g., `ReadFile()`, `WriteFile()`), which differ in naming and behavior, but the overall concepts (system calls, caching, etc.) have parallels.

### 2.2 File Descriptors
- In Linux/Unix, every open file (or socket, or pipe) has a **file descriptor** (an integer) in the **process**.
- Internally, the kernel tracks which **inode** (file metadata) each file descriptor refers to.

---

## 3. High-Level Read Flow (Step by Step)

Below is a simplified step-by-step for reading a file named `test.txt` which is **5,000 bytes** in size.

### Assumptions for Simplicity
1. **Logical Sector Size** = 4096 bytes
2. **Physical Sector Size** = 4096 bytes  
   → PBA = LBA (1:1 mapping, so no extra complications)
3. **File System Block Size** = 4096 bytes  
   → One file system block = one LBA = one physical sector
4. A **page** in virtual memory is also 4096 bytes.

**Hence**, to read 5,000 bytes, we need:
- 1 block (4,096 bytes)
- Plus another partial block (the next 4 KB block) because 5,000 - 4,096 = 904 bytes. Still requires a **full** 4 KB read.

---

## 4. Detailed Diagram of the Read Path

Below is a **Mermaid** diagram illustrating the flow from an application down to the disk.

```mermaid
flowchart TB
    A[Application<br>(User Space)] --1. read(fd, buf, size)--> B[System Call<br>POSIX API]
    B --2. Check FS & Page Cache--> C[Kernel FS Layer]
    C --3. If block in cache? Yes or No--> D[Page Cache]
    D --4. If missing, read block(s) from disk--> E[Block Device Driver]
    E --5. Issue LBA read to disk--> F[Storage Device (HDD/SSD)]
    F --returns data--> E
    E --returns data--> D
    D --6. Copy data to user buffer--> A
```

1. **Application** calls `read()`.
2. OS (kernel) sees which file descriptor, determines which blocks to fetch.
3. **Kernel** checks if the file’s blocks are in the **page cache**.
   - If **cache miss**, it requests the blocks from the disk.
   - If **cache hit**, it can skip the disk read.
4. Data arrives in kernel memory (page cache).
5. **Kernel** copies the requested bytes from page cache to the user’s `buf`.
6. `read()` call returns the number of bytes read.

---

## 5. Breaking Down the Translations

### 5.1 Finding the File Blocks
- The file system consults its **metadata** to see which **block** belongs to `test.txt`.
- For example:  
  - Start block = 6  
  - Next block = 3  
  - (No more blocks after block 3 for this file’s size)

### 5.2 Converting FS Blocks to LBA
- Because in this scenario **1 file system block = 1 LBA**:
  - **Block #6** → **LBA #6**
  - **Block #3** → **LBA #3**

### 5.3 Checking the Page Cache
- The kernel (via the file system layer) looks up:
  - “Is **block #6** (or LBA #6) already cached in a page cache frame?”
  - “Is **block #3** (or LBA #3) already cached?”

If a block is found in the cache, no disk I/O is needed for that part.

### 5.4 Disk Read (If Needed)
- If **block #6** is **not** in cache, the kernel issues an I/O request:
  - `READ(LBA = 6, length = 1 block)` (4 KB).
- The disk handles the request. For SSD/HDD:
  - **HDD** might do a direct offset calculation (LBA #6 is at some physical location).
  - **SSD** uses an FTL (Flash Translation Layer) to map LBA #6 to a particular flash page internally.

When the disk finishes, it returns the 4 KB block to the kernel’s page cache.

---

## 6. The Two Copies

### 6.1 Disk → Kernel (Page Cache)
After reading from the disk, data lands in **kernel memory**. This memory region is a “page” in the **page cache**.

### 6.2 Kernel → User Buffer
The kernel then copies from the page cache to the **user space** buffer passed to `read(fd, buf, size)`.  
- This is a memory-to-memory copy within the same machine, but across the user-kernel boundary.

> **Note**: Some zero-copy mechanisms exist for advanced I/O, but standard reads do this **double-copy** approach.

---

## 7. Example Code Snippet (Pseudo-C)

```c
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main() {
    int fd = open("test.txt", O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    char buffer[5000];
    ssize_t nread = read(fd, buffer, 5000); // Step 1: System call
    if (nread < 0) {
        perror("read");
        close(fd);
        return 1;
    }

    // buffer now holds the content from the file
    printf("Read %zd bytes\n", nread);

    close(fd);
    return 0;
}
```

Behind the scenes:

1. `open()` → obtains a file descriptor from the kernel, referencing `test.txt`’s inode.
2. `read(fd, buffer, 5000)` → triggers all the steps we discussed (cache check, block lookup, I/O if needed, etc.).
3. After the data is in the kernel’s page cache, it’s copied to `buffer`.

---

## 8. Brief Note on Writes

- **Write Path** is similar but reversed:
  1. `write(fd, user_buffer, size)` → copy from user buffer into **kernel page cache**.
  2. The kernel marks those pages **dirty**.
  3. At some later time (or on `fsync`), dirty pages are flushed to disk.

This buffering is why a normal `write` call returns quickly (the OS writes to cache, not the disk immediately). For durability guarantees, an application should call `fsync()` or open the file with sync/append modes.

---

## 9. SSD vs. HDD Translation

### 9.1 HDD
- Typically **LBA** → **Physical location** with a near 1:1 offset calculation.  
- The disk’s internal logic is simpler (though advanced HDDs can do more behind the scenes).

### 9.2 SSD
- The **Flash Translation Layer (FTL)** manages wear-leveling and remapping.  
- **LBA #6** might physically reside anywhere in the NAND.  
- Writes may cause erasures, block reclamations, etc., all hidden behind the FTL.  
- The OS sees an LBA interface, but the SSD is doing extra translation and management internally.

---

## 10. Summary of Key Points

1. **POSIX Read**:
   - `read(fd, buf, size)` triggers a series of **translations** from file descriptor → file system block → LBA → actual physical block on disk.
2. **Page Cache**:
   - Minimizes disk I/O by caching recently used blocks.  
   - Requires **two copies**: disk → kernel, kernel → user space.
3. **Block-Size Alignment**:
   - Reading/writing partial data still results in **full** block read/writes.
4. **SSD vs. HDD**:
   - On SSD, the **FTL** does its own internal mapping.  
   - On HDD, the LBA often correlates to a physical offset on the platter.
5. **Performance**:
   - Cache **hits** are fast (pure memory copy).
   - Cache **misses** require **physical** I/O, which is relatively slow.
6. **Writes**:
   - Typically buffered in the page cache and only flushed to disk later, unless forcibly synced.

---

## 11. Further Reading

- **Linux Man Pages**: `man 2 read`, `man 2 write`, `man 2 fsync`
- **Advanced I/O**: `O_DIRECT`, `mmap()`, `sendfile()` (bypass extra copies).
- **HDD vs. SSD Internals**:
  - [How a Hard Drive Works](https://en.wikipedia.org/wiki/Hard_disk_drive#Technology)
  - [SSD FTL Explanation](https://en.wikipedia.org/wiki/Flash_translation_layer)
- **OS Textbooks**:
  - “Operating Systems: Three Easy Pieces” by Remzi & Andrea Arpaci-Dusseau for deeper OS kernel I/O.

**Tags**:  
- #OperatingSystems  
- #FileIO  
- #POSIX  
- #PageCache  
- #BlockDevices  

---

**End of Notes**.