https://medium.com/@vikas.singh_67409/file-deletion-in-linux-9fd903c7b509
### 1. **Introduction**
File deletion in Linux, though seemingly straightforward, involves a complex interplay of filesystem structures, kernel operations, and system calls. Understanding the underlying mechanisms is crucial for any Staff+ engineer, as it impacts file management, system performance, and the handling of race conditions.

In Linux, there isn't a "delete" operation per se. Instead, when a user executes the `rm` command, the system performs a series of operations that ultimately result in the unlinking of a file. This note provides a deep dive into these operations, covering key concepts like inodes, dentries, the `unlinkat` system call, and the nuances of file deletion when open file handles are involved.

---
### 2. **System Calls and File Deletion**
#### **2.1 The `unlinkat` System Call**

- **Purpose**: The `unlinkat` system call removes the link between a file and its inode, effectively deleting the file from the filesystem.
- **Example**:
  ```bash
  $ strace rm test.txt
  execve("/bin/rm", ["rm", "test.txt"], 0x7ffc62d5cf48 /* 55 vars */) = 0
  ...
  unlinkat(AT_FDCWD, "test.txt", 0) = 0
  ```
- **Explanation**: `strace` is a Linux utility that traces system calls made by a program. When `rm` is executed, it triggers the `unlinkat` system call, which removes the `test.txt` file's dentry from the directory and decrements the inode's hard link count.
#### **2.2 The `linkat` System Call**
- **Purpose**: The `linkat` system call creates a new link (hard or symbolic) to an existing file.
- **Example**:
  ```bash
  $ strace ln test.txt test2.txt
  execve("/bin/ln", ["ln", "test.txt", "test2.txt"], 0x7ffce6d53050 /* 55 vars */) = 0
  ...
  linkat(AT_FDCWD, "test.txt", AT_FDCWD, "test2.txt", 0) = 0
  ```
- **Explanation**: Here, `linkat` creates a new hard link to `test.txt` as `test2.txt`. Both files now point to the same inode.
#### **2.3 Race Condition Avoidance**
- **Problem**: Race conditions can occur if components of a directory path are modified in parallel with the file operations, potentially leading to unintended behavior.
- **Solution**: The `unlinkat` and `linkat` system calls, along with other `*at` system calls, mitigate these race conditions by operating relative to an open directory file descriptor. This provides a stable reference to the directory, even if it is renamed during the operation.
---
### 3. **Filesystem Structures: Inodes and Dentries**

#### **3.1 Inode Structure**
- **Definition**: An inode (index node) is a data structure on a filesystem that stores metadata about a file.
- **Components**:
  - File permissions and attributes
  - Timestamps (creation, modification, access)
  - Pointers to data blocks on the disk
  - Number of hard links
- **Example**:
  ```bash
  $ ls -li test.txt
  44704312 -rw-r--r-- 1 vikas vikas 18 Feb 27 23:59 test.txt
  ```
  - **44704312**: The inode number for `test.txt`
  - **1**: Number of hard links to this inode

#### **3.2 Dentry Structure**
- **Definition**: A dentry (directory entry) is a data structure that maps a filename to an inode.
- **Function**: Dentries are part of a directory's data, which consists of a table mapping filenames to inode numbers.
- **Example**:
  - When `test.txt` is deleted using `rm`, its dentry is removed from the directory's inode table.
---
### 4. **File Deletion Process**
#### **4.1 Unlinking a File**
- **Steps**:
  1. **Remove Dentry**: The `rm` command triggers the `unlinkat` system call, which removes the dentry for the file from the directory.
  2. **Decrement Hard Link Count**: The corresponding inode's hard link count is decremented by one.
  3. **Deallocate Inode and Data Blocks**: If the hard link count drops to zero, the inode is deallocated, and the associated data blocks are freed for reuse.
- **Important Note**: The blocks are not immediately overwritten, which makes it possible to recover the file (partially) if there is minimal subsequent filesystem activity.
#### **4.2 Open File Handles**
- **Disk vs. Memory Inodes**: The inode structure on disk differs slightly from the in-memory structure. The in-memory inode includes an additional field: the number of open file handles.
- **Impact on Deletion**:
  - A file is fully deleted only when the number of hard links is zero *and* the number of open file handles drops to zero.
  - **Equation**:
  - ![[Screenshot 2024-08-13 at 2.46.16 PM.png]]
- **Practical Example**:
  - **Windows vs. Linux**: Unlike Windows, where an open file cannot be deleted, Linux allows unlinking a file even if it's open. The file remains on disk until all handles are closed.
#### **4.3 Example: JVM and Open File Handles**
- **Scenario**:
  - The JVM loads classes from JAR files and keeps these files open.
  - If the JAR files are deleted (e.g., during an upgrade), the JVM continues to use the old versions because the files remain on disk until the JVM is restarted.
- **Example**:
  ```bash
  $ lsof -Fn -p 13704 | grep jar$ | head | cut -c2-
  /home/vikas/idea/jre64/lib/ext/localedata.jar
  /home/vikas/idea/jre64/lib/ext/cldrdata.jar
  ...
  ```

---

### 5. **Handling Race Conditions**

#### **5.1 The `*at` System Calls**

- **Problem**: Race conditions occur when parts of a directory path change during file operations, leading to unintended behavior.
- **Solution**: The `openat`, `unlinkat`, and `linkat` system calls take a directory file descriptor as an argument, preventing race conditions by ensuring the file operations are relative to the directory's current state.
- **Benefits**:
  - Stable reference to the directory, even if it is renamed.
  - Prevents the underlying filesystem from being unmounted if it is in use.

#### **5.2 Example Scenario**

- **Scenario**: Creating `dir1/dir2/xxx.dep` only if `dir1/dir2/xxx` exists.
- **Risk**: If `dir1` or `dir2` are symbolic links, they could be modified to point to a different location between the existence check and file creation.
- **Solution**:
  - Use `openat` to open a file descriptor to `dir2`.
  - Perform subsequent operations relative to this file descriptor to avoid race conditions.

---

### 6. **Conclusion**

File deletion in Linux is a sophisticated process involving multiple system calls (`unlinkat`, `linkat`), intricate filesystem structures (inodes, dentries), and considerations for race conditions. As a Staff+ engineer, understanding these details is crucial for managing file operations, ensuring system reliability, and debugging complex issues.

The nuances of file deletion, especially with open file handles and race conditions, highlight the robustness of the Linux filesystem. This understanding is essential when designing systems that interact heavily with the filesystem, particularly in environments where performance, reliability, and concurrency are critical.

---

**Equations Recap**:

1. **Inode Deallocation Condition**:
   \[
   \text{File Deleted} \iff (\text{\# of Hard Links} = 0) \land (\text{\# of Open File Handles} = 0)
   \]

2. **Race Condition Mitigation**:
   \[
   \text{Race Condition Avoidance} \iff \text{Operations Relative to Open Directory File Descriptor}
   \]

---

**Practical Insights**:

- **Proactive Problem-Solving**: By understanding the file deletion process, engineers can anticipate and mitigate issues related to open file handles, lingering data, and race conditions.
- **Real-World Applications**: The interaction between the JVM and filesystem during upgrades demonstrates the importance of understanding how file handles can influence application behavior post-deletion.

These notes are intended to provide comprehensive insights into file deletion in Linux, ensuring that Staff+ engineers can apply this knowledge in real-world scenarios, leading to more reliable and maintainable systems.