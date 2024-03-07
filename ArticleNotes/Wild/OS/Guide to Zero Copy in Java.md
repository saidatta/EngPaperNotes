https://heapdump.cn/article/3290793
### Preface
Zero Copy is a pivotal concept in system performance optimization, particularly in data transfer operations. It's a technique that reduces CPU usage and context switching, thereby enhancing the efficiency of I/O operations. This technique is integral to understanding why systems like Kafka and RocketMQ exhibit high performance.
### 1. Understanding Zero Copy
Zero Copy, as the name implies, eliminates the need to copy data from one memory area to another during I/O operations. Traditionally, data transfer involves copying data from the source to the kernel space and then to the user space or vice versa, consuming significant CPU resources and time. Zero Copy aims to minimize these operations, thereby reducing CPU utilization and improving I/O efficiency.
### 2. Traditional I/O Execution Process
![[Screenshot 2024-02-04 at 10.14.50 PM.png]]
In a conventional I/O process, data transfer involves multiple steps:
1. **Reading data:** From the disk to the kernel buffer and then copying it to the user buffer.
2. **Writing data:** First to the socket buffer and finally to the network card.
This process involves context switching between user and kernel modes, and data is copied multiple times, leading to inefficiency.
### 3. Key Concepts Behind Zero Copy
Before diving into Zero Copy implementations, let's review some fundamental concepts:
- **Kernel Space and User Space:** Memory allocated to each process is divided into user space (accessible by user applications) and kernel space (accessible by the OS kernel).
- **User Mode and Kernel Mode:** Processes run in user mode when executing user space code and in kernel mode when executing kernel space code.
- **Context Switching:** The process of saving the state of a CPU so that it can be restored and execution resumed later.
	![[Screenshot 2024-02-04 at 10.16.18 PM.png]]
- **Virtual Memory:** Allows mapping of virtual addresses to physical addresses, enabling shared memory areas and reducing copy operations.
![[Screenshot 2024-02-04 at 10.18.41 PM.png]]
- **DMA (Direct Memory Access):** A feature that allows peripheral devices to access memory without involving the CPU, thus reducing CPU load during data transfer.
	1. The user application process calls the read function, initiates an IO call to the operating system, enters a blocking state, and waits for data to be returned.
	2. After receiving the instruction, the CPU initiates instruction scheduling to the DMA controller.
	3. After receiving the IO request, DMA sends the request to the disk;
	4. The disk puts the data into the disk control buffer and notifies the DMA
	5. DMA copies data from the disk controller buffer to the kernel buffer.
	6. DMA sends a signal to the CPU that the data has been read, and exchanges the work to the CPU, which is responsible for copying the data from the kernel buffer to the user buffer.
	7. The user application process switches from the kernel mode back to the user mode and releases the blocking state.
![[Screenshot 2024-02-04 at 10.19.03 PM.png]]
### 4. Implementations of Zero Copy

Zero Copy can be achieved through various methods:

#### 4.1 mmap + write
![[Screenshot 2024-02-04 at 10.25.20 PM.png]]
- **mmap** is used to map disk file data directly into the kernel space, which is then shared with user space, reducing the need for data copying.
- **write** transfers data from the user buffer to the network, still requiring a copy from the user to socket buffer.
#### 4.2 sendfile
![[Screenshot 2024-02-04 at 10.26.17 PM.png]]
- **sendfile** optimizes the data transfer process by eliminating the need to copy data to the user space. Data goes directly from the kernel buffer to the socket buffer.
- Java NIO's `transferTo()` and `transferFrom()` methods leverage the underlying `sendfile` system call for efficient data transfer.

#### 4.3 sendfile + DMA scatter/gather
![[Screenshot 2024-02-04 at 10.26.53 PM.png]]
- An enhancement of `sendfile` that utilizes DMA scatter/gather, allowing direct data transfer from the kernel buffer to the network card without intermediate copying.

### 5. Zero Copy in Java

Java offers support for Zero Copy through NIO (New I/O):
#### 5.1 Java NIO's Support for mmap
Java NIO's `MappedByteBuffer` class facilitates memory-mapped file I/O, enabling efficient file reading and writing by mapping files directly into memory.

```java
FileChannel readChannel = FileChannel.open(Paths.get("./source.txt"), StandardOpenOption.READ);
MappedByteBuffer buffer = readChannel.map(FileChannel.MapMode.READ_ONLY, 0, readChannel.size());
FileChannel writeChannel = FileChannel.open(Paths.get("./dest.txt"), StandardOpenOption.WRITE, StandardOpenOption.CREATE);
writeChannel.write(buffer);
```

#### 5.2 Java NIO's Support for sendfile

Java NIO's `FileChannel` class includes `transferTo()` and `transferFrom()` methods that utilize the `sendfile` system call, providing an efficient way to transfer data between channels.

```java
FileChannel sourceChannel = FileChannel.open(Paths.get("./source.txt"), StandardOpenOption.READ);
FileChannel destChannel = FileChannel.open(Paths.get("./dest.txt"), StandardOpenOption.WRITE, StandardOpenOption.CREATE);
sourceChannel.transferTo(0, sourceChannel.size(), destChannel);
```

### Conclusion

Zero Copy is a crucial optimization technique in high-performance computing and networking. By leveraging Java NIO's capabilities and understanding the underlying system concepts, software engineers can implement efficient data transfer operations, significantly reducing CPU usage and improving application performance.