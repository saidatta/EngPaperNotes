## Overview of io_uring
- **Description**: `io_uring` is an asynchronous I/O interface provided by Linux, added in 2019. As of Linux 5.12.10, it has evolved significantly.
- **Implementation File**: Located in `.fs/io_uring.c` in the Linux kernel.
## Userland API of io_uring
- **System Calls**: Uses three syscalls - `io_uring_setup`, `io_uring_enter`, and `io_uring_register`.
  - `io_uring_setup`: Establishes the `io_uring` context.
  - `io_uring_enter`: Used to submit and retrieve completion tasks.
  - `io_uring_register`: Registers buffer shared between kernel and user.
- **Queue Structures**:
  - **SQ (Submission Queue)**: Circular queue in continuous memory for storing operation data.
  - **CQ (Completion Queue)**: Circular queue for storing results of operations.
  - **SQE (Submission Queue Entry)**: Represents an item in the submission queue.
  - **CQE (Completion Queue Entry)**: Represents a completed item in the queue.

## Initializing io_uring
- **Function**: 
	- `long io_uring_setup(u32 entries, struct io_uring_params __user *params)`
- **Process**:
  - Returns a file descriptor and stores supported functions and data structure offsets in `params`.
  - Users map the memory area shared with the kernel to access `io_uring` context information.
	  - ![[Screenshot 2023-12-30 at 2.09.19 PM.png]]
## Memory Mapping in io_uring_setup
- **SQE and CQE Sizes**: SQE is 64B, and CQE is 16B.
- **Allocation**: If the CQ length is not set in `params`, the kernel allocates `entries` for SQE and `entries * 2` for CQE.
## Description of Tasks in io_uring
- **Capability**: Handles various I/O requests like file operations (read, write, open), network operations (connect, send, recv), etc.
- **Example**: Using `fsync` to demonstrate the operation structure.
## Implementation of Operations
- **Operation Definitions**: Defined in `io_op_def io_op_defs[]` array, specifying parameters for supported operations.
- **Preparation and Execution Functions**: Each operation (e.g., `fsync`) corresponds to preparation (`io_fsync_prep`) and execution (`io_fsync`) functions. Asynchronous operations end with `_async`.
## Transmitting Operational Information
- **Process**: Users write operations into SQ, and results are harvested from CQ.
- **Structures**: 
  - **SQE**: 64B structure in `include/uapi/linux/io_uring.h` containing operation information.
  - **CQE**: 16B structure storing execution results.
## Submission and Completion of Tasks
- **Interaction**: Uses a circular queue for task submission (SQ) and harvesting (CQ).
- **Kernel and User Mode Processing**: Describes how tasks are submitted, processed by the kernel, and harvested by the user .![[Screenshot 2023-12-30 at 2.11.10 PM.png]]

![[Screenshot 2023-12-30 at 2.20.21 PM.png]]## Implementation Details in the Kernel
- **Options**: `io_uring` offers options like `IORING_SETUP_IOPOLL` for polling and `IORING_SETUP_SQPOLL` for creating a kernel thread to harvest user-submitted tasks.
![[Screenshot 2023-12-30 at 2.20.00 PM.png]]
- **Performance Modes**:
  - **Default**: Submit tasks via `io_uring_enter`.
  - **IOPOLL**: Polling mode for task submission and harvesting.
  - **SQPOLL**: Tasks submitted and harvested without syscalls, kernel thread awakens through `io_uring_enter`.
  - ![[Screenshot 2023-12-30 at 2.19.45 PM.png]]
## Task Dependency Management in io_uring
- **Use Case**: Managing sequences of operations, like multiple writes followed by `fsync`.
- **Control Options**: `IO_SQE_LINK`, `IOSQE_IO_DRAIN`, `IOSQE_IO_HARDLINK` for establishing task sequences.
- **Internal Management**: Uses linked lists for managing task dependencies.
- ![[Screenshot 2023-12-30 at 2.18.43 PM.png]]
## Summary and Insights
- **Usage Scenarios**: Selection of `io_uring` modes depends on operational needs like polling or real-time performance.
- **Performance Considerations**: For Buffered I/O, `io_uring` might not significantly outperform direct syscalls. Asynchronous flags (`IOSQE_ASYNC`) and task control options (`IO_SQE_LINK`, etc.) offer more control over task execution.
## Appendix: Testing io_uring with fio
- **Commands**: Examples of using `fio` with different `io_uring` modes.
- **Kernel Function Tracing**: Using tools like `bcc` (eBPF) and `trace-cmd` to

 generate flame graphs for function calls.
 ![[Screenshot 2023-12-30 at 2.19.06 PM.png]]

---
### System Calls (Detailed)
- **`io_uring_setup` Example**:
  ```c
  struct io_uring_params params;
  memset(&params, 0, sizeof(params));
  int fd = io_uring_setup(entries, &params);
  ```
- **`io_uring_enter` Usage**:
  - Used to submit and complete I/O tasks.
  - Example of submitting SQEs and waiting for completion:
    ```c
    int ret = io_uring_enter(fd, to_submit, min_complete, flags, sig);
    ```
### Queue Structures (Detailed)
- **SQ (Submission Queue)**:
  - Stores the index of SQEs.
  - Example Structure Initialization:
    ```c
    struct io_uring_sq {
        unsigned *head;
        unsigned *tail;
        unsigned *ring_mask;
        unsigned *ring_entries;
        unsigned *flags;
        unsigned *array;
    };
    ```
- **CQ (Completion Queue)**:
  - Holds the completion data for I/O tasks.
  - Example Structure:
    ```c
    struct io_uring_cq {
        unsigned *head;
        unsigned *tail;
        unsigned *ring_mask;
        unsigned *ring_entries;
        struct io_uring_cqe *cqes;
    };
    ```

## Memory Mapping in io_uring_setup
- **Mapping SQ and CQ**:
  - Users map the memory area for SQ and CQ after `io_uring_setup`.
  - Example Memory Mapping:
    ```c
    struct io_uring_sq *sq = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    struct io_uring_cq *cq = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    ```
## Task Submission and Completion
- **Writing an SQE**:
  - Users populate an SQE structure with I/O operation details.
  - Example SQE Setup:
    ```c
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, fd, buf, nbytes, offset);
    sqe->user_data = user_data;
    ```
- **Reading a CQE**:
  - Users read the completion data from a CQE.
  - Example CQE Reading:
    ```c
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    // Process cqe->res as the result
    io_uring_cqe_seen(&ring, cqe);
    ```
## Implementing I/O Operations
- **Preparing and Executing `fsync`**:
  - Example Preparation Function:
    ```c
    static int io_fsync_prep(struct io_kiocb *req, const struct io_uring_sqe *sqe) {
      req->flags = sqe->fsync_flags;
      // Other preparation code
    }
    ```
  - Execution Function:
    ```c
    static int io_fsync(struct io_kiocb *req, unsigned int issue_flags) {
      // Actual fsync operation
    }
    ```
## Task Dependency Management (Expanded)
- **Linking Tasks**: Use `IO_SQE_LINK` to establish a sequence.
  - Example Sequence:
    ```c
    io_uring_prep_write(sqe, fd, buf1, nbytes, offset1);
    sqe->flags |= IOSQE_IO_LINK;
    io_uring_prep_fsync(sqe2, fd, IORING_FSYNC_DATASYNC);
    ```
## Using io_uring with fio (Expanded)
- **Benchmarking Example**:
  - Benchmarking `io_uring` with different modes using `fio`:
    ```bash
    fio --filename=testfile --size=1G --ioengine=io_uring --rw=read --bs=4k --iodepth=4 --direct=1 --name=test
    ```

---
