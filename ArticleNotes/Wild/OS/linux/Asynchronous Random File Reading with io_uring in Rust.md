https://www.skyzh.dev/blog/2021-01-30-async-random-read-with-rust/

---
## Project Overview
- **Repository**: [GitHub - skyzh/uring-positioned-io](https://github.com/skyzh/uring-positioned-io)
- **Objective**: Implement asynchronous random file reading in Rust using `io_uring`.
- **Usage Example**:
  ```rust
  ctx.read(fid, offset, &mut buf).await?;
  ```
## io_uring Introduction
- **Description**: An asynchronous I/O interface in the Linux kernel, introduced in Linux 5.1 (May 2019).
- **Applications**:
  - Used in projects like RocksDB, Tokio, and QEMU.
  - Demonstrates better performance than Linux AIO in most tests.
## Random File Reading Scenario
- **Database Systems Use Case**: Need for multiple threads to read any file location concurrently.
- **Existing Methods**: 
  - `mmap` for direct memory access.
  - `pread` for offset-based reading.
- **Limitation**: Both methods block the current thread.
## Basic Usage of io_uring
- **Syscalls**: Refer to liburing for user-friendly API.
- **Tokio's io_uring Crate**: Provides a Rust API for `io_uring`.
### Creating a Ring
- **Code Example**:
  ```rust
  use io_uring::IoUring;
  let ring = IoUring::new(256)?;
  let ring = ring.concurrent();
  ```
- **Description**: Each ring corresponds to a submission queue (SQ) and a completion queue (CQ).
### File Reading Process
- **Task Construction**: Use `opcode::Read` for file reading tasks.
- **Adding Task to Queue**:
  ```rust
  use io_uring::{opcode, types::Fixed};
  let read_op = opcode::Read::new(Fixed(fid), ptr, len).offset(offset);
  let entry = read_op.build().user_data(user_data);
  unsafe { ring.submission().push(entry)?; }
  ```
- **Task Submission**: Call `ring.submit()` to submit tasks to the kernel.

## Implementation of Asynchronous File Reading
- **Context Creation**: Create `UringContext` for using `io_uring`.
- **UringPollFuture**: Runs in the background for task submission and polling.
- **UringReadFuture**: Interface for reading files, creating `UringTask` with a channel inside.
![[Screenshot 2023-12-30 at 2.52.03 PM.png]]
## Benchmarking and Performance Comparison
- **Setup**: Comparison with `mmap` using 128 1G files for random 4K block reads.
- **Test Cases**:
  - Various thread counts and concurrency levels for `mmap` and `io_uring`.
- **Metrics**: Throughput (op/s) and Latency (ns).
- **Results**: `mmap` outperformed the packaged `io_uring`.

## Possible Improvements![[Screenshot 2023-12-30 at 2.53.15 PM.png]]
![[Screenshot 2023-12-30 at 2.53.27 PM.png]]
![[Screenshot 2023-12-30 at 2.53.34 PM.png]]
- ![[Screenshot 2023-12-30 at 2.54.02 PM.png]]
- **Performance Analysis**: Comparing Rust/C implementation on `nop` instruction to assess Tokio's packaging overhead.
- **Direct I/O Testing**: Yet to be tested; currently only Buffered I/O tested.
- **Comparison with Linux AIO**: To benchmark against another asynchronous I/O method.
- **Memory Management**: Addressing potential memory leaks when Future is aborted.

## Conclusion
The implementation of asynchronous random file reading in Rust using `io_uring` demonstrates a novel approach to file I/O operations. Although the current implementation shows room for improvement in performance, it provides a foundation for further optimization and enhancement.
## some possible improvements
- It seems that the performance is not very good now `io_uring`after the packaging between me and Tokio. `io_uring`You can then test the overhead introduced by Tokio's layer of packaging by comparing the performance of Rust/C on the nop instruction.
- Test the performance of Direct I/O. Currently only Buffered I/O has been tested.
- Compare with Linux AIO. (Performance can't be worse than Linux AIO (crying bitterly)
- Use perf to see where the current bottleneck is. Currently, I cannot apply for memory `cargo flamegraph`after mounting it `io_uring`. (Give it a try, maybe there will be a sequel
- Currently, the user must ensure that `&mut buf`it is valid throughout the read cycle. If Future is aborted, there will be a memory leak problem. For similar issues with futures-rs, see [https://github.com/rust-lang/futures -rs/issues/1278](https://translate.google.com/website?sl=auto&tl=en&hl=en&u=https://github.com/rust-lang/futures-rs/issues/1278) . Tokio's current I/O solves this problem by making two copies (first to the cache, then to the user).
- Perhaps writing files and other operations can also be packaged.
---
