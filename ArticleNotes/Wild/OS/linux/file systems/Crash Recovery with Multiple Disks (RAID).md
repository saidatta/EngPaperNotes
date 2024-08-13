https://medium.com/@vikas.singh_67409/crash-recovery-with-multiple-disks-raid-de2e1fcaa961
---
### Overview
RAID (Redundant Array of Independent Disks) is a technology developed to address the limitations of single-disk storage systems by using multiple disks to increase performance, reliability, and storage capacity. This note delves into the technical details of RAID, its various levels, and how it manages crash recovery. The focus is on the principles of RAID, including data striping, mirroring, and parity, along with the specific challenges each RAID level addresses. Examples, code snippets, and equations are provided for comprehensive understanding.
### Key Concepts
- **RAID**: A method of combining multiple disk drives into a single logical unit to improve performance, reliability, and storage capacity.
- **Data Striping**: Distributing data across multiple disks to increase performance.
- **Mirroring**: Duplicating data across multiple disks to ensure redundancy.
- **Parity**: A method of error checking that enables recovery of lost data in the event of a disk failure.
---
### The Origins of RAID
RAID was introduced in the famous Berkeley paper titled "A Case for Redundant Array of Inexpensive Disks" in 1988. The technology aimed to overcome the bottleneck caused by the relatively slow performance of mechanical disks, which could not keep pace with the rapid improvements in CPU and memory performance as predicted by Moore's Law.
### The Need for RAID
- **Performance Bottlenecks**: As CPU and memory performance improved, the slow I/O performance of mechanical disks became a significant bottleneck, especially for workloads with random data access patterns.
- **Reliability Concerns**: Using multiple disks increases the likelihood of disk failures, reducing the overall system reliability if redundancy is not implemented.
### RAID Levels
RAID levels define different configurations for combining multiple disks, each offering varying trade-offs between performance, reliability, and storage efficiency.

---
### **First Level RAID: Mirrored Disks (RAID 1)**
#### Description
RAID 1 involves mirroring data across two or more disks. Each write operation is duplicated, ensuring that there is always a redundant copy of the data. This setup provides high fault tolerance at the cost of doubling the number of disks required.
#### Technical Details
- **MTTF (Mean Time to Failure)**: The redundancy in RAID 1 significantly increases the MTTF. For example, if one disk fails every 30,000 hours, the RAID 1 array can sustain over 500 years of operation without data loss, assuming no simultaneous failures.
- **Read Performance**: RAID 1 can improve read performance by allowing data to be read from both disks in parallel.
#### Example Configuration
- **Number of Disks**: 2 (minimum)
- **Data Redundancy**: 100% (50% storage efficiency)
- **Fault Tolerance**: High (can survive multiple disk failures if not simultaneous)

```text
Disk 1: [ Data Block A | Data Block B | Data Block C | ... ]
Disk 2: [ Data Block A | Data Block B | Data Block C | ... ]
```

#### RAID 1 Illustration
![[Screenshot 2024-08-13 at 3.13.16 PM.png]]

```
Disk 1: | A | B | C | D |
Disk 2: | A | B | C | D |
```

In this setup, data blocks A, B, C, and D are mirrored across both disks.

---

### **Second Level RAID: Hamming Code for Error Correction**

#### Description
RAID 2 uses Hamming code for error detection and correction, similar to the technique used in RAM. Data is striped across disks at the bit level, with additional disks storing parity and error-correcting code (ECC) bits.
#### Technical Details
- **Bit-Level Striping**: Data is split at the bit level across multiple disks, with one or more disks dedicated to parity and ECC.
- **Error Detection/Correction**: The Hamming code can detect and correct single-bit errors and detect two-bit errors.
#### Example Configuration
- **Number of Disks**: Varies (depends on the ECC overhead)
- **Data Redundancy**: Low overhead compared to RAID 1
- **Fault Tolerance**: Moderate (can correct single-bit errors)
```text
Disk 1: [ Bit 1 of Block A | Bit 1 of Block B | ... ]
Disk 2: [ Bit 2 of Block A | Bit 2 of Block B | ... ]
...
Disk N: [ ECC for Block A | ECC for Block B | ... ]
```
#### RAID 2 Illustration
![[Screenshot 2024-08-13 at 3.14.23 PM.png]]
```
Disk 1: | 0 | 1 | 0 | 1 |
Disk 2: | 1 | 0 | 1 | 0 |
Disk 3: | 1 | 1 | 0 | 0 |
Disk 4: | Parity |
```

In this setup, the data is striped at the bit level across the disks, with additional disks storing parity and ECC.

---
### **Third Level RAID: Single Check Disk Per Group (RAID 3)**
#### Description
RAID 3 reduces the number of check disks by using a single parity disk for a group of data disks. Data is striped at the byte level, and the parity disk stores the XOR of all corresponding bytes from the data disks.
#### Technical Details
- **Byte-Level Striping**: Data is split at the byte level across multiple disks, with one disk dedicated to storing the parity bit.
- **Error Recovery**: If a data disk fails, the missing data can be reconstructed using the XOR of the remaining data and the parity disk.
#### Example Configuration
- **Number of Disks**: Minimum of 3 (2 data disks + 1 parity disk)
- **Data Redundancy**: Low overhead compared to RAID 1 and RAID 2
- **Fault Tolerance**: Can tolerate a single disk failure

```text
Disk 1: [ Byte 1 of Block A | Byte 1 of Block B | ... ]
Disk 2: [ Byte 2 of Block A | Byte 2 of Block B | ... ]
Disk 3: [ Parity (Byte 1 XOR Byte 2) | ... ]
```
#### RAID 3 Illustration
![[Screenshot 2024-08-13 at 3.16.17 PM.png]]

```
Disk 1: | A1 | B1 | C1 | D1 |
Disk 2: | A2 | B2 | C2 | D2 |
Disk 3: | P  | P  | P  | P  |
```

In this setup, the parity disk stores the XOR of the bytes from the data disks.

---
### **Fourth Level RAID: Independent Read and Writes (RAID 4)**
#### Description
RAID 4 improves on RAID 3 by allowing independent read and write operations on each disk. Data is striped at the block level, and a single parity disk is used for error recovery.
#### Technical Details
- **Block-Level Striping**: Data is split at the block level across multiple disks, with one disk dedicated to storing the parity block.
- **Independent I/O**: Unlike RAID 3, RAID 4 allows for independent read and write operations on each disk, improving performance for small I/O operations.
#### Example Configuration
- **Number of Disks**: Minimum of 3 (2 data disks + 1 parity disk)
- **Data Redundancy**: Low overhead
- **Fault Tolerance**: Can tolerate a single disk failure
```text
Disk 1: [ Block 1 of File A | Block 1 of File B | ... ]
Disk 2: [ Block 2 of File A | Block 2 of File B | ... ]
Disk 3: [ Parity (Block 1 XOR Block 2) | ... ]
```

#### RAID 4 Illustration
![[Screenshot 2024-08-13 at 3.17.36 PM.png]]

```
Disk 1: | A1 | B1 | C1 | D1 |
Disk 2: | A2 | B2 | C2 | D2 |
Disk 3: | P  | P  | P  | P  |
```

**Equation for Parity Calculation:**
![[Screenshot 2024-08-13 at 3.18.36 PM.png]]
This equation allows the parity block to be updated efficiently without reading all data blocks.

---
### **Fifth Level RAID: Spread Data/Parity Over All Disks (RAID 5)**
#### Description
RAID 5 distributes both data and parity across all disks in the array, eliminating the bottleneck caused by a single parity disk in RAID 4. Each disk in the array contains both data and parity blocks.
#### Technical Details
- **Distributed Parity**: Parity information is spread across all disks, allowing for better load balancing and higher parallelism.
- **Fault Tolerance**: Can tolerate a single disk failure and reconstruct the lost data using the parity information from the other disks.
#### Example Configuration
- **Number of Disks**: Minimum of 3
- **Data Redundancy**: Balanced between performance and storage efficiency
- **Fault Tolerance**: Can tolerate a single disk failure
```text
Disk 1: [ Data Block 1 | Data Block 2 | Parity Block 3 | ... ]
Disk 2: [ Parity Block 1 | Data Block 3 | Data Block 4 | ... ]
Disk 3: [ Data Block 5 | Parity Block 2 | Data Block 6 | ... ]
```
#### RAID 5 Illustration
```
Disk 1: | A1 | A2 | P3 | A4 |
Disk 2: | P1 | B2 | B3 | B4 |
Disk 3:

 | C1 | P2 | C3 | C4 |
```

In this setup, parity is distributed across all disks, improving parallelism and performance.

---
### **RAID 0: Data Striping Without Redundancy**

#### Description
RAID 0, often referred to as "striping," is the simplest RAID configuration. It involves splitting data across multiple disks in a way that allows for increased performance, but it does not provide any redundancy or fault tolerance. In RAID 0, data is divided into blocks and distributed evenly across all disks in the array. Each block is written to a different disk, which allows multiple disks to read or write data simultaneously.

#### Technical Details

- **Data Striping**: RAID 0 stripes data at the block level across all disks in the array. For example, if you have two disks in a RAID 0 array, the first block of data is written to Disk 1, the second block to Disk 2, the third block to Disk 1, and so on.
- **Performance**: RAID 0 significantly improves read and write performance because multiple disks can operate concurrently. The total throughput of the array is approximately the sum of the throughput of all individual disks.
- **No Redundancy**: RAID 0 does not provide any redundancy or fault tolerance. If one disk in the array fails, all data in the RAID 0 array is lost because part of each file is stored on each disk.

#### Example Configuration

- **Number of Disks**: Minimum of 2 (but can be more)
- **Data Redundancy**: None
- **Fault Tolerance**: None (a single disk failure results in complete data loss)

```text
Disk 1: [ Block 1 | Block 3 | Block 5 | ... ]
Disk 2: [ Block 2 | Block 4 | Block 6 | ... ]
```

#### RAID 0 Illustration

```
Disk 1: | A1 | B1 | C1 | D1 |
Disk 2: | A2 | B2 | C2 | D2 |
```

In this setup, the data blocks are alternately written to each disk. For example, the first block of File A (`A1`) is written to Disk 1, and the second block (`A2`) is written to Disk 2.

#### Performance Equation

If each disk in the RAID 0 array has a read/write speed of `S` and you have `N` disks in the array, the total throughput `T` of the RAID 0 array can be approximated by:
![[Screenshot 2024-08-13 at 3.22.14 PM.png]]
For example, if each disk can read or write at 100 MB/s and you have 4 disks in a RAID 0 array, the total throughput will be approximately 400 MB/s.
#### Use Cases
- **High-Performance Applications**: RAID 0 is often used in situations where high speed is more critical than data reliability, such as in video editing, gaming, or high-performance computing.
- **Temporary Data Storage**: RAID 0 can be used for temporary data that can be easily recreated or is not critical, as there is no protection against data loss.
#### Limitations
- **No Fault Tolerance**: The most significant limitation of RAID 0 is that it offers no fault tolerance. If any single disk in the RAID 0 array fails, all data in the array is lost.
- **Risk of Data Loss**: The probability of data loss increases with the number of disks in the array because the failure of any one disk results in total data loss.
#### Conclusion
RAID 0 is a valuable configuration for scenarios where performance is the primary concern, and data redundancy is not necessary. However, it should be used with caution due to its lack of fault tolerance. It is best suited for applications that can afford to lose data and can benefit from the increased performance provided by striping data across multiple disks.

-----
### RAID Write Hole Issue
#### Problem Description
The RAID write hole refers to a situation where a system crash or power failure occurs during a write operation, resulting in a mismatch between data and parity. This can lead to data corruption if not handled correctly.
#### Solution: Write-Ahead Logging (WAL)
- **Write-Ahead Logging**: Before updating the actual data blocks and parity, write-ahead logs are maintained to ensure that incomplete updates can be rolled back or completed during recovery.
- **Implementation**: WAL is implemented by writing the new data and parity information to a log before applying the updates to the RAID array.

---
### Conclusion
RAID provides a powerful mechanism for improving both the performance and reliability of disk storage systems by using multiple disks in parallel. Each RAID level offers different trade-offs between redundancy, performance, and storage efficiency. RAID's principles of data striping, mirroring, and parity are fundamental to modern storage solutions, and understanding these concepts is crucial for designing resilient storage architectures.

---
### Further Reading and References
- **Berkeley RAID Paper**: "A Case for Redundant Array of Inexpensive Disks (RAID)"
- **RAID Concepts**: Understanding RAID configurations and their applications in enterprise storage.
- **Write-Ahead Logging**: Techniques and implementation details in databases and file systems for ensuring data consistency.
These notes provide an in-depth technical exploration of RAID, offering detailed insights into its various levels, challenges, and solutions for crash recovery.