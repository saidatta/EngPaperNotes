### Objective

Make full use of the performance of modern storage SSDs, and significantly reduce the read and write amplification of LSMTree to improve its performance while providing the same API.

### Background

On traditional disks, the performance of sequential IO is about 100 times that of random IO. Based on this, LSMTree realizes the random read and write of massive KV as random read and write of memory + sequential flashing + regular consolidation (compact) to improve read and write Performance, especially suitable for scenarios **where there are more writes than reads** and **the timeliness is relatively strong (data is most frequently accessed recently).**

![wisckey-lsm-tree.png](https://i.loli.net/2020/03/21/IetKyMmGNQVrBsS.png)

_Author: Wooden Bird Miscellaneous Notes [https://www.qtmuniao.com/2020/03/19/wisckey/](https://www.qtmuniao.com/2020/03/19/wisckey/) , please indicate the source for reprinting_

**Read and write amplification** . For write amplification, since LSMTree has many layers, in order to speed up the reading speed, it is necessary to continuously perform merge sorting to compact, resulting in each KV being read and written multiple times. For read amplification, multi-layer query is required to find the specified key in the vertical direction, and binary query is required in the horizontal direction because there are multiple key ranges in the same layer. Of course, in order to speed up the search, you can configure a [bloom filter for each layer](https://en.wikipedia.org/wiki/Bloom_filter), to quickly skip when the key to be looked up does not exist in this layer. As the amount of data increases, read and write amplification will be even greater.

![wisckey-rw-amplification.png](https://i.loli.net/2020/03/21/GeaXn8tA51T2LKp.png)

Nowadays, the price of SSD is getting lower and the scale of use is getting bigger and bigger, and the parallel random read performance of SSD is very good, which is not so much different from sequential read. Of course, random writing should be avoided as much as possible, because it is not as fast as random reading, and it will reduce the life of SSD.

![wisckey-ssd.png](https://i.loli.net/2020/03/21/MbcStji3UQs1a4N.png)

## [](https://www.qtmuniao.com/2020/03/19/wisckey/#%E6%A0%B8%E5%BF%83%E8%AE%BE%E8%AE%A1 "core design")core design

The core design of WiscKey mainly has the following four items:

1.  The key and value are stored separately, the Key still exists in the LSM-tree, and the Value exists in an additional log file (vLog).
2.  For out-of-order value data, use SSD parallel random read to speed up the read speed.
3.  Use unique crash consistency and garbage collection strategies to efficiently manage Value log files.
4.  Remove WAL without affecting consistency, improving write performance for small data traffic.

## [](https://www.qtmuniao.com/2020/03/19/wisckey/#%E8%AE%BE%E8%AE%A1%E7%BB%86%E8%8A%82 "plan the details")plan the details

### [](https://www.qtmuniao.com/2020/03/19/wisckey/#%E9%94%AE%E5%80%BC%E5%88%86%E5%BC%80 "key value separated")key value separated

The Key is still stored in the LSM-tree structure. Because the space occupied by the Key is usually much smaller than that of the Value, the number of layers of the LSM-tree in WiscKey will be small, and there will not be too much read and write amplification. Store Value in an additional log file

In it, it is called vLog. Of course, some meta information of Value, such as the position information of Value in vLog, will be stored in the LSM-tree along with the Key, but it occupies a small space.

![wisckey-architecture.png](https://i.loli.net/2020/03/21/Q6IjovGPFybNHOA.png)

**read** . Although the Key and Value need to be read separately (that is, a read needs to be decomposed into a memory (high probability) lookup in the LSM-tree and a random lookup on the SSD), because the speed of both is compared to the original layer-by-layer lookup. block, it will not take more time than LevelDB.

**write** . First append Value to vLog to get its offset vLog-offset in vLog. Then write the Key and `<vLog-offset, value-size>`together into the LSM-tree. An append operation, a memory write operation, are fast.

**delete** . Using the asynchronous deletion strategy, only the key in the LSM-tree is deleted, and the Value in the vLog will be recycled by the regular garbage collection process.

Despite the above advantages, the separation of Key Value also brings many challenges, such as Range Query, garbage collection, and consistency issues.

### Challenge 1: Range queries

Range query (Range Query, traversing KV-Pair in order by specifying the start and end keys) is a very important feature of contemporary KV storage. The key-value pairs in LevelDB are stored in the order of Key, so range query can be performed by sequentially traversing related Memtable and SSTable. But the Value of WiscKey is unordered, so a large number of random queries are required. However, as shown in Figure 3, we can use multi-threaded parallel random query to fill up the SSD bandwidth and greatly improve the query speed.

Specifically, when performing a range query, first load the required Keys sequentially in the LSM-tree, and then use SDD's multi-threaded random read to pre-read and place them in the Buffer, so that the read Keys and buffers can be sequentially combined The Value in is returned to the user to achieve high performance.

### [](https://www.qtmuniao.com/2020/03/19/wisckey/#%E6%8C%91%E6%88%982%EF%BC%9A%E5%9E%83%E5%9C%BE%E5%9B%9E%E6%94%B6 "Challenge 2: Garbage Collection")Challenge 2: Garbage Collection

LevelDB uses the compact mechanism for delayed garbage collection, and the same mechanism is used for Key recovery in WiscKey. But for Value, due to its existence in vLog, additional garbage collection mechanism needs to be considered.

The simplest and rude way is to scan the LSM-tree structure in WiscKey first to obtain the Key collection in use; then scan the vLog to recycle all Values ​​that are not referenced by the Key collection. But obviously, this is a heavy (and time-consuming) operation. In order to maintain consistency, it may be necessary to stop providing external services, similar to stop-the-world in the early JVM GC.

And we obviously need a more lightweight approach. WiscKey's approach is ingenious. Its basic idea is to treat all the Value data in the vLog as a strip, maintain all the data in use **in** the middle of the strip, and use two pointers to mark the beginning and end of the valid data area in the middle. The head (head) can only perform append operations, and the tail (tail) can be garbage collected. So how do we maintain this valid intermediate data area?

![wisckey-gc.png](https://i.loli.net/2020/03/21/y2OaUcY9bnRS6PD.png)

When garbage collection is required, read a block of data from the end (Block, containing a batch of data entries, each data entry contains `<ksize, vsize, key, value>`four fields, each time a piece is read to reduce IO) into memory; for each data entry , if it is in use, append it to the vLog stripe head; otherwise discard it; then move the tail pointer (tail) to skip this data.

The tail pointer is a critical variable that needs to be persisted in case of downtime. The method of WiscKey is to reuse the LSM-tree structure for storing Keys, and use a special Key ( `<‘‘tail’’, tail-vLog-offset>`) to store it together with Keys. The head pointer is the end of the vLog file and does not need to be saved. In addition, WiscKey's garbage collection timing can be flexibly configured according to the situation, such as regular collection, collection when a certain threshold is reached, collection when the system is idle, and so on.

### Challenge 3: Crash Consistency

When the system crashes, LSM-tree usually provides guarantees such as the atomicity of KV insertion and the order of recovery. WiscKey can also provide the same consistency, but since the key and value are stored separately, the implementation mechanism is slightly more complicated (at least atomicity will be more difficult).

For **the atomicity** of data insertion , we consider the following situations. After recovery from downtime, when a user queries a Key,

1.  If it cannot be found in the LSM-tree, the system assumes it does not exist. Even though the Value may have been appended to the vLog, it will be recycled later.
2.  If it can be found in the LSM-tree, check the data entry in the corresponding vLog `<ksize, vsize, key, value>`, and check in turn whether the entry exists, whether the location is in the middle legal segment, and whether the Key can match. If not, delete the Key and tell the user that it doesn't exist. In order to prevent the data from being hung up after writing only half of it, resulting in incomplete data entries, a checksum can also be added to the data entries.

Through the above process, we can guarantee the atomicity of KV writing: for users, KV either exists or does not exist.

**For the sequence** of data insertion , contemporary file systems (such as ext4, btrfs, xfs) guarantee the sequence of appending, that is, if data entries D1, D2, D3 ... Dx, Dx+1, ... are sequentially appended in the vLog If Dx is not appended to the vLog when the system is down, no subsequent data entries will be appended to the system. In this way, the sequence of data insertion can be guaranteed.

As will be mentioned below, in order to improve the appending efficiency of small-sized Value, WiscKey uses write Buffer. Therefore, some data entries may be lost during downtime. For this reason, WiscKey provides a synchronous write switch to allow WiscKey to abandon the Buffer, force the Value to be written to the vLog file, and then write the corresponding Key into the LSM-tree.

### Optimization 1: vLog cache

For intensive, small-size write traffic, if the user `put(K, V)`invokes `write`the source and appends a data entry to the vLog, such frequent IO will lead to poor performance and cannot fully utilize the SSD bandwidth, as shown in the following figure:

![wisckey-buffer-write.png](https://i.loli.net/2020/03/21/DxZYFRzunNrOpgB.png)

Therefore, WiscKey uses a Buffer to cache the written Value, and it is actually appended to the vLog only when the user requests or reaches the set size threshold. You also need to make some modifications when querying. For each query, you must first search in the buffer, and then search in the vLog. But the cost of doing this is, as mentioned above, when the system crashes, this part of the unflushed data in the buffer will be lost.

### Optimization 2: Save WAL

WAL, Write Ahead Log, pre-write log, is a data recovery mechanism commonly used in database systems. In the traditional LSM-tree, since the data is directly written into the memory, a log is recorded before each operation for downtime recovery; during downtime recovery, the log is read one by one to restore the data structure in the memory. But in this way, each write request increases the disk IO, thereby reducing the system write performance.

Since the data entries in the vLog record all inserted Keys in sequence, the vLog can be reused as the WAL of the LSM-tree in WiscKey. As an optimization, you can `<‘‘head’’, head-vLog-offset>`also . When recovering from a downtime, first obtain this point, and then read the Key in the vLog one by one from this point to restore the LSM -tree.