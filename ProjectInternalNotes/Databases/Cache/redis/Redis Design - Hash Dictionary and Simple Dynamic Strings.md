Redis uses several internal data structures to manage and store data efficiently. Among these structures, hash dictionaries and simple dynamic strings (SDS) are particularly significant.
## Redis Hash Dictionary
The hash dictionary is the main data structure in Redis. Essentially, Redis is a giant hash table storing all key-value pairs, regardless of their data type.

The time complexity of accessing the hash table is O(1), which is one of the reasons for Redis' speed. The hash value of each key is calculated to locate its corresponding hash bucket, which then leads to the desired data.

However, as more data is written into Redis, hash conflicts are inevitable - different keys may calculate the same hash value. Redis resolves these conflicts using **chained hashing**, where elements in the same bucket are stored in a linked list. 
```C
typedef struct dictEntry {
    void *key;
    void *val;
    struct dictEntry *next;
} dictEntry;
```
In the case where a linked list becomes too long, negatively impacting search performance, Redis uses two global hash tables to perform a **rehash operation** to increase the existing hash bucket count and reduce collisions. This operation consists of:
- Allocating more space to hash table 2
- Remapping and copying the data from hash table 1 to hash table 2
- Releasing the space in hash table 1
The remapping process is not performed all at once to avoid blocking Redis services. Instead, Redis uses **progressive rehashing** - with each client request, data from the first index in hash table 1 is copied to hash table 2, spreading the rehash operation across multiple requests.
## SDS - Simple Dynamic Strings
Redis uses the Simple Dynamic String (SDS) data structure to store string data. Compared to the C language string structure, SDS has several advantages:
- **O(1) time complexity to get string length**: SDS maintains a length attribute (`len`), which enables it to fetch the string length in O(1) time. In contrast, a C string needs to traverse the entire string, making it an O(n) operation.
- **Space pre-allocation**: Upon modification, SDS not only allocates space required for the string, but also allocates additional unused space, known as over-allocation. This pre-allocation of space optimizes future operations that may increase the length of the string.
- **Lazy space release**: When an SDS string is shortened, instead of reclaiming excess memory space, Redis marks it as unused with the `free` attribute. This space can be used directly in future append operations, reducing memory allocation overhead.
- **Binary safety**: Unlike C strings that terminate at '\0', SDS strings use the length attribute to denote the end of the string. This allows SDS to safely store binary data, which may contain '\0' characters.
```C
struct sdshdr {
    int len;
    int free;
    char buf[];
};
```
## zipList - Compressed List
Another important data structure in Redis is the ziplist, which is a compressed sequential data structure used to store lists with small amounts of data. Each ziplist entry can hold either an integer or a string value.
The ziplist has three header fields: `zlbytes` (total bytes used by the list), `zltail` (offset to the last item), and `zllen` (number of entries). It also includes a `zlend` field to denote the list's end.
```C
typedef struct ziplist<T> {
    int32 zlbytes; // Total bytes occupied by the ziplist
    int32 zltail_offset; // Offset to the last entry from the start, for quick access
    int16 zllength; // Number of entries
    T[] entries; // Entry content list, stored compactly
    int8 zlend; // End of ziplist marker, always 0xFF
} ziplist;
```
Finding the first and last elements in a ziplist is an O(1) operation, as their locations can be directly inferred from the header fields. However, finding other elements is less efficient - each entry must be searched one by one, resulting in an O(N) operation.

--------
## Double-Ended List
The Redis List data type is often utilized in situations such as queues and timeline lists of followers, similar to the functionality provided by a social media application like Weibo. Both FIFO queues and LIFO stacks can be easily managed by the double-ended list due to its inherent design.
### Key Characteristics
Redis' implementation of linked lists exhibits several distinct features:
1. **Double-ended**: Each node in the linked list holds pointers to both the previous and the next node. Therefore, retrieving the preceding or following node is an O(1) operation.
2. **Acyclic**: The list is not cyclic. The `prev` pointer of the head node and the `next` pointer of the tail node both point to `NULL`, signaling the end of the list during traversal.
3. **Head and tail pointers**: The list structure contains pointers to both the head and tail nodes. This allows the program to access these nodes directly in O(1) time.
4. **List length counter**: Redis uses the `len` attribute of the list structure to keep track of the number of nodes in the list. Again, this allows the program to find the length of the list in O(1) time.
5. **Polymorphism**: Each linked list node utilizes a `void*` pointer for storing node values. This enables the list to store various types of values by using the `dup`, `free`, and `match` attributes of the list structure for type-specific functions.
In later versions, Redis modified the list data structure, introducing Quicklist as a replacement for Ziplist and Linkedlist.
### Quicklist
Quicklist is a hybrid of Ziplist and Linkedlist. It breaks down the Linkedlist into segments. Each segment uses Ziplist for compact storage, and multiple Ziplists are then connected serially using two-way pointers. This design decision contributes significantly to Redis's high performance by balancing memory usage and speed of operations.
## SkipList
The sorted set data type in Redis is implemented through a data structure known as "Skip List".
A Skip List is an ordered data structure that allows for efficient node access by maintaining multiple pointers to other nodes within each node. The Skip List supports average O(logN) and worst-case O(N) complexity for node lookups and can handle sequential operations in batches.
Skip List adds a multi-level index to the linked list, which enables rapid data positioning through a few index jumps.
## Array of Integers (IntSet)
When a set only contains integer-valued elements and the number of elements in the set is not large, Redis uses the Integer Set as the underlying implementation of the set key.
The structure of an Integer Set is as follows:
```c
typedef struct intset {
     // Encoding
     uint32_t encoding;
     // Number of elements in the set
     uint32_t length;
     // Array for storing elements
     int8_t contents[];
} intset;
```
The `contents` array is the core of the Integer Set. Each element in the set is an item in the `contents` array, and all items are sorted from smallest to largest without any duplicates. The `length` attribute stores the number of elements in the Integer Set, that is, the length of the `contents` array.
## Reasonable Data Encoding
Redis uses objects (`redisObject`) to represent key values in the database. Whenever we create a key-value pair in Redis, at least two objects are created. One object represents the key, and the other represents the value.
```c
typedef struct redisObject {
   // Type
   unsigned type:4;
   // Encoding
   unsigned encoding:4;
   // Pointer to the underlying data structure
   void *ptr;
   //...
} robj;
```
The `type` field records the object type, including String, List, Hash, Set, and Sorted Set objects.
Different data types are encoded and transformed in the following ways:
1. **String**: If a number is being stored, Redis uses the int type encoding. If the value is not a number, raw encoding is used.
2. **List**: List objects can be encoded as Ziplists or Linkedlists. If the string length is less than 64 bytes and the number of elements is less than 512, Redis uses Ziplist encoding. Otherwise, it converts to Linkedlist encoding. These thresholds can be modified in `redis.conf`.
3. **Hash**: Hash objects can be encoded as Ziplists or Hashtables. If all key-value pairs in the Hash object have string lengths less than 64 bytes, and the total number of key-value pairs is less than 512, Redis uses Ziplist encoding. Otherwise, Hashtable encoding is used.
4. **Set**: Set objects can be encoded as Integer Sets (intset) or Hashtables. If all elements are integers and the number of elements is within a certain range, intset encoding is used. Otherwise, Hashtable encoding is used.
5. **Sorted Set (Zset)**: Zset objects can be encoded as Ziplists or Ziplists with Skip Lists (zkiplist). If the number of elements is less than 128 and all member lengths are less than 64 bytes, Ziplist encoding is used. Otherwise, it switches to zkiplist encoding.
## Single-Threaded Model
Redis operates on a single-threaded model for network I/O and key-value pair operations. Other activities, such as persistence, cluster data synchronization, and asynchronous deletion, are handled by separate threads.
### Why Single-Threaded?
While multi-threading can increase system throughput and better utilize CPU resources, it can also introduce overheads and complexities. For example, context switching between threads, data safety concerns during concurrent modifications, and the increase in code complexity.
Conversely, a single-threaded model offers several advantages:
1. No overhead due to thread creation.
2. Avoids CPU consumption due to context switching.
3. Eliminates concurrency issues such as locks, deadlocks, etc.
4. Simplifies code and processing logic.

While it may seem counter-intuitive to only use a single CPU core in today's multi-core world, it's important to remember that, because Redis operations are primarily memory-bound, CPU is rarely the bottleneck. The performance of Redis is more likely to be constrained by the amount of available memory or network bandwidth. As a result, the benefits of a simpler single-threaded model outweigh the potential advantages of multi-threading.

----
## I/O Multiplexing and Basic I/O Model
Redis utilizes an I/O multiplexing model to effectively handle multiple concurrent connections. It does this using an event framework implemented by epoll (Edge-triggered Poll) and its own structures. In this framework, various actions such as reading, writing, closing, and establishing connections are treated as events. The epoll multiplexing feature ensures that no time is wasted on I/O operations.
## Understanding Basic I/O Operations
Before discussing I/O multiplexing in-depth, it's essential to understand the basic steps involved in I/O operations.
A basic network I/O model will go through the following steps when processing a `GET` request:
1. **Accept connection from the client**: The server starts by accepting the client's connection request.
2. **Read request**: The server reads the incoming request from the client using a socket `recv`.
3. **Parse the request**: The server parses the request sent by the client.
4. **Execute instructions**: The server performs the `GET` operation as instructed by the client.
5. **Respond to client**: The server sends a response back to the client, essentially writing data back to the socket.
In these steps, `bind/listen`, `accept`, `recv`, `parse`, and `send` are all part of network I/O processing. The `GET` operation is a key-value data operation.
Redis is single-threaded, and in its simplest form, it executes the above operations sequentially in one thread. The challenge with this is that the `accept` and `recv` operations are blocking - if Redis is trying to establish a connection with a client but the connection cannot be established immediately, Redis will block at the `accept()` function, potentially causing other clients to fail to connect with Redis. Similarly, when Redis reads data from a client through `recv()`, if the data has not arrived, Redis will also block at `recv()`.
## I/O Multiplexing in Redis
I/O multiplexing allows multiple socket connections to be managed by a single thread. Three primary technologies facilitate this multiplexing: `select`, `poll`, and `epoll`. Among these, `epoll` is the most advanced and efficient.

The concept behind `epoll` is that the kernel, rather than the application, monitors the application's file descriptors. When a client operates, it creates sockets with various event types. On the server side, the I/O multiplexing module queues these messages, and the event dispatcher forwards them to different event handlers based on their types.

To illustrate, in a single-threaded Redis, the kernel consistently monitors socket connections for any connection request or data request. Once a request is received, it is passed to the Redis thread for processing. This allows a single Redis thread to handle multiple I/O streams.

`epoll` offers an event-based callback mechanism where corresponding event handlers are called for different events. This way, Redis is always processing events to optimize its response performance. The Redis thread will not block on a specific listening or connected socket. As a result, Redis can handle requests from multiple clients concurrently, improving its overall efficiency.
## The Essence of Redis Speed
The speed and efficiency of Redis can be attributed to several reasons:
1. **Pure memory operations**: Simple access operations are executed in memory, which is significantly faster than disk operations.
2. **Global hash table**: Redis utilizes a global hash table with a time complexity of O(1). To minimize the impact of hash collisions, Redis employs rehash operations to increase the number of hash buckets and reduce collisions. It uses a progressive rehash to avoid blocking caused by large one-time data remapping.
3. **Non-blocking I/O**: Redis uses I/O multiplexing, allowing a single thread to poll descriptors and convert database opening, closing, reading, and writing operations into events. This results in higher efficiency.
4. **Single-threaded model**: This model ensures the atomicity of each operation and minimizes thread context switching and contention.
5. **Optimized data structures**: Redis uses special data structures such as compressed tables for storing short data and skip lists for ordered data structures, increasing read speed.
6. **Dynamic encoding**: Redis selects different encodings based on the actual stored data type, further optimizing data storage and access.
---