https://redis.io/docs/latest/commands/object-encoding/
https://juejin.cn/post/7173667782725206047
---
### **Overview: Redis Common Data Types**
Redis supports nine primary data types:  
- **String**  
- **Hash**  
- **List**  
- **Set**  
- **Zset (Sorted Set)**  
- **BitMap**  
- **HyperLogLog**  
- **GEO**  
- **Stream**  

Each of these types is optimized for different use cases and underlying storage mechanisms. This guide provides a detailed dive into the data structures, their applications, and underlying mechanisms.

---
## **1️⃣ String**

### **Underlying Data Structures**
- **SDS (Simple Dynamic String)** is used instead of C’s native `char[]` for storing strings to avoid buffer overflows, improve performance, and offer safe dynamic resizing.
  
### **Why not use char[]?**
- **O(N)** complexity for `strlen()` in `char[]` vs **O(1)** in SDS.
- Prone to **buffer overflows** during unsafe operations like concatenation in C.

### **SDS Structure Fields**
```c
struct SDS {
    int len;         // length of the string
    int alloc;       // total allocated space
    char flags;      // stores the type of SDS
    char buf[];      // the actual data
};
```

#### **Advantages of SDS**
1. **Constant-time length retrieval** (O(1)).
2. **Pre-allocated space** allows growth without frequent reallocations.
3. **Embstr encoding** optimizes space allocation for small strings by keeping `RedisObject` and `SDS` in contiguous memory.

#### **Memory Layout: Embstr vs Raw Encoding**![[Screenshot 2024-09-23 at 11.33.14 AM.png]]

- **Embstr**: Used for small strings (under 44 bytes in Redis 5.0). Efficient due to single allocation.
- **Raw**: Used for large strings. Requires two memory allocations but is mutable.

![[Screenshot 2024-09-23 at 11.33.40 AM.png]]
![[Screenshot 2024-09-23 at 11.33.47 AM.png]]
![[Screenshot 2024-09-23 at 11.33.52 AM.png]]
#### **Application Scenarios**
- **Counters**: Redis’s single-threaded execution ensures atomic updates, making it ideal for counting operations (e.g., page views).
- **Session Sharing**: Distributed systems can store user session data centrally in Redis.
  
---

## **2️⃣ List**

### **Underlying Data Structures**
1. **Compressed List**: Used when elements are small and few (<512 elements, <64 bytes each).
	1. ![[Screenshot 2024-09-23 at 11.37.33 AM.png]]
2. **Quicklist**: Introduced in Redis 3.2 to avoid performance issues like chain updates in compressed lists.![[Screenshot 2024-09-23 at 11.37.54 AM.png]]
![[Screenshot 2024-09-23 at 11.40.33 AM.png]]
#### **Quicklist Structure**
```plaintext
+------------+    +-------------+    +-------------+
| Ziplist 1  | -> | Ziplist 2   | -> | Ziplist 3   | -> NULL
+------------+    +-------------+    +-------------+
```

- **Compressed Lists**: Arrays with some fields for metadata. Compact but slow for large data.
- **Quicklist**: Combines a linked list of compressed lists, reducing overhead from resizing.

#### **Application Scenarios**
- **Message Queues**: Using `LPUSH` (push) and `RPOP` (pop) creates a simple queue.
- **Blocking Reads**: Use `BRPOP` for blocking reads. This can be combined with `BRPOPLPUSH` to ensure message reliability.

---

## **3️⃣ Hash**

### **Underlying Data Structures**
1. **Listpack**: Replaces compressed lists from Redis 7.0. Optimized for small hashes.
2. **Hash Table**: Used for larger datasets.

#### **Listpack Structure**
```plaintext
+---------------------+
| len  | data  | next  |
+---------------------+
```

**Listpack vs. Compressed Lists**:
- Avoids chain updates by discarding the `prevlen` field, leading to faster memory operations.
#### **Application Scenarios**
- **Caching Objects**: Hashes are more efficient for objects with frequently updated fields (e.g., shopping carts).
- **User Profiles**: A user’s profile attributes can be stored as fields in a hash.

---

## **4️⃣ Set**

### **Underlying Data Structures**
1. **Integer Set**: Used for small sets with integer elements.
2. **Hash Table**: Used when the number of elements exceeds a threshold.

#### **Integer Set**
```c
struct intset {
    uint32_t encoding;  // int16, int32, int64
    uint32_t length;    // number of elements
    int8_t contents[];  // actual elements
};
```

#### **Application Scenarios**
- **Like/Dislike Systems**: A user can only "like" a post once, leveraging set uniqueness.
- **Common Follows**: Use intersection operations to find mutual follows between users.

---

## **5️⃣ Zset (Sorted Set)**

### **Underlying Data Structures**
1. **Skip List**: For efficient range queries.
2. ![[Screenshot 2024-09-23 at 11.39.58 AM.png]]
3. **Hash Table**: For direct element access.

#### **Skip List Structure**
```c
struct zskiplistNode {
    double score;          // element's score
    struct zskiplistNode *forward[];  // next node at each level
};
```

- **Efficient Range Queries**: Skip lists allow logarithmic time range queries, making them ideal for ranked data.

#### **Skip List Visualization**
```plaintext
Level 2:  [Header] --+-----------------> [Node 4] ---> NULL
Level 1:  [Header] -> [Node 1] -----> [Node 4] -----> NULL
Level 0:  [Header] -> [Node 1] -> [Node 2] -> [Node 3] -> [Node 4] -> NULL
```

#### **Application Scenarios**
- **Leaderboards**: Scores from games or apps are stored with a ranking.
- **Time-Based Sorting**: Use timestamps as scores for efficient sorting of time-sensitive data.

---

## **6️⃣ BitMap**

### **What is BitMap?**
- **Bit-level storage**: Allows the use of individual bits to track states (e.g., 1 for active, 0 for inactive).
  
#### **Application Scenarios**
- **Sign-in Tracking**: Track user sign-ins across 365 days, using 1 bit per day.

```plaintext
Day 1:  1
Day 2:  0
Day 3:  1
...
```

- **User Status**: Monitor login statuses with a single bit per user.

---

## **7️⃣ HyperLogLog**

### **Approximate Cardinality Estimation**
- Uses probabilistic counting for estimating unique elements in large datasets with an error margin of 0.81%.

#### **Application Scenarios**
- **Counting Visitors (UV)**: Efficiently track millions of unique page views using minimal memory.

---

## **8️⃣ GEO**

### **Underlying Principle**
- Uses **Zset** to store geographic data. Latitude and longitude are encoded into a 52-bit integer score.
  
#### **Application Scenarios**
- **Finding Nearby Locations**: Use `GEORADIUS` to find points within a radius.
  
---

## **9️⃣ Stream**

### **Message Queuing with Consumer Groups**
- Streams in Redis provide robust message queuing, consumer groups, and message acknowledgment.

#### **Message IDs**
- Generated as `timestamp + counter`, ensuring global uniqueness.

#### **Application Scenarios**
- **Event Processing**: Using streams for event-driven architectures where each event is processed only once by a group.

---

## **Conclusion**

Understanding Redis’s underlying data structures helps make more informed decisions about which types to use for specific use cases. Whether implementing leaderboards, message queues, or unique visitor counts, Redis's versatile data types provide efficient, scalable solutions.