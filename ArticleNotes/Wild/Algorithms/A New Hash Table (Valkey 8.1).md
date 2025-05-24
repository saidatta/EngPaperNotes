https://valkey.io/blog/new-hash-table/

**Date:** 2025-03-28  
**Author:** [Viktor Söderqvist](https://valkey.org)  
**Topic:** Hash Table Implementation & Performance Optimization in Valkey 8.1

## Table of Contents
1. [[#Key Takeaways|Key Takeaways]]
2. [[#Background-Theory|Background Theory]]
3. [[#Existing Implementation - The dict|Existing Implementation - The dict]]
4. [[#Goals and Motivation|Goals and Motivation]]
5. [[#Design of the New Hash Table|Design of the New Hash Table]]
6. 1. [[#Bucket Layout|Bucket Layout]]
7. 2. [[#Metadata Section|Metadata Section]]
8. 3. [[#Child Buckets for Overflow|Child Buckets for Overflow]]
9. 4. [[#Secondary Hash|Secondary Hash]]
10. [[#Implementation Details|Implementation Details]]
11. 1. [[#Incremental Rehashing|Incremental Rehashing]]
12. 2. [[#Scanning and SCAN Command|Scanning and SCAN Command]]
13. 3. [[#Random Element Sampling|Random Element Sampling]]
14. [[#Performance and Memory Usage|Performance and Memory Usage]]
15. [[#Code Examples and Snippets|Code Examples and Snippets]]
16. [[#Visualizations|Visualizations]]
17. [[#Discussion and Comments|Discussion and Comments]]
18. [[#References and Further Reading|References and Further Reading]]

---

## Key Takeaways
- **20–30 bytes saved** per key-value pair compared to the old design (dict).  
- Improved **CPU cache efficiency** by structuring data in 64-byte buckets.  
- **Incremental rehashing** still supported (required for large data sets).  
- **SCAN** iteration & **random element sampling** still preserved.  
- Gains in memory usage are substantial, especially for keys with TTL.  
- Potential **latency** improvements for small objects and large pipelining workloads.
---
## Background Theory
### Hash Tables & Memory Access
- A **hash table** maps keys to positions in memory (buckets or slots) via a hash function.
- **Cache lines** (commonly 64 bytes) are atomic chunks fetched by the CPU from main memory.
- Minimizing **pointer usage** and **cache misses** is crucial for high-performance data structures.
### Swiss Table Inspiration
- Swiss tables (e.g., Google’s [Abseil](https://abseil.io/) or LLVM’s [`SmallVector`](https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h)) place metadata closely with stored elements to reduce extra pointers and random memory accesses.
- Valkey’s new design: similarly groups multiple entries into a **single cache line** bucket.
---
## Existing Implementation - The dict
1. **Chained Hash Table**  
   - When multiple keys land in the same hash slot, they form a **linked list**.  
   - Each `dictEntry` has a `next` pointer → leads to extra memory overhead.  

2. **dict Memory Layout**  

   ```plaintext
   ┌─────────────────┐
   │  dict           │
   ├─────────────────┤
   │  table 0        │ -> array of pointers to dictEntry
   ├─────────────────┤
   │  table 1        │ -> used during rehash
   └─────────────────┘

   For each slot in table 0:
   ┌──────────────────────┐
   │  dictEntry           │
   ├──────────────────────┤
   │  key pointer         │
   ├──────────────────────┤
   │  value pointer       │
   ├──────────────────────┤
   │  next pointer        │ -> potential linked list next
   └──────────────────────┘
   ```
   
3. **Memory Access Cost**  
   - **4 memory reads** to lookup a single (key, value): 2 for `dictEntry`, 1 for key pointer, 1 for value.  
   - Each collision link adds extra reads.  
   - Each pointer costs **8 bytes** in a 64-bit system.

---

## Goals and Motivation
1. **Minimize Memory Access**  
   - Reduce random access and pointer chasing.  
   - Increase CPU cache hit rate.  
2. **Reduce Memory Usage**  
   - Save overhead by removing pointers.  
   - Possibly embed `key`, `value`, and metadata in the same structure.  
3. **Preserve Required Features**  
   - [[#Incremental Rehashing|Incremental Rehashing]]  
   - [[#Scanning and SCAN Command|SCAN iteration]]  
   - [[#Random Element Sampling|Random element sampling]]  

---

## Design of the New Hash Table
Valkey 8.1 introduces a **cache-line-optimized** hash table:

### Bucket Layout
- **One bucket = 64 bytes** (aligned to the CPU’s cache line size).
- Each bucket holds up to **7 elements** plus an 8-byte **metadata section**.
- If a bucket fills (7 items) and another item hashes to the same bucket:
  - A **child bucket** is allocated and linked.

#### Diagram: Primary Bucket

```plaintext
 ┌────────────────────────────┐
 │  Bucket (64 bytes total)   │
 ├─────────────┬──────────────┤
 │ Metadata (8)│ 7 Items (56) │
 └─────────────┴──────────────┘
```

Each **item** is effectively a `serverObject` that combines `key`, `value`, and additional fields (like TTL).

### Metadata Section
- **1 bit** indicates presence of a child bucket.  
- **7 bits** indicate whether each of the 7 slots is occupied.  
- **7 bytes** store the **secondary hash** (1 byte per stored element).  

```plaintext
Metadata (8 bytes):
[ ChildBucketFlag | 7 Occupancy Bits | 7 Secondary Hash Bytes ]
```

### Child Buckets for Overflow
- **Rarely** used if hash function distributes keys well.
- Each child bucket is also 64 bytes, with the **same** layout (metadata + up to 7 items).
- Potentially unbounded chain length, but in practice is short.

### Secondary Hash
- The main 64-bit hash is split:
  - Part used to index the bucket.  
  - The remaining **8 bits** (1 byte) used for quick comparison in the bucket.  
- During lookup:
  - **Compare the 1-byte secondary hash** before comparing full keys.  
  - Probability of false positive collision: **1/256**.  
- Greatly reduces the need for pointer chasing to the actual key.

---

## Implementation Details
### Incremental Rehashing
- **Resizing** the hash table gradually so the server remains responsive.
- Similar to old `dict` approach: operate on only a portion of the hash table per iteration.

### Scanning and SCAN Command
- **Iterators** remain valid even if resizing occurs.
- The new hash table design tracks visited buckets in a consistent manner:
  - Possibly a combination of top-level buckets + child buckets, ensuring no duplication or omission.

### Random Element Sampling
- Some commands (e.g., `RANDOMKEY`) rely on quick random sampling.
- The new design can:
  - Randomly pick a bucket.  
  - If child bucket is needed, traverse quickly if needed.  
  - Return the random item from a bucket slot.  

---

## Performance and Memory Usage
- **~20 bytes** saved per key-value pair, more for keys with TTL (~30 bytes).
- The overhead difference is visualized with two graphs in the blog post:
  1. **Memory overhead vs. value size**  
  2. **Memory overhead with TTL**  

### Observed Results
- **Zigzag pattern** in the graphs due to the memory allocator’s discrete allocation sizes.  
- For **extremely small** objects and large pipelines, improved CPU usage and latency.  
- Biggest impact: **reduced memory** → smaller cluster sizes or more data stored with the same memory.

---

## Code Examples and Snippets
Below are hypothetical C-style code snippets inspired by the description. They illustrate how one might define the new bucket structure and how a lookup could be performed. These are **not** the exact production code from Valkey but serve as a reference.

### Bucket Structure

```c
#include <stdint.h>
#include <stdbool.h>

#define BUCKET_SIZE 64
#define MAX_ENTRIES 7

typedef struct serverObject {
    // Embedded key and value:
    // In actual code, key could be a string struct, 
    // value could be a pointer to data or inline data.
    // For demonstration:
    uint64_t key_hash;
    void* value;
    uint64_t expire_time; // e.g., for TTL
    // Additional flags, references, etc.
} serverObject;

typedef struct bucketMeta {
    // 1 bit: has_child; 7 bits: occupancy
    // 7 bytes: secondary hash array
    // For simplicity, store them in separate fields
    bool has_child;
    uint8_t occupancy; // lower 7 bits used for occupancy
    uint8_t sec_hashes[MAX_ENTRIES]; // 7 secondary hashes
} bucketMeta;

typedef struct hashBucket {
    bucketMeta meta;
    // 7 serverObjects inlined
    serverObject items[MAX_ENTRIES];
    // If has_child == true, pointer to overflow
    struct hashBucket* child;
} hashBucket;
```

### Lookup Function (Simplified)

```c
serverObject* lookup(hashBucket* table, size_t capacity, uint64_t hash, /*some hashing params*/ const void* key_data) {
    // 1. Calculate bucket index
    size_t index = hash % capacity;
    hashBucket* b = &table[index];
    
    while (true) {
        // 2. Check each occupied slot in bucket
        for (int i = 0; i < MAX_ENTRIES; i++) {
            if (((b->meta.occupancy >> i) & 0x1) == 1) {
                // Compare secondary hash first
                if (b->meta.sec_hashes[i] == (uint8_t)(hash >> (64 - 8))) {
                    // Possible match, compare full key if needed
                    // In real code: compare actual key string or ID
                    // Placeholder: compare hash or more
                    if (b->items[i].key_hash == hash) {
                        // Found
                        return &b->items[i];
                    }
                }
            }
        }
        // 3. If no match but has child, continue
        if (b->meta.has_child && b->child != NULL) {
            b = b->child;
        } else {
            // Not found
            return NULL;
        }
    }
    
    return NULL; // unreachable
}
```

### Insert Function (Simplified Pseudocode)

```c
bool insert(hashBucket* table, size_t capacity, uint64_t hash, serverObject new_obj) {
    size_t index = hash % capacity;
    hashBucket* b = &table[index];
    
    while (true) {
        // Attempt to find an empty slot
        for (int i = 0; i < MAX_ENTRIES; i++) {
            if (((b->meta.occupancy >> i) & 0x1) == 0) {
                // Occupy the slot
                b->items[i] = new_obj;
                b->meta.sec_hashes[i] = (uint8_t)(hash >> (64 - 8));
                // Set bit in occupancy
                b->meta.occupancy |= (1 << i);
                return true;
            }
        }
        // If full, allocate a child if not present
        if (!b->meta.has_child) {
            // allocate child
            b->child = allocate_new_bucket();
            b->meta.has_child = true;
        }
        // Move on to child bucket
        b = b->child;
    }
    // Should never fail in theory
    return false;
}
```

---

## Visualizations
1. **Old vs. New Memory Layout**  
   ```mermaid
   flowchart LR
       A[Old 'dict'] --> B
       B[dictEntry] --> C[key pointer] & D[value pointer] & E[next pointer]
       A --> F[New 'bucket (64B)']
       F --> G[Metadata (8B)]
       F --> H[7 items (serverObject embedded)]
       F --> I[Possible child pointer]
   ```

2. **Bucket Overflow**  
   ```mermaid
   graph LR
       subgraph Parent Bucket
       PB[Bucket 64B] --> Items(7 items)
       PB --> CB_Flag(Child Bit Set)
       end
       PB --> OB[Child Bucket 64B]
   ```

3. **Secondary Hash Filtering**  
   - A short example table to show how the one-byte partial hash filters keys:

   | Key Hash (64 bits)   | Bucket Index (X bits) | Secondary (8 bits) |
   |----------------------|-----------------------|--------------------|
   | 0xDEAD_BEEF_1234_5678| 0x567                | 0x78               |
   | 0xAA55_00FF_0000_1111| 0x111                | 0x11               |

---

## Discussion and Comments

> **Comment 1**: *“Why not just use an existing Swiss table library?”*  
> **Answer**: The Valkey team needed specialized features: incremental rehashing, SCAN iteration, random sampling. These are typically not provided out-of-the-box by Swiss table libraries.

> **Comment 2**: *“What happens if we get extremely large overflow chains?”*  
> **Answer**: In principle, unbounded chain length is possible but extremely unlikely with a well-distributed hash function. Collision probability is minimized.

> **Comment 3**: *“Is this design final or will there be further refinements?”*  
> **Answer**: This design is part of Valkey 8.1; future refinements are always possible, but the big memory improvements are already realized.

> **Comment 4**: *“How does random sampling handle buckets with child chains?”*  
> **Answer**: The system may randomly pick a top-level bucket, and if it has a child bucket, it can decide to either pick from the parent or child with uniform probability, or move to the child as needed.

---

## References and Further Reading
- **Valkey Official Docs**: [valkey.org](https://valkey.org/)  
- **Swiss Table**: [Abseil’s Open Source Swiss Table](https://abseil.io/about/design/swisstables)  
- **CPU Cache and Performance**:  
  - “What Every Programmer Should Know About Memory,” Ulrich Drepper.  
  - [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html).  
- **Incremental Rehashing Techniques**: [Redis Implementation](https://redis.io/topics/rehashing).  

---

## Footnotes / MOC Link
- Navigate to the [Valkey 8.1 Release Notes](valkey-8_1-release.md) for a deeper feature-by-feature breakdown.  
- For broader architectural topics, see [[Valkey Architecture Overview]] in your vault.  

---

**End of Notes**