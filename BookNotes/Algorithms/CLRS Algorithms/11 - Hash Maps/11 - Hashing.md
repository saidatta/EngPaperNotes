#### Overview
- Hash tables implement dynamic sets supporting dictionary operations: INSERT, SEARCH, and DELETE.
- Effective for compiler symbol tables where keys are character strings.
- Practical efficiency: O(1) average time for element search.
#### Comparing Hash Tables and Linked Lists
- Worst-case for both: Θ(n) time.
- Under reasonable assumptions, hash table searches are much faster.
#### Python's Implementation
- Python dictionaries use hash tables.
#### 11.1 Direct-address Tables
##### Concept
- A direct-address table is a data structure that uses an array to store elements with unique keys. 
- The key of an element is used as the index of the array, so that the element can be accessed in constant time. A direct-address table can perform insert, search, and delete operations in O(1) time, but it has some limitations.
- The keys must be integers or can be converted to integers.
- The range of keys must be relatively small, otherwise the array will be too large and waste memory.
- The keys must be distinct, otherwise there will be collisions.
- Suitable for small key universes.
- Direct addressing uses an array, T[0 : m − 1], where m is the size of the key universe.
##### Operations
- **SEARCH**: `DIRECT-ADDRESS-SEARCH(T, k)` returns `T[k]`.
- **INSERT**: `DIRECT-ADDRESS-INSERT(T, x)` sets `T[x.key] = x`.
- **DELETE**: `DIRECT-ADDRESS-DELETE(T, x)` sets `T[x.key] = NIL`.
##### Example: Key Universe U = {0, 1, ..., 9}
- For set K = {2, 3, 5, 8}, corresponding slots in T are non-NIL.
- Other slots contain NIL.
#### Key Concepts
- Hashing generalizes direct addressing.
- Collisions: More than one key maps to the same index.
- Handling Collisions: Chaining and Open Addressing.
#### 11.2 Chaining
- Resolve collisions by linking elements at the same index.
- Hash table T with slots T[0 ... m-1].
- If several elements hash to the same slot, they form a linked list.
#### 11.3 Hash Functions
- Converts a key into an array index.
- Goal: Uniform distribution of keys.
#### 11.4 Open Addressing
- Handles collisions by probing for the next available slot.
- Variants: Linear probing, quadratic probing, double hashing.\
#### 11.5 Performance in Hierarchical Memory
- Design hash tables for efficient access in modern memory systems.

---
### Code Examples
#### Direct Address Table Implementation (Python)
```python
class DirectAddressTable:
    def __init__(self, size):
        self.table = [None] * size

    def search(self, key):
        return self.table[key]

    def insert(self, key, value):
        self.table[key] = value

    def delete(self, key):
        self.table[key] = None
```

#### Hash Function Example
```python
def simple_hash_function(key, table_size):
    return key % table_size
```
### Equations
- **Average Search Time**: O(1) under ideal conditions.
---