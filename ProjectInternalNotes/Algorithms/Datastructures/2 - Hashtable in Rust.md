## 1. Overview of the Data Structure

This code implements a **hash table** with **separate chaining** collision resolution. Specifically:
- The `HashTable<K, V>` struct maintains:
  - A `Vec<LinkedList<(K, V)>>` called `elements`. Each entry in `elements` is a **bucket**, implemented as a linked list storing `(key, value)` pairs.
  - A `count` of how many total `(key, value)` pairs have been inserted.

- The `hash` function is defined by a custom trait `Hashable` instead of using Rust’s standard library `Hash` trait.  
- **Collisions** (cases where different keys map to the same bucket index) are resolved by **appending** `(K, V)` pairs to the corresponding linked list at that index.

---

## 2. The `Hashable` Trait

```rust
pub trait Hashable {
    fn hash(&self) -> usize;
}
```

- Any key `K` must implement `Hashable` so that the hash table can compute a `usize` hash value for it.  
- In typical Rust code, one would use the standard library’s `Hash` trait and a hasher (e.g., `std::collections::hash_map::DefaultHasher`), but here we do it manually.  
- The code also requires `K: PartialEq` so we can check if two keys are the same when searching or inserting.

---
## 3. Initializing the Hash Table
```rust
impl<K: Hashable + PartialEq, V> HashTable<K, V> {
    pub fn new() -> HashTable<K, V> {
        let initial_capacity = 3000;
        let mut elements = Vec::with_capacity(initial_capacity);

        for _ in 0..initial_capacity {
            elements.push(LinkedList::new());
        }

        Self { elements, count: 0 }
    }
}
```

- The hash table starts with `3000` buckets—each represented by an empty `LinkedList<(K, V)>`.  
- Setting an **initial capacity** attempts to reduce early resizes.  
- Each bucket is a `LinkedList` so we can easily append new `(K, V)` pairs.

When the table is created:
- `self.elements` is a `Vec<LinkedList<(K, V)>>` of length `3000`.  
- `self.count` (number of stored entries) is `0`.

This also implements `Default` by calling `HashTable::new()`, so one can write `HashTable::<K, V>::default()` if desired.

---

## 4. Insert Operation

```rust
pub fn insert(&mut self, key: K, value: V) {
    if self.count >= self.elements.len() * 3 / 4 {
        self.resize();
    }
    let index = key.hash() % self.elements.len();
    self.elements[index].push_back((key, value));
    self.count += 1;
}
```

1. **Load Factor Check & Resize**  
   - Before inserting, the code checks if `count >= self.elements.len() * 3 / 4`.  
   - If the table is at least **75% full**, we call `resize()`. This is a heuristic that tries to keep the average chain length short, ensuring performance remains near O(1).  

2. **Computing the Bucket Index**  
   - We compute `index` as `key.hash() % self.elements.len()`.  
   - This ensures the index is within the current capacity of `elements`.  

3. **Appending the New (K, V) to the Bucket**  
   - We do `self.elements[index].push_back((key, value));`.  
   - The new pair is added **without checking if the key already exists** in that bucket.  
     - This means **duplicates** can exist if the same key is inserted multiple times. The **first** matching key found during a search will be returned, effectively ignoring subsequent duplicates.  

4. **Increment `count`**  
   - Finally, `self.count` is incremented by 1.

### 4.1 Potential Discussion Point: Overwriting Keys

- Because we just append `(key, value)` blindly, if the **same key** is inserted again, we store **another** `(key, value)` pair in the chain. The `search` method returns the first one found, so the second is effectively ignored.  
- A more typical hash table might replace the existing value for the same key. That would require a small loop to check each `(k, v)` in the chain first, and if `k == key`, update `v`.

---

## 5. Search Operation

```rust
pub fn search(&self, key: K) -> Option<&V> {
    let index = key.hash() % self.elements.len();
    self.elements[index]
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}
```

1. **Compute the Bucket Index**  
   - Again, `index = key.hash() % self.elements.len()`.  

2. **Iterate Over the LinkedList**  
   - We call `.iter()` on the linked list and use `.find(...)` to locate the **first** matching `(k, v)` pair where `k == key`.  
   - `PartialEq` ensures we can do `*k == key` for equality checking.  

3. **Return Reference to Value**  
   - The `.map(|(_, v)| v)` ensures that if we find a match, we return a reference to the value `&V`. If no match is found, we return `None`.

Thus, **search** is a linear scan within the bucket. If the hash function is good and the table is well-sized, the average chain length is short (expected O(1) time). Worst case, if all elements are in one bucket, search time is O(n).

---

## 6. Resizing Logic

```rust
fn resize(&mut self) {
    let new_size = self.elements.len() * 2;
    let mut new_elements = Vec::with_capacity(new_size);

    for _ in 0..new_size {
        new_elements.push(LinkedList::new());
    }

    for old_list in self.elements.drain(..) {
        for (key, value) in old_list {
            let new_index = key.hash() % new_size;
            new_elements[new_index].push_back((key, value));
        }
    }

    self.elements = new_elements;
}
```

1. **Calculate New Size**  
   - `new_size = self.elements.len() * 2`. The capacity **doubles**.  

2. **Create New Vector of Empty LinkedLists**  
   - We build a fresh `Vec<LinkedList<_>>` of length `new_size`.  

3. **Rehash Old Elements**  
   - We call `self.elements.drain(..)`, which **empties** the old vector, returning each `LinkedList<(K, V)>`.  
   - We iterate over each old linked list, and for each `(key, value)` pair, we compute `new_index = key.hash() % new_size` and push it into `new_elements[new_index]`.  

4. **Swap In the New Vector**  
   - Finally, `self.elements = new_elements;`.  
   - Note that `self.count` remains the same. We have just changed the bucket structure.

### Why Rehash?

- Once the capacity changes, `index = hash % new_capacity` changes. Each key must be placed into the correct new bucket.  
- This rehashing ensures good distribution and preserves correctness with the new capacity.

---

## 7. Test Suite

The code has a series of unit tests in `#[cfg(test)] mod tests`:

1. **`test_insert_and_search`**  
   - Inserts a `(key=TestKey(1), value=TestKey(10))` and asserts that searching for `TestKey(1)` returns `Some(&TestKey(10))`.

2. **`test_resize`**  
   - Inserts enough items (`initial_capacity * 3/4 + 1`) to force a resize (because of the load factor check) and asserts that the **capacity** (`hash_table.elements.capacity()`) has indeed increased.

3. **`test_search_nonexistent`**  
   - Verifies that searching for a key that was never inserted yields `None`.

4. **`test_multiple_inserts_and_searches`**  
   - Inserts a small range of `(key, value)` pairs and confirms we can retrieve each correct value.

5. **`test_not_overwrite_existing_key`**  
   - Demonstrates the code does **not** overwrite if the same key is inserted with a new value. It asserts that the search returns the original value.  
   - This matches the logic in `insert` (no check for existing keys; it just appends).  
   - The `.find(...)` in `search` will return the **first** matching `(key, value)` in the chain, so the “new” value is never used.

6. **`test_empty_search`**  
   - Creates an empty hash table and confirms that searching for any key returns `None`.

Overall, these tests verify both correctness (can find inserted items) and behavior when the table is near the load factor limit.

---

## 8. Complexity Analysis

1. **Insert**  
   - Average case: **O(1)** if the hash function distributes keys evenly (constant chain length).  
   - Worst case: **O(n)** if every key collides in the same bucket, so we traverse the entire linked list.

2. **Search**  
   - Same as Insert: Average case **O(1)**, worst case **O(n)**.

3. **Resize**  
   - Occurs when load factor > 0.75, i.e., \(\frac{\text{count}}{\text{capacity}} \ge 0.75\).  
   - Resizing is **O(n)** since we must rehash every key.  
   - Over the long run, **amortized** cost is near O(1) per insert if the load factor is maintained below a constant threshold, because resizes are infrequent.

---

## 9. Potential Improvements / Considerations

1. **Key Overwrite**  
   - If you want a behavior where inserting the same key updates the existing value, you would search the chain for the key first and replace the value if found (or append if not).

2. **Handling Larger Keys**  
   - This design uses a custom `Hashable` trait. In a real-world Rust application, you’d typically use `std::hash::Hash` plus a hasher like `DefaultHasher` or a specialized hasher (e.g., `FxHasher`, `AHasher`, etc.).  

3. **Open Addressing vs. Separate Chaining**  
   - Here we use **separate chaining** with a `LinkedList`. Alternatively, one can use **open addressing** (linear probing, quadratic probing, etc.), which can be more cache-friendly.  
   - Rust’s standard `HashMap` typically uses a specialized bucket array for better performance.  

4. **Collision Security**  
   - With a simple `hash()` function, maliciously chosen inputs can degrade performance. Real implementations often use randomization or robust hash functions to mitigate collisions.

5. **Memory Footprint**  
   - Using `LinkedList` can be suboptimal in terms of memory overhead and cache locality. A `Vec<(K, V)>` or `VecDeque<(K, V)>` might be more performant in many cases.

---

## Summary

In essence:

- **Core Idea**: We store `(key, value)` pairs in `elements[bucket_index]` where `bucket_index = hash(key) % capacity`.  
- **Collisions**: We chain them in a linked list.  
- **Load Factor**: Once we exceed 75% usage, we **resize** by doubling the number of buckets and rehash everything.  
- **Insertion**: We simply append `(key, value)` to the appropriate linked list.  
- **Lookup**: We iterate through the chain to find the first matching key.  
- **No Key Overwrite**: If a key is inserted twice, the second is effectively ignored by `search` because the first match is returned.  

This is a minimal hash table demonstrating how to do basic hashing in Rust with separate chaining, manual `Hashable`, dynamic resizing, and a simple test suite. It shows the standard lifecycle: **insert**, **search**, **resize**.