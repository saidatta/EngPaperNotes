aliases: [CountMinSketch, Probabilistic, Approximate Counting, Data Structures]
tags: [rust, engineering, advanced, hashing, data-structures]

**Context**: When you have a large or unbounded data stream and you want to estimate the **frequency** of items without storing a massive exact frequency map, the **Count-Min Sketch** (CMS) can help. It trades off **accuracy** for **constant** space by using multiple hash functions and arrays.
## 1. **Motivation**
- A **frequency map** for incoming items (like a `HashMap<Item, usize>`) can become huge if there is a large variety of items.
- **Count-Min Sketch**: a **probabilistic** data structure giving an **approximate** count:
  1. **Constant** memory usage determined by your chosen parameters (width, depth).
  2. Probability that the count is close to the real count.
  3. Some collisions lead to counts that **cannot** be lower than the true count, only higher.
---
## 2. **Core Idea**
### 2.1 Single Row
Imagine we only have **1** array of length `WIDTH`, and a **single** hash function:
1. For each item `item`, compute `hash(item)`, then map it to an index `idx = hash(item) % WIDTH`.
2. We increment `counts[idx]` by `1` (or by a certain amount).
**Issue**: Collisions easily over-count items. If `WIDTH` is large, we can reduce collisions but at a big memory cost.
### 2.2 Multiple Rows (Depth)
The Count-Min Sketch uses **`DEPTH`** different **hash functions**, each with its **own** array row of size `WIDTH`. 
**To increment**:
1. For each row (1..DEPTH):
   - Compute a unique hash for `item`.
   - `idx = hash(item) % WIDTH`.
   - `counts[row][idx] += 1`.
**To get the approximate count**:
1. For each row, compute the same index as above.
2. Read `counts[row][idx]`.
3. **Take the minimum** of those `DEPTH` values.
**Rationale**: 
- Because multiple different hash functions are used, if an item had collisions in row 1, it might not collide in row 2. 
- The **minimum** across rows can’t be **less** than the actual frequency, but it might be bigger if there were collisions in all rows.
---
## 3. **Struct Definition**
```rust
pub struct HashCountMinSketch<Item: Hash, const WIDTH: usize, const DEPTH: usize> {
    phantom: std::marker::PhantomData<Item>,
    counts: [[usize; WIDTH]; DEPTH],
    hashers: [RandomState; DEPTH],
}
```
1. **`WIDTH`**: length of each row.
2. **`DEPTH`**: how many rows. 
3. **`counts`** is a 2D array: `counts[row][column]` holds the partial count for that row.
4. **`hashers`** is an array of length `DEPTH` of `RandomState`s (the default build hasher in Rust). 
   - Each `RandomState` is used as a pseudo-distinct hash function.
**Derivations**:
- `PhantomData<Item>`: ensures the compiler knows `Item` is the generic type used in hashing. 
- `Default` implementation: create `DEPTH` random hash states, zero the matrix.
---
## 4. **Operations**
### 4.1 Increment
```rust
fn increment_by(&mut self, item: Item, count: usize) {
    for (row, r) in self.hashers.iter_mut().enumerate() {
        let hashed = r.hash_one(&item);
        let col = (hashed % WIDTH as u64) as usize;
        self.counts[row][col] += count;
    }
}
```
**Steps**:
- For each row, compute the item’s hash with that row's `RandomState`.
- Map to `[0..WIDTH)`.
- Add the given `count` to `counts[row][col]`.
### 4.2 Get Count
```rust
fn get_count(&self, item: Item) -> usize {
    self.hashers
        .iter()
        .map(|r| {
            let hashed = r.hash_one(&item);
            let col = (hashed % WIDTH as u64) as usize;
            self.counts[row][col]
        })
        .min()
        .unwrap()
}
```
**Steps**:
- For each row, find the index, read `counts[row][col]`.
- Return the **minimum** across all rows.
> Because collisions only *inflate* counts, the minimum is the best guess that can’t be an underestimate.
---
## 5. **Collision Example Visualization**
**Imagine** `DEPTH=7`, `WIDTH=5`, and we have 2 items `"TEST"` and `"OTHER"`:
```
Rows/Hash Functions: 7
Columns (WIDTH=5) indices: 0..4

    row0: [ .. .. .. .. .. ]
    row1: [ .. .. .. .. .. ]
    row2: [ .. .. .. .. .. ]
    row3: [ .. .. .. .. .. ]
    row4: [ .. .. .. .. .. ]
    row5: [ .. .. .. .. .. ]
    row6: [ .. .. .. .. .. ]
```
- Insert `"TEST"`, we compute 7 different hash results, each giving a column index. We increment `+1` in each row’s computed column.
- Insert `"OTHER"`, likewise.
**If** collisions happen, some cells might be incremented for more than one item, inflating that cell.  
**But** for retrieving `"TEST"`’s count, you see inflated numbers in some cells but hopefully at least one row that doesn't conflict. So the **minimum** across rows is near the true count.

---
## 6. **When to Use Count-Min Sketch**
- **Large** or **unbounded** data streams.
- You want **approximate** frequency counts but can tolerate collisions.
- Memory must remain **constant** or at least sub-linear in the number of items.
- Perfect for use-cases like:
  - Counting URL hits or packet frequencies in high-traffic system.
  - Identifying heavy hitters or top elements in streaming data.
**Trade-offs**:
- Overestimation is possible if collisions are high.
- The number of collisions is reduced by having multiple hash functions (DEPTH) and a sufficiently large WIDTH.
---
## 7. **Theoretical Bounds**
- Probability that the sketch's count for an item is above `(true_count + error)` can be controlled by setting `DEPTH` and `WIDTH` based on the desired probability and error rate. 
- Typically, `WIDTH ~ e / error` and `DEPTH ~ ln(1/prob)` are standard heuristic formulas from the [original Count-Min Sketch paper][cms-paper]. 
```plaintext
error ~ (ε) = depends on 1 / width
failure probability ~ δ = depends on depth
```
- Detailed proofs are beyond the scope here, but practically, you tune `WIDTH` and `DEPTH` to your tolerance for collisions and memory usage.
---
## 8. **Test Snippets**
### 8.1 Colliding Indices
We do a test with large `DEPTH=50` and `WIDTH=50`, insert `"something"`:
```rust
#[test]
fn hash_functions_should_hash_differently() {
    let mut sketch: HashCountMinSketch<&str, 50, 50> = Default::default();
    sketch.increment("something");
    // check positions of '1's in each row.
}
```
We want to see **not** all are in the **same** column, i.e. the row's hashed index is different. This ensures that each `RandomState` hopefully acts as a distinct hash function.
### 8.2 Checking internal counts
```rust
#[test]
fn inspect_counts() {
    let mut sketch: HashCountMinSketch<&str, 5, 7> = Default::default();
    sketch.increment("test");
    // ...
    assert_eq!(2, sketch.get_count("test"));
}
```
We see how each row has exactly one increment for "test," etc.
### 8.3 QuickCheck for Overestimation
**Goal**: The returned count never **underestimates** the real count. 
We insert random `(Item, count)` pairs, verifying `sketch.get_count(item) >= actual_count`.

---
## 9. **Overall Code**
```rust
trait CountMinSketch {
    type Item;
    fn increment(&mut self, item: Self::Item);
    fn increment_by(&mut self, item: Self::Item, count: usize);
    fn get_count(&self, item: Self::Item) -> usize;
}

pub struct HashCountMinSketch<Item: Hash, const WIDTH: usize, const DEPTH: usize> {
    phantom: std::marker::PhantomData<Item>,
    counts: [[usize; WIDTH]; DEPTH],
    hashers: [RandomState; DEPTH],
}

impl<Item: Hash, const WIDTH: usize, const DEPTH: usize> Default
    for HashCountMinSketch<Item, WIDTH, DEPTH>
{
    fn default() -> Self {
        let hashers = std::array::from_fn(|_| RandomState::new());
        Self {
            phantom: Default::default(),
            counts: [[0; WIDTH]; DEPTH],
            hashers,
        }
    }
}

impl<Item: Hash, const WIDTH: usize, const DEPTH: usize> CountMinSketch
    for HashCountMinSketch<Item, WIDTH, DEPTH>
{
    type Item = Item;

    fn increment(&mut self, item: Item) {
        self.increment_by(item, 1);
    }

    fn increment_by(&mut self, item: Item, count: usize) {
        for (row, rng) in self.hashers.iter().enumerate() {
            let hashed = rng.hash_one(&item);
            let col = (hashed as usize) % WIDTH;
            self.counts[row][col] += count;
        }
    }

    fn get_count(&self, item: Item) -> usize {
        self.hashers
            .iter()
            .enumerate()
            .map(|(row, rng)| {
                let hashed = rng.hash_one(&item);
                let col = (hashed as usize) % WIDTH;
                self.counts[row][col]
            })
            .min()
            .unwrap()
    }
}
```
---
## 10. **Conclusion & Further Reading**
**Count-Min Sketch** is a handy **approximate counting** data structure for streaming/online scenarios. It uses multiple hash rows to reduce collisions, ensuring:
- **Constant** space complexity: `O(DEPTH * WIDTH)`.
- **Update** time `O(DEPTH)`.
- **Query** time `O(DEPTH)` (just a few additions or lookups).
- **Never** **underestimates** but can **overestimate** due to collisions.
**Further References**:
1. [Cormode & Muthukrishnan's Paper (2005) - "An Improved Data Stream Summary: The Count-Min Sketch and its Applications"][cms-paper].
2. [Rust Book on generics & default type parameters](https://doc.rust-lang.org/book/ch10-00-generics.html).
3. [Rust docs on hashing & `RandomState`](https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html).
[cms-paper]: https://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf