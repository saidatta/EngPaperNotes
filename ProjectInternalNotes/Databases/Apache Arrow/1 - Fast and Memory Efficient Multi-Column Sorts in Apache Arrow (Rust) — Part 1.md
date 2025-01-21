
aliases: [Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust, Part 1]
tags: [apache-arrow, rust, sorting, lexicographical, performance]
## Table of Contents
1. [[#Introduction|Introduction]]
2. [[#Multicolumn-/-Lexicographical-Sort-Problem|Multicolumn / Lexicographical Sort Problem]]
3. [[#Basic-Implementation|Basic Implementation]]
4. [[#Normalized-Keys-/-Byte-Array-Comparisons|Normalized Keys / Byte Array Comparisons]]
5. [[#Code-Snippets-and-Examples|Code Snippets and Examples]]
6. [[#Visualizing-Memory-and-Comparisons|Visualizing Memory and Comparisons]]
7. [[#References-and-Further-Reading|References and Further Reading]]
---
## Introduction

Sorting is one of the most fundamental operations in modern databases and analytic systems. **By some estimates, more than half of the execution time in data processing systems is spent on sorting**, making sorting a prime target for optimization.

A classic academic treatment on this topic is [Implementing Sorting in Database Systems by Goetz Graefe](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2667&rep=rep1&type=pdf). Although much of its content is still highly applicable, the complexity in modern systems has evolved, and the techniques discussed in that paper (e.g., external sorts, in-memory optimizations, distribution-based approaches) often need modern adaptation. 

**The Rust implementation of Apache Arrow** provides a columnar in-memory data format and computational kernels designed for analytics. This post focuses on the newly added **row format** in the Rust implementation of Apache Arrow (`arrow-rs`) and how we use that row format to efficiently achieve **multi-column (lexicographical) sorting**. We can achieve **over 3x performance improvement** for sorts, especially those involving multiple columns, strings, dictionary-encoded data, or a large number of columns. 

Key points from this post:
1. Motivation for multi-column sorting in analytics.
2. Basic comparator-based approach and its performance pitfalls.
3. The idea of normalizing multi-column rows into a **single contiguous byte array** (a.k.a. “normalized keys”).
4. How normalized keys allow more efficient sorting with straightforward `memcmp`-style comparisons and predictable access patterns.

---

## Multicolumn / Lexicographical Sort Problem

Many programming languages (including Rust) have optimized, **type-specific sorting** algorithms for single-column arrays. However, **analytic systems** often:
- Require **sorting over multiple columns**.
- **Cannot assume known data types at compile time** (e.g., data might arrive in many different Arrow types: `Int64`, `Float64`, `Utf8`, `Dictionary`, etc.).

A **lexicographical** (or multi-column) sort means we sort rows by the first column; if there are ties, we continue to the second column, and so on. For example, consider sorting rows by `(State, Orders)`.

### Example

| Customer | State | Orders  |
|----------|-------|---------|
| 12345    | MA    | 10.12   |
| 532432   | MA    | 8.44    |
| 12345    | CA    | 3.25    |
| 56232    | WA    | 6.00    |
| 23442    | WA    | 132.50  |
| 7844     | CA    | 9.33    |
| 852353   | MA    | 1.30    |

Sorting by `(State, Orders ASC)`, the sorted result is:

| Customer | State | Orders  |
|----------|-------|---------|
| 12345    | CA    | 3.25    |
| 7844     | CA    | 9.33    |
| 852353   | MA    | 1.30    |
| 532432   | MA    | 8.44    |
| 12345    | MA    | 10.12   |
| 56232    | WA    | 6.00    |
| 23442    | WA    | 132.50  |

Even if a query only wants the “lowest 10 orders” per state (i.e., a “TopK” scenario), the same multi-column **comparison** steps are needed. 

---
## Basic Implementation

### Conceptual Pseudocode

A straightforward way to implement multi-column sorting is:
1. Provide a function like `lexsort_to_indices` that:
   - Takes a list of columns as input.
   - Returns a **list of indices** in sorted order (instead of rearranging the columns themselves).
2. Construct a **comparator** that, given two row indices, compares the column values at those indices.

```python
# Takes a list of columns and returns the lexicographically
# sorted order as a list of indices
def lexsort_to_indices(columns):
    comparator = build_comparator(columns)

    # Construct a list of integers from 0..N
    indices = list(range(columns.num_rows()))
    # Sort it using the comparator
    indices.sort(key=comparator_function_like)
    
    return indices

def build_comparator(columns):
    def comparator(idx):
        # "Flatten" the row at idx, or build a tuple for example
        # This might be for a "key function" style in Python
        # or we can do a custom compare function.
        return tuple(column[idx] for column in columns)
    return comparator
```

> In Rust, we typically implement this by constructing a **closure** or **struct** that holds references to the Arrow arrays, plus the sorting criteria, and implements `std::cmp::Ord`. 

#### Comparator Performance Issues

1. **Conditionals on each column**: The comparator calls code that depends on the column data type at runtime. For each comparison, we might have a big `match` or branch on type variants.
2. **Dynamic dispatch**: We might be calling `compare(column, left_idx, right_idx)` that itself has dynamic dispatch or a chain of if-else statements for each Arrow type (`Int32`, `Float64`, `Utf8`, etc.). 
3. **Unpredictable memory accesses**: Columnar data is typically laid out by column. When comparing two *rows*, we jump between columns for the left index and right index in memory.

These can all reduce sorting speed significantly.

---
## Normalized Keys / Byte Array Comparisons

A known optimization is to represent each row as a **contiguous byte slice** so that a **byte-wise (`memcmp`)** comparison yields the same ordering as the multi-column, multi-type comparison. This is often called a **“normalized key”** approach.
### Why This Helps

1. **Predictable Access**: We access a contiguous slice of bytes rather than scattered locations in different columns.
2. **Hardware Optimization**: Modern architectures can compare memory blocks extremely fast (e.g., using built-ins like `memcmp`, SIMD instructions, or specialized instructions).
3. **No Dynamic Dispatch**: Once data is in a normalized key, a single pass `memcmp` can do the comparison, with no branching on column types.

In practice, we:
1. Convert each row from the set of Arrow columns into a single **row-oriented** byte buffer.
2. Keep track of fixed-length vs. variable-length fields.
3. Use that row-encoded buffer for the entire sort comparison.

The next post (Part 2) will explain how to handle:
- **Variable-length fields** such as `Utf8` strings or binary arrays.
- **Dictionary-encoded** data, which can be more efficient if we leverage the dictionary-encoded IDs directly in the normalized form.

---

## Code Snippets and Examples

Below are some **Rust**-oriented examples to illustrate how one might implement these ideas in practice with Arrow arrays. Note that this is **conceptual**; the actual Apache Arrow Rust implementation might differ in detail.

### 1. Building a Comparator with Dynamic Dispatch

```rust
use arrow::array::{
    ArrayRef, Float64Array, Int64Array, StringArray, // etc
};
use std::cmp::Ordering;

/// A small struct to hold references to the arrays and the sort order
pub struct MultiColumnComparator<'a> {
    /// Each element is a column plus sort options, e.g., ascending/descending
    columns: Vec<(&'a ArrayRef, bool)>, 
}

impl<'a> MultiColumnComparator<'a> {
    pub fn new(columns: Vec<(&'a ArrayRef, bool)>) -> Self {
        Self { columns }
    }
    
    /// Compare two row indices, returning an Ordering
    pub fn compare(&self, left_idx: usize, right_idx: usize) -> Ordering {
        for (col, ascending) in &self.columns {
            let ordering = match col.data_type() {
                arrow::datatypes::DataType::Int64 => {
                    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    let left_val = arr.value(left_idx);
                    let right_val = arr.value(right_idx);
                    left_val.cmp(&right_val)
                },
                arrow::datatypes::DataType::Float64 => {
                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    let left_val = arr.value(left_idx);
                    let right_val = arr.value(right_idx);
                    left_val.partial_cmp(&right_val).unwrap_or(Ordering::Equal)
                },
                arrow::datatypes::DataType::Utf8 => {
                    let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                    let left_val = arr.value(left_idx);
                    let right_val = arr.value(right_idx);
                    left_val.cmp(right_val)
                },
                _ => {
                    // Additional branches for more types
                    Ordering::Equal
                }
            };
            
            let ordering = if *ascending {
                ordering
            } else {
                ordering.reverse()
            };
            
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        
        Ordering::Equal
    }
}

// Usage: building the comparator and sorting indices
fn lexsort_indices(columns: Vec<ArrayRef>, ascendings: Vec<bool>) -> Vec<usize> {
    let row_count = columns[0].len();
    let pair_cols: Vec<(&ArrayRef, bool)> = columns.iter().zip(ascendings).collect();
    let comparator = MultiColumnComparator::new(pair_cols);
    
    let mut indices: Vec<usize> = (0..row_count).collect();
    indices.sort_by(|&l, &r| comparator.compare(l, r));
    indices
}
```

**Issues**:
- Multiple branches on `DataType`.
- Repeated downcasts during comparisons.
- Possibly large CPU overhead when we do many comparisons.

### 2. Normalizing Rows into a Byte Buffer

A conceptual example. We assume fixed-size types (e.g., `Int64` and `Float64`) are each 8 bytes. For `Utf8` or variable-length arrays, the logic is more complex—explained in the next post.

```rust
/// Convert a set of columns into a vector of row-encoded byte buffers.
/// This is a naive example for fixed-size columns only (no strings, no nulls).
fn to_row_format(columns: &[ArrayRef]) -> Vec<Vec<u8>> {
    let num_rows = columns[0].len();
    let row_size = columns.len() * 8; // assumption: each column is 8 bytes
    
    // Preallocate a vec of bytes for each row
    let mut rows = vec![vec![0u8; row_size]; num_rows];
    
    for (col_idx, col) in columns.iter().enumerate() {
        match col.data_type() {
            arrow::datatypes::DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                for row_idx in 0..num_rows {
                    let val = arr.value(row_idx);
                    let offset = col_idx * 8;
                    // Convert val to bytes in little-endian, for example
                    rows[row_idx][offset..offset+8].copy_from_slice(&val.to_le_bytes());
                }
            },
            arrow::datatypes::DataType::Float64 => {
                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                for row_idx in 0..num_rows {
                    let val = arr.value(row_idx);
                    let offset = col_idx * 8;
                    rows[row_idx][offset..offset+8].copy_from_slice(&val.to_bits().to_le_bytes());
                }
            },
            _ => unimplemented!("Handle other types"),
        }
    }
    
    rows
}

/// Sorting with row-format approach
fn lexsort_indices_with_rows(columns: Vec<ArrayRef>) -> Vec<usize> {
    let num_rows = columns[0].len();
    let rows = to_row_format(&columns);
    let mut indices: Vec<usize> = (0..num_rows).collect();
    
    // Compare by memcmp
    indices.sort_by(|&l, &r| {
        let row_l = &rows[l];
        let row_r = &rows[r];
        row_l.cmp(row_r) // This does lexicographical comparison of the byte slices
    });
    
    indices
}
```

In real systems, you’d likely store these row-bytes in a **single allocation** (e.g., an `Arc<MutableBuffer>` or something similar) for better cache behavior. Also, for variable-length data, we need additional tricks such as:
- Storing offset and length fields.
- Inlining some part of the string if it’s short enough (i.e., “short-string optimization”).
- Dictionary-encoded values.

---

## Visualizing Memory and Comparisons

### 1. Comparator-Based Access

Imagine each column is a separate contiguous array:

```
   row
  index      Column0 (Customer)  Column1 (State)       Column2 (Orders)
        ┌─────┐   ┌─────┐   ┌─────┐      
      0 │     │   │     │   │     │  compare(left_idx, right_idx)
       ┌├─────┤─ ─├─────┤─ ─├─────┤┐             
        │     │   │     │   │     │ ◀────────────┐
       └├─────┤─ ─├─────┤─ ─├─────┤┘             │
        │     │   │     │   │     │Comparator →  │ read states, compare, if tie → read orders
        ├─────┤   ├─────┤   ├─────┤             │
        │     │   │     │   │     │             │
        ├─────┤   ├─────┤   ├─────┤             │
        │     │   │     │   │     │             │
        └─────┘   └─────┘   └─────┘             │
                                                │
       ┌┌─────┐─ ─┌─────┐─ ─┌─────┐┐            │
        │     │   │     │   │     │ ◀───────────┘
       └├─────┤─ ─├─────┤─ ─├─────┤┘
        │     │   │     │   │     │
        ├─────┤   ├─────┤   ├─────┤
    N-1 │     │   │     │   │     │
        └─────┘   └─────┘   └─────┘
```

- **High overhead** from jumping between arrays and switching on column types.

### 2. Row-Oriented Byte Slices

When we normalize each row into a single byte slice, comparisons become more **sequential**:

```
 Row Index   Row-encoded buffer (all columns in a single slice)
     0       [  bytes for col0, col1, col2, ...           ]
     1       [  bytes for col0, col1, col2, ...           ]
     2       [  bytes for col0, col1, col2, ...           ]
     ... 
     N-1     [  bytes for col0, col1, col2, ...           ]
```

Then we can do something like:

```rust
memcmp(&rows[left_idx], &rows[right_idx]) // in a stable or partial sort
```

All columns are in **one contiguous region**. The CPU can accelerate comparisons with instructions specialized for this task.

---

## References and Further Reading

1. **Goetz Graefe**: [Implementing Sorting in Database Systems](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2667&rep=rep1&type=pdf).
2. **DuckDB Blog**: [Sorting in DuckDB](https://duckdb.org/2022/09/12/sorting.html), especially the “Binary String Comparison” section.
3. **Arrow Rust Code**: 
   - [Sorting code in arrow-rs](https://github.com/apache/arrow-rs/blob/master/arrow/src/compute/kernels/sort.rs)  
   - [Ord utilities in arrow-rs](https://github.com/apache/arrow-rs/blob/master/arrow/src/compute/kernels/ord.rs)

In the **next part** of this series, we’ll dive deeper into:
- How to handle **variable-length data** (e.g., strings) efficiently.
- How to leverage **dictionary encoding** to store and compare data more efficiently.
- Additional optimizations, such as **radix sort** for integer columns and **parallel mergesort**.

**Stay tuned!**
```

---

## Tips for Using These Notes in Obsidian

1. **Linking to Next Parts**: If you create a new note for the follow-up on variable-length and dictionary-encoded data, you can add links like `[[Next Part of the Series]]`.
2. **Visual Diagrams**: You can embed diagrams using Obsidian’s Markdown or PlantUML plugin if you want interactive visuals.
3. **Further Exploration**: Create local links to your code files or other references:  
   `[[arrow-rs-sort-implementation]]` for a deep dive into the actual Arrow Rust code.

---

**End of Note**