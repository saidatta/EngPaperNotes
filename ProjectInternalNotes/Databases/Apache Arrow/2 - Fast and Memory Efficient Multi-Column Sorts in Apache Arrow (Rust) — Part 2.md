aliases: [Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust, Part 2]
tags: [apache-arrow, rust, sorting, row-format, dictionary-encoding, performance]

## Table of Contents
1. [[#Introduction|Introduction]]
2. [[#Row-Format|Row Format]]
   1. [[#Unsigned-Integers|Unsigned Integers]]
   2. [[#Signed-Integers|Signed Integers]]
   3. [[#Floating-Point|Floating Point]]
   4. [[#Byte-Arrays-Including-Strings|Byte Arrays (Including Strings)]]
   5. [[#Dictionary-Arrays|Dictionary Arrays]]
3. [[#Sort-Options|Sort Options]]
4. [[#Conclusion|Conclusion]]
5. [[#Code-Snippets-and-Examples|Code Snippets and Examples]]
6. [[#References-and-Further-Reading|References and Further Reading]]

---
## Introduction

In [[Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust, Part 1|Part 1]], we introduced the problem of **multi-column sorting** (a.k.a. lexicographical sorting) and explored the challenges of implementing it efficiently. We noted that moving away from **comparator-based** approaches to a **row-oriented** (serialized) format can significantly speed up sorts—often **3x faster** or more, particularly when dealing with strings, dictionary-encoded data, or large numbers of columns.

This post (Part 2) dives deeper into:
- How exactly the **new row format** in the Rust implementation of Apache Arrow (`arrow-rs`) is structured.
- How different data types (integers, floats, strings, dictionary-encoded arrays) are encoded within that row format.
- How the format handles **null values**, **ASC/DESC** ordering, and **NULLs first/last** scenarios.

---

## Row Format

### Basic Idea
We represent each row as a **variable-length byte sequence** formed by concatenating the **encoded** form of each column. The column-specific encoding ensures that lexicographical (byte-by-byte) comparisons on these byte sequences match the intended column-level comparisons (including ASC vs. DESC ordering, null ordering, etc.).

```text
          Input Arrays                         Row Format
           (Columns)
   ┌─────┐   ┌─────┐   ┌─────┐               ┏━━━━━━━━━━━━━┓
   │     │   │     │   │     │               ┃             ┃
   ├─────┤ ┌ ┼─────┼ ─ ┼─────┼ ┐  ──────────▶┗━━━━━━━━━━━━━┛
   │     │   │     │   │     │
   ├─────┤ └ ┼─────┼ ─ ┼─────┼ ┘
   │     │   │     │   │     │
   └─────┘   └─────┘   └─────┘
               ...
   ┌─────┐ ┌ ┬─────┬ ─ ┬─────┬ ┐              ┏━━━━━━━━┓
   │     │   │     │   │     │  ─────────────▶┃        ┃
   └─────┘ └ ┴─────┴ ─ ┴─────┴ ┘              ┗━━━━━━━━┛

   Customer    State    Orders
    UInt64      Utf8     F64
```

#### Key Principles

1. **No Ambiguity**  
   The format is designed so that each column’s boundary can be detected without additional “escaping” or scanning ambiguities.  
2. **Sorting as Byte-Comparison**  
   Once encoded, we can compare entire columns (including complex types like strings) by **memcmp**-style comparisons on the row slice.  

Below, we describe how each Arrow datatype is encoded in the row format.

---

## Unsigned Integers

For an **unsigned integer** column (e.g., `u32`, `u64` in Rust, or `UInt32`, `UInt64` in Arrow), each non-null value is encoded as:

1. A **0x01** byte to mark a **non-null**.
2. The integer’s bytes in **big-endian** order (most significant byte first).

A **null** is encoded as **0x00**, followed by an all-zero representation of the integer’s size.  
### Example: 32-bit Unsigned Integers

Let’s consider a `UInt32` array in Rust/Arrow. Internally, Arrow typically stores them in **little-endian** format. We need to invert that to big-endian to preserve correct ordering.

#### Encoding Steps
1. **Check for null**. If null ⇒ `0x00` + 4 zero bytes.
2. If non-null ⇒ `0x01` + big-endian bytes of the integer.

#### Example Values

| Value   | 32-bit Little Endian | Row Format (with big-endian + 0x01) |
|---------|-----------------------|--------------------------------------|
| 3       | `03 00 00 00`        | `01 00 00 00 03`                     |
| 258     | `02 01 00 00`        | `01 00 00 01 02`                     |
| 23423   | `7F 5B 00 00`        | `01 00 00 5B 7F`                     |
| `NULL`  | —                    | `00 00 00 00 00`                     |

In the **row format** column segment:
- A leading `0x01` means “valid integer follows” in big-endian.
- A leading `0x00` means “null.”
---

## Signed Integers

Signed integers (e.g., `Int32`, `Int64`) in Rust and many architectures use **two’s complement**. Their “order” in sorted sequences can be preserved if we do the following:

1. **Flip the sign bit** (the highest-order bit) to effectively convert the signed range into an unsigned range.  
   - Positive values have a 0 sign bit; negative values have a 1 sign bit in two’s complement.
   - Flipping that bit (often referred to as `x ^ 0x80...`) re-centers negative values in the domain so that negative numbers come before positive numbers in simple unsigned comparison.
2. Encode the resulting bytes as if it’s an unsigned integer, as in the previous section.

In short, the steps are:

1. If null ⇒ `0x00` + all-zero bytes for the integer size.
2. If non-null ⇒
   - Flip the **most significant bit** (the top bit) of the integer’s **native-endian** representation to handle signed ordering.
   - Reorder to **big-endian**.
   - Prepend `0x01`.

### Example: 32-bit Signed Integers

| Value | 32-bit Little Endian | Flip High Bit of the **4th byte** | Row Format (after big-endian + `0x01`) |
|-------|-----------------------|-----------------------------------|----------------------------------------|
| 5     | `05 00 00 00`        | `05 00 00 80`                     | `01 80 00 00 05`                       |
| -5    | `FB FF FF FF`        | `FB FF FF 7F`                     | `01 7F FF FF FB`                       |

Thus, `5` becomes `0x01 80 00 00 05`, while `-5` becomes `0x01 7F FF FF FB`.  

---

## Floating Point

Floating point values (`Float32` or `Float64`) must be compared using the **IEEE 754 total order**. Rust provides [`f32::total_cmp`](https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp) and [`f64::total_cmp`](https://doc.rust-lang.org/std/primitive.f64.html#method.total_cmp), or you can replicate that logic manually.

**Steps**:
1. Interpret the float bits in **native-endian** (usually little-endian on most platforms).
2. Convert them into a **signed integer** of the same bit width (e.g., 32 bits for `f32`, 64 bits for `f64`).
3. Flip the sign bit (similar to the signed integer approach).
4. Encode as a “signed integer” in big-endian, prepending `0x01` if non-null.

This ensures that **NaN**s, infinities, negative zeros, etc. are consistently ordered per IEEE 754 `totalOrder`.

---

## Byte Arrays (Including Strings)

Byte arrays, including `Utf8` columns in Arrow, can vary significantly in length. We need to:
1. Encode **null** values distinctly (leading `0x00`).
2. Encode **empty** strings distinctly (leading `0x01`).
3. Allow for arbitrary-length data without confusion about where the byte array ends.
### Row Format Encoding
1. **Null** ⇒ `0x00`.
2. **Empty** ⇒ `0x01`.
3. **Non-Empty** ⇒
   1. Write `0x02`.
   2. Write the data in **32-byte** blocks (for actual implementation). For simplicity of illustration, examples use **4-byte** blocks.
   3. After each full block, write `0xFF` as a **continuation token** if more blocks remain.
   4. For the **final block**, pad it with `0x00` up to the block boundary. Instead of `0xFF`, write **the unpadded length** in a single byte.

#### Example with 4-Byte Blocks

| String            | Encoding                                                 |
|-------------------|----------------------------------------------------------|
| `"MEEP"`          | `0x02 'M' 'E' 'E' 'P' 0x04`                              |
| `""` (empty)      | `0x01`                                                  |
| `NULL`            | `0x00`                                                  |
| `"Defenestration"`| `0x02 'D' 'e' 'f' 'e' 0xFF  'n' 'e' 's' 't' 0xFF  'r' 'a' 't' 'i' 0xFF 'o' 'n' 0x00 0x00 0x02` |

Let’s break down `"Defenestration"` with a 4-byte block size:

1. `0x02` => start of non-empty byte array.
2. Block1: `['D','e','f','e']`, then `0xFF` => continuation.
3. Block2: `['n','e','s','t']`, then `0xFF` => continuation.
4. Block3: `['r','a','t','i']`, then `0xFF` => continuation.
5. Final Partial Block: `['o','n']`, plus 2 bytes of padding (`0x00 0x00`), then `0x02` to indicate final block length is 2.

**Why?**  
- This style is **loosely inspired by COBS** (Consistent Overhead Byte Stuffing).
- We use `0xFF` as a **block continuation marker**, and a small integer at the end to say “the last block has size = X”.
- It's **amenable to vectorization** (you can copy 32 bytes in a single SIMD instruction on modern CPUs).

---
## Dictionary Arrays

**Dictionary-encoded columns** (similar to “categorical” in pandas) are common for storing **low-cardinality** data efficiently. In Arrow:
- Each column chunk (`ArrayData`) has a **dictionary** and **keys** (small integer indexes into the dictionary).
- Different chunk batches might have **different** dictionaries, and we can’t assume the dictionary is sorted or consistent across chunks.

### Naive Approach

A naive approach might be:
1. **Replace dictionary keys** with their “expanded” logical values.
2. Encode those values via the relevant encoding rules (strings, integers, etc.).

But that defeats dictionary compression benefits—especially for large repeated strings, we lose the advantage of storing them once in the dictionary.

### Order-Preserving Mapping

The approach in `arrow-rs` is to build an **order-preserving mapping** from each dictionary value to a **unique variable-length byte prefix**. Whenever we encounter a new dictionary value, we **assign** it a new prefix that compares higher (or lower) than any previously assigned prefix. Because you can always create a new prefix by adding a single byte (e.g., `0x01`, or `0x02` after an existing prefix, etc.), you don’t need to rebuild the entire encoding.

1. If the dictionary value is `NULL` ⇒ single `0x00`.
2. If non-null:
   - Write a `0x01`.
   - Write the **encoded prefix** (the “mapping”) in a **0x00-terminated** form.

Here’s a conceptual example of the mapping:

```
"Bar"        --> 01
"Fabulous"   --> 01 02
"Soup"       --> 05
"ZZ"         --> 07
```

When encoding the dictionary key “Fabulous,” we produce something like:

```
0x01 (non-null) + [ 03 05 ... ] + 0x00 (terminator)
```

*(In the blog post, they show a direct mapping from strings to short sequences, but the actual structure can vary as long as order is preserved.)*

**Crucially**, we don’t need to see all possible dictionary values at once. As we process new arrays (batches), we assign new entries in the mapping. Because each assigned prefix is guaranteed to be in the correct sorted position, we maintain a global order for dictionary values.

---

## Sort Options

### ASC / DESC

To handle **ascending** or **descending** sorting at the column level, we can **invert** the encoded bytes for descending columns:
- For ascending: use the normal encoding described above.
- For descending: after forming the bytes, **flip each byte** except the initial `0x00` or `0x01` (which indicates null vs. valid).  
  This inverts the ordering for that column while keeping null detection intact.

### Null Ordering

SQL standard allows `NULLS FIRST` or `NULLS LAST`. The row format can:
- Encode null with `0x00` for `NULLS FIRST`, or
- Encode null with `0xFF` for `NULLS LAST`.
  
Because `0xFF` is higher than any `0x01` or normal data bytes, those nulls become “largest” in the ordering.

---

## Conclusion

Across these **two articles**, we’ve seen:

1. **Why** multi-column sorting is challenging and how comparing row-oriented byte slices outperforms repeated dynamic dispatch.
2. **How** the Rust Arrow row format is constructed, covering integers, floats, bytes, and dictionary arrays.
3. **Null** handling and **ASC/DESC** toggling in a straightforward manner.

**Practical Impact**:
- This format is used in the Rust Arrow crate to provide a lexicographical sort that runs **3x faster** than the comparator-based approach, especially for large or string-heavy data. 
- It also improves related operations like **sort-merge join**, **grouping**, **window functions**, etc.

We encourage you to explore the code, try it out, and report any issues or questions on the project’s bug tracker.

---

## Code Snippets and Examples

### 1. High-Level Encoder Trait (Conceptual)

```rust
use arrow::array::{Array, ArrayRef, Float64Array, UInt64Array, StringArray};
use arrow::datatypes::DataType;

trait RowEncoder {
    /// Encode a single element at `row_idx` in `array` into `buffer`.
    /// The `sort_option` might specify ascending/descending, null ordering, etc.
    fn encode_into(
        &self,
        array: &ArrayRef,
        row_idx: usize,
        buffer: &mut Vec<u8>,
        sort_option: SortOption,
    );
}

struct UInt64Encoder;

impl RowEncoder for UInt64Encoder {
    fn encode_into(
        &self,
        array: &ArrayRef,
        row_idx: usize,
        buffer: &mut Vec<u8>,
        sort_option: SortOption,
    ) {
        let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
        if arr.is_null(row_idx) {
            // Could be 0x00 for NULLS FIRST or 0xFF for NULLS LAST
            buffer.push(sort_option.null_byte);
            // Then push 8 zero bytes
            buffer.extend_from_slice(&[0; 8]);
        } else {
            // 0x01 for non-null
            buffer.push(0x01);
            let val = arr.value(row_idx);
            let encoded = val.to_be_bytes(); // big-endian
            // If descending, invert the bytes (except for the initial 0x01)
            let final_bytes = if sort_option.descending {
                encoded.map(|b| !b)
            } else {
                encoded
            };
            buffer.extend_from_slice(&final_bytes);
        }
    }
}

// Example for Float64 (simplified, ignoring totalOrder details)
struct Float64Encoder;

impl RowEncoder for Float64Encoder {
    fn encode_into(
        &self,
        array: &ArrayRef,
        row_idx: usize,
        buffer: &mut Vec<u8>,
        sort_option: SortOption,
    ) {
        let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
        if arr.is_null(row_idx) {
            buffer.push(sort_option.null_byte);
            buffer.extend_from_slice(&[0; 8]);
        } else {
            buffer.push(0x01);
            let val = arr.value(row_idx);
            // Convert float -> sign-flipped integer -> big-endian
            let bits = val.to_bits();
            // Proper total_cmp flipping might be needed here
            let be = bits.to_be_bytes();
            let final_bytes = if sort_option.descending {
                be.map(|b| !b)
            } else {
                be
            };
            buffer.extend_from_slice(&final_bytes);
        }
    }
}

#[derive(Clone, Copy)]
struct SortOption {
    descending: bool,
    null_byte: u8, // could be 0x00 or 0xFF
}
```

> In reality, the Arrow row format code is more elaborate, especially for dictionary arrays and variable-length arrays.

### 2. Variable-Length Encodings Example

A pseudo-code snippet to encode short `StringArray` values:

```rust
struct ByteArrayEncoder;

impl RowEncoder for ByteArrayEncoder {
    fn encode_into(
        &self,
        array: &ArrayRef,
        row_idx: usize,
        buffer: &mut Vec<u8>,
        sort_option: SortOption,
    ) {
        let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
        if arr.is_null(row_idx) {
            buffer.push(sort_option.null_byte);
            return;
        }
        let s = arr.value(row_idx);
        if s.is_empty() {
            // 0x01 for empty
            buffer.push(0x01);
            return;
        }
        // 0x02 for non-empty
        buffer.push(0x02);

        // We chunk `s` into 32-byte blocks (example: 4 for brevity).
        let chunk_size = 32;
        let mut i = 0;
        while i < s.len() {
            let block_end = (i + chunk_size).min(s.len());
            let block = &s.as_bytes()[i..block_end];
            // copy block
            buffer.extend_from_slice(block);
            if block_end < s.len() {
                // if more bytes remain, write 0xFF as continuation
                buffer.push(0xFF);
            }
            i = block_end;
        }

        // For the final block, we need to pad and write the length in 1 byte
        let remainder = s.len() % chunk_size;
        if remainder == 0 {
            // If it fits exactly, we typically still write 0x00 as length?
            buffer.extend_from_slice(&[0x00]);
        } else {
            // pad
            let pad = chunk_size - remainder;
            buffer.extend_from_slice(&vec![0x00; pad]);
            // write remainder as length
            buffer.push(remainder as u8);
        }
        
        // If descending, we might invert these bytes except the initial indicator
        if sort_option.descending {
            // invert from the second byte onward
            let start_idx = buffer.len() - (s.len() + ???); 
            // actual logic depends on how you handle the final block
            // ...
        }
    }
}
```

---

## References and Further Reading

1. **Rust Arrow Code**  
   - [arrow-rs GitHub repository](https://github.com/apache/arrow-rs)  
   - Look for row-oriented encoding and the corresponding PRs/issues for details.
2. **COBS Encoding**  
   - [Consistent Overhead Byte Stuffing (COBS)](https://en.wikipedia.org/wiki/Consistent_Overhead_Byte_Stuffing)
3. **DuckDB Blog** on sorting and short fixed strings  
   - [DuckDB: Sorting blog post](https://duckdb.org/2022/09/12/sorting.html)
4. **IEEE 754 totalOrder**  
   - [Rust’s `f64::total_cmp`](https://doc.rust-lang.org/std/primitive.f64.html#method.total_cmp)

---

## Final Thoughts

- The **row format** is a powerful approach to bridging the gap between columnar storage (great for analytics) and **row-based** comparisons needed for sorting, merging, grouping, etc.
- By ensuring a clear, **unambiguous** byte layout, we gain **fast memcmp**-style comparisons and can handle variable-length data, nulls, dictionary encoding, and sort directions.
- The performance benefits are substantial—**3x+ speedups** for sorting, plus improved merges, joins, and window operations in **DataFusion** and other Arrow-based systems.
- Further optimizations (SIMD-based copying, parallel mergesort, etc.) become more natural once data is in a unified byte-comparison-friendly representation.