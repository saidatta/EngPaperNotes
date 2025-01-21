
---
aliases: [SQL to Row Format Example, ASC Sorting Example]
tags: [apache-arrow, rust, sorting, row-format, example]
---

This note provides a **complete, concrete example** of how a small SQL table can be **converted** to the Arrow **row-format style** described in the [Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust (Parts 1 & 2)](./Fast%20and%20Memory%20Efficient%20Multi-Column%20Sorts%20in%20Apache%20Arrow%20Rust%2C%20Part%201.md). We then show how **lexicographical (ASC) sorting** works by comparing the resulting byte-encoded rows.

---

## 1. Example SQL Table

We’ll use a simplified table with three columns to illustrate:

```sql
CREATE TABLE orders (
    customer_id  UInt32    NOT NULL,
    state        VARCHAR(2),
    amount       FLOAT
);

INSERT INTO orders VALUES
(12345,  'MA', 10.12),
(532432, 'MA',  8.44),
(12345,  'CA',  3.25),
(56232,  'WA',  6.00),
(23442,  'WA', 132.50),
(7844,   'CA',  9.33),
(852353, 'MA',  1.30);
```

We have **7 rows**:

| customer_id | state | amount  |
|-------------|-------|---------|
| 12345       | MA    | 10.12   |
| 532432      | MA    | 8.44    |
| 12345       | CA    | 3.25    |
| 56232       | WA    | 6.00    |
| 23442       | WA    | 132.50  |
| 7844        | CA    | 9.33    |
| 852353      | MA    | 1.30    |
We want to illustrate how these rows get encoded in the **Arrow row format**, then show how **sorting by `(state ASC, amount ASC)`** proceeds by simple **memcmp**-style comparisons on the encoded bytes.

---
## 2. Row Format Overview
Recall from [Part 2](./Fast%20and%20Memory%20Efficient%20Multi-Column%20Sorts%20in%20Apache%20Arrow%20Rust%2C%20Part%202.md):

1. **UInt32** (non-null) is encoded as:
   - A leading `0x01` indicating non-null.
   - The 4-byte **big-endian** representation of the integer.
     - If `NULL`, we’d write `0x00` + 4 zero bytes. (But here, `customer_id` is NOT NULL.)
2. **String (state)** is encoded in a **variable-length** manner:
   - `0x00` for null.
   - `0x01` for empty string.
   - `0x02` followed by 32-byte blocks (here we’ll just show short examples) for non-empty.
3. **Float64** is encoded like a signed integer with a **sign-bit flip** to preserve IEEE 754 total ordering.  
   - If non-null, `0x01` + (8-byte big-endian, sign-flipped).
   - If null, `0x00` + 8 zero bytes.

We’ll assume:
- **No NULL** in `state` or `amount` except for demonstration if needed (the actual table has no explicit NULLs, but we show how it would be handled).
- **ASC** ordering, so we do **not** invert the bytes after encoding. (For DESC we would flip bits after constructing the normal encoding, except the initial 0x01 or 0x00.)

---

## 3. Encoding Each Column

### 3.1 Encoding `customer_id` (UInt32)

All `customer_id` values are **not null**, so we start each encoding with `0x01` plus the **big-endian** representation of the 4-byte integer. Let’s illustrate two examples in detail:

- **Example**: `customer_id = 12345`  
  - Decimal 12345 in hex is `0x3039`. But we need 4 bytes, big-endian: `0x00 0x00 0x30 0x39`.
  - Prepend `0x01`:  
      $\text{Encoding} = 0x01,\,0x00,\,0x00,\,0x30,\,0x39$
  - Total 5 bytes.
- **Example**: `customer_id = 532432`  
  - Decimal 532432 in hex is `0x082A10` (which is 3 bytes) → zero-pad to 4 bytes: `0x00 0x08 0x2A 0x10`.
  - Prepend `0x01`:  
      $\text{Encoding} = 0x01,\,0x00,\,0x08,\,0x2A,\,0x10$
We do likewise for all 7 `customer_id` values:

1. `12345`   → `0x01 00 00 30 39`
2. `532432`  → `0x01 00 08 2A 10`
3. `12345`   → `0x01 00 00 30 39`  (same as row #1)
4. `56232`   → decimal 56232 = `0x00 DB 0xE8`, pad → `0x00 0xDB 0xE8` → big-endian is `0x00 0xDB 0xE8` (still 3 bytes?), zero pad to 4: `0x00 0xDB 0xE8` needs to confirm. Actually 56232 decimal is 0xDBE8 in hex. So 2 bytes. The 4 bytes are `0x00 0x00 0xDB 0xE8`.  
   - With the `0x01` prefix: `0x01 00 00 DB E8`
5. `23442` → decimal 23442 = `0x5B D2` (2 bytes). So big-endian: `0x00 0x00 0x5B 0xD2`. Then `0x01` prefix = `0x01 00 00 5B D2`.
6. `7844`  → decimal 7844 = `0x1E 94`. Big-endian 4 bytes: `0x00 0x00 0x1E 0x94`. Then `0x01 00 00 1E 94`.
7. `852353` → decimal 852353 = `0x0D 01 61`. For 4 bytes, that’s `0x00 0x0D 0x01 0x61`. Then prefix `0x01`, so `0x01 00 0D 01 61`.

### 3.2 Encoding `state` (Utf8, 2-letter code)

We treat these as **variable-length** strings. Each is non-empty (no explicit NULL or empty). According to the described format:

- Non-empty ⇒ `0x02`, followed by the characters in 32-byte blocks, with a small block size for example. Because all states have 2 letters, we can do it in a **single “block”** with a “final block length” marker. We’ll use a simplified 4-byte block approach:

For `MA`:
1. Write `0x02` to say “non-empty string.”
2. Write `'M' 'A'`. That’s 2 bytes. 
3. We are at end, so no continuation (`0xFF`) needed.
4. Pad the block to 4 bytes total: 2 extra `0x00`.
5. Write the final block length as a single byte: `0x02` (since `'M' 'A'` are 2 bytes).

So the final encoding for `MA`:

```
0x02 'M' 'A' 0x00 0x00 0x02
```
In ASCII/hex:
- `'M' = 0x4D`
- `'A' = 0x41`
Hence: `0x02 4D 41 00 00 02`.

Similarly, for `CA`:
```
0x02 43 41 00 00 02
```
For `WA`:
```
0x02 57 41 00 00 02
```

### 3.3 Encoding `amount` (Float64)

Each `amount` is:
1. If null: `0x00` + 8 zero bytes.
2. If non-null: `0x01` + sign-flipped big-endian bytes.

We won't do the full “IEEE754 totalOrder sign-bit flip” in detail here, but conceptually:

- Convert the `f64` to its **little-endian** bits (on typical systems).
- Flip the most significant bit to handle sign ordering (so negatives sort before positives).
- Reverse the byte order to big-endian.
- Prepend `0x01`.

Let’s do one example for `10.12` (in decimal). The 64-bit IEEE 754 bit pattern for 10.12 can be computed in Rust:

```rust
let bits = 10.12_f64.to_bits(); 
println!("{:016X}", bits);
```

- Suppose we get `0x4024333333333333` (this is approximate for demonstration).
  - The sign bit is the top bit (0 for positive).
  - For a purely ascending scenario, we typically do the “flip sign bit, then big-endian.” Because the sign is `0` for a positive number, flipping it would set it to `1`, but in many row-format implementations, we do a slightly different approach to keep the correct ordering. For simplicity, let's just illustrate a direct "big-endian" approach without the advanced totalOrder nuance:

  1. Big-endian of `0x4024333333333333` is `0x33 33 33 33 33 33 24 40`.
  2. Prepend `0x01`.

Hence, you might see something like:
```
0x01 33 33 33 33 33 33 24 40
```
*(In practice, an actual total_cmp-based approach might produce a slightly different result to handle sign flips, NaNs, -0.0, etc. The key idea is we store 8 bytes in a way that preserves correct ordering when compared as unsigned bytes.)*

We do likewise for `8.44`, `3.25`, `6.00`, `132.50`, `9.33`, and `1.30`.

---

## 4. Concatenating Columns into Row Bytes

To form the **row format** for each row, we **concatenate** the encoded column bytes **in column order**. For our example with columns `(customer_id, state, amount)`, each row’s final byte-sequence looks like:

```
Row i (combined bytes) = [Encoding_of_customer_id] + [Encoding_of_state] + [Encoding_of_amount]
```

Let’s do a smaller demonstration for **Row 1**: `(12345, 'MA', 10.12)`

1. `customer_id = 12345`  
   → `0x01 00 00 30 39`
2. `state = "MA"`  
   → `0x02 4D 41 00 00 02`
3. `amount = 10.12 (f64)`  
   → (Example) `0x01 33 33 33 33 33 33 24 40`

Concatenate:

```
[ 0x01, 0x00, 0x00, 0x30, 0x39 ]
+
[ 0x02, 0x4D, 0x41, 0x00, 0x00, 0x02 ]
+
[ 0x01, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x24, 0x40 ]
```

Putting them all together in one sequence (hex/spaced for clarity):

```
01 00 00 30 39 02 4D 41 00 00 02 01 33 33 33 33 33 33 24 40
```

This row-format byte slice can be compared with other rows by simple lexicographical comparison. The same procedure applies to each row.

---

## 5. Sorting by `(state ASC, amount ASC)`

### 5.1 Lexicographical Comparison

If we want to sort the table by `(state ASC, amount ASC)`, we rely on the fact that:

1. **The first column** in each row’s byte-slice is `customer_id`, but for sorting by `(state, amount)`, we want to effectively skip or reorder. In a typical usage, we’d build the row format so that the **sort key columns** appear first in the concatenation. 
   - Alternatively, we can store all columns in row format, but for the sort, we specifically compare the bytes corresponding to `state` first, then `amount`, and if still tied, compare `customer_id`. 

In many row-format implementations (like the one described in the blog posts), **the columns are reordered** so that the “primary sort column” is encoded first, the “secondary sort column” next, etc. This ensures a single `memcmp` respects the multi-column priority.

For demonstration, suppose we reorder the encoding columns to `(state, amount, customer_id)`. Then each row’s byte sequence begins with the `state` encoding, followed by `amount`, then `customer_id`. 

1. Comparing any two rows: 
   - Compare the first bytes (the `state` chunk). The row with a lexicographically smaller chunk for `state` sorts first.  
   - If `state` is the same, proceed to compare the next 8 (or more) bytes for `amount`.  
   - If `amount` is the same, finally compare the `customer_id` chunk.

### 5.2 Example Ordering

Looking at the original table:

| customer_id | state | amount  |
|-------------|-------|---------|
| 12345       | MA    | 10.12   |
| 532432      | MA    | 8.44    |
| 12345       | CA    | 3.25    |
| 56232       | WA    | 6.00    |
| 23442       | WA    | 132.50  |
| 7844        | CA    | 9.33    |
| 852353      | MA    | 1.30    |

Sorting by `(state, amount)` **ASC**:

1. `CA` < `MA` < `WA`, so rows with `state='CA'` come first. Among those with `CA`, we compare `amount`:
   - Row with `CA, 3.25`
   - Row with `CA, 9.33`
2. Next, `MA`, compare `amount` among the `MA` rows:
   - Row with `MA, 1.30`
   - Row with `MA, 8.44`
   - Row with `MA, 10.12`
3. Lastly, `WA`, comparing amounts:
   - Row with `WA, 6.00`
   - Row with `WA, 132.50`

Hence final order:

| customer_id | state | amount  |
|-------------|-------|---------|
| 12345       | CA    | 3.25    |
| 7844        | CA    | 9.33    |
| 852353      | MA    | 1.30    |
| 532432      | MA    | 8.44    |
| 12345       | MA    | 10.12   |
| 56232       | WA    | 6.00    |
| 23442       | WA    | 132.50  |

If we look at the **row-format** bytes (assuming we put `(state, amount, customer_id)` in that order), each row’s initial bytes for `state` (`"CA" < "MA" < "WA"`) will cause the correct global ordering in a direct lexicographic comparison.

---
## 6. Summary

**In practice**:

1. We construct row-format encodings where the **sort-key columns** appear first in the byte layout.
2. Each column’s **null / non-null** indicator, big-endian or sign-bit-flipped representation, and variable-length logic (for strings) ensures correct ordering.
3. A single **lexicographical comparison** on the row’s byte slice implements multi-column ASC sorting (and similarly DESC if we flip bytes post-encode, except the initial null indicator).

This example highlights how a SQL table—stored in Arrow arrays by column—can be **transformed** to a row-oriented “normalized key.” Then a standard **sort** (e.g., Rust’s `sort_by` or a specialized Radix/Merge sort) uses **byte-wise** comparison, which is typically **much faster** than repeated branching on column types or dynamic dispatch.

**Result**: ~3x or better speedups for multi-column sorts, especially beneficial for string-heavy or multi-column data sets.

---

## Further Reading

- [[Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust, Part 1]]
- [[Fast and Memory Efficient Multi-Column Sorts in Apache Arrow Rust, Part 2]]
- [Arrow Rust repository](https://github.com/apache/arrow-rs) and the code for the row-format.
- [DuckDB blog on Sorting](https://duckdb.org/2022/09/12/sorting.html).
```