https://www.youtube.com/watch?v=WSdg_Km4rIg
---
### **Introduction to MVCC**
MVCC (Multiversion Concurrency Control) is a concurrency control method used in databases to handle simultaneous transactions without causing conflicts. It allows multiple versions of data to exist concurrently, providing each transaction with a consistent snapshot of the database, effectively resolving common concurrency issues like **dirty reads** and **lost updates**.

MVCC is widely used in databases such as **PostgreSQL**, **MySQL (InnoDB)**, and **Oracle**, providing **snapshot isolation** as well as increased performance in read-heavy environments.

#### **Concurrency Problems MVCC Solves**
1. **Dirty Reads**: Reading uncommitted data that might be rolled back.
2. **Lost Updates**: Concurrent updates to the same data might result in one update overwriting the other.
3. **Non-repeatable Reads**: A transaction reads the same row multiple times and gets different values each time.
4. **Phantom Reads**: A transaction reads a set of rows that satisfies a condition, but another transaction inserts or deletes rows that would change the result.

---

### **MVCC Core Concept**
MVCC operates by maintaining **multiple versions** of data instead of locking rows. Each transaction works with its own snapshot of the database, and updates result in **new versions** of the data rather than modifying the existing version. This approach avoids direct conflicts by allowing read and write operations to proceed concurrently on different data versions.

#### **Example:**
Consider an attribute `A` in a database row with an initial value of 34.

1. **Transaction 1 (T1)** updates `A` from 34 to 42.
2. **Transaction 2 (T2)** starts reading `A` while T1 is still in progress.
   - In MVCC, T2 reads the original value 34 (before T1 is committed).
   - If T1 fails or rolls back, the new version (42) is discarded, and T2 never sees the uncommitted change.
3. After T1 commits, new transactions will see the updated value 42.

This resolves problems like dirty reads since no transaction ever reads uncommitted changes.

---

### **MVCC Storage Strategies**

#### **1. Append-Only Storage Model**
In the **append-only** model, each new version of data is appended, forming a **version chain**.

- **Structure**: Each row contains a pointer to its previous version, like a **linked list**.
- **Example**:
  - `Version 0`: A = 34 (initial)
  - `Version 1`: A = 42 (created by T1)
  
  ```sql
  | A  | Version | Previous Version |
  |----|---------|------------------|
  | 34 |    0    |       NULL       |
  | 42 |    1    |        0         |
  ```

- **Benefit**: Easy to retrieve the latest version and traverse older versions.

#### **2. Time Travel Storage Model**
This model separates **current data** and **historical data**. The latest data is stored in the primary table, while older versions are stored in a **time travel table**.

- **Structure**:
  - **Main Table**: Stores only the latest version.
  - **Time Travel Table**: Stores historical versions, with pointers to previous data.

- **Use Case**:
  - Keep frequently accessed data (latest version) on **faster storage** (e.g., SSD).
  - Store historical data on **slower, cheaper storage** (e.g., HDD).

- **Diagram**:
  ```
  Main Table:
  | A  | Version | Pointer to Older Version |
  |----|---------|--------------------------|
  | 42 |    1    |         Time Table        |
  
  Time Travel Table:
  | A  | Version | Previous Version |
  |----|---------|------------------|
  | 34 |    0    |       NULL       |
  ```

#### **3. Delta Storage Model**
Instead of storing full data in every version, the **delta storage model** stores **deltas** (differences) between consecutive versions.

- **Benefit**: Saves space by only recording what has changed between versions.
- **Trade-off**: Slower reads for older versions since deltas must be applied recursively to reconstruct the data.

- **Example**:
  - Original value: A = 34
  - T1 updates A to 42.
  
  ```sql
  Version 1: Delta = +8 (42 - 34)
  ```

- **Formula**:
  To get the value of A at Version 0:
  \[
  A_{v0} = A_{v1} - \Delta_{v1} = 42 - 8 = 34
  \]

---

### **Understanding MVCC in Databases**

#### **PostgreSQL MVCC**
PostgreSQL implements MVCC by storing **transaction IDs (XIDs)** and **tuple versions**. Each row contains metadata about:
- **xmin**: The XID when the row was created.
- **xmax**: The XID when the row was deleted (if applicable).
- **Snapshot Isolation**: Each transaction takes a snapshot of the database at a certain point, and only committed changes visible at that snapshot can be read.

#### **MySQL (InnoDB) MVCC**
InnoDB (MySQL’s default storage engine) implements MVCC using **undo logs** to store previous versions of data. When a row is updated, the old data is moved to the undo log, and a new version is written to the table.

- **Rollback Segments**: Previous versions of the data are stored in rollback segments, which allow queries to view historical versions.
- **Purging**: MySQL periodically **purges** outdated versions to free up space.

---

### **Challenges with MVCC**

#### **1. Write Skew**
Write skew is a concurrency anomaly that arises in MVCC when two transactions concurrently update related data based on an outdated snapshot. It occurs because transactions operate on old versions, unaware of the latest changes made by others.

- **Example**:
  ```sql
  Transaction 1: Set B = A + 1;
  Transaction 2: Set A = B + 1;
  ```
  Both transactions read the same initial values, leading to inconsistent updates.

#### **2. Storage Overhead**
MVCC maintains older versions of data, which consumes significant storage over time. This issue is mitigated by:
- **Garbage Collection (GC)**: A background process that removes old, unused versions once they are no longer needed by any active transactions.

---

### **Conclusion**
MVCC is a powerful concurrency control technique that allows databases to handle multiple transactions efficiently while maintaining consistency. By maintaining multiple versions of data, it eliminates common concurrency issues but comes with trade-offs, such as increased storage usage and complexity in resolving anomalies like write skew.

---

### **Code Example (MVCC in PostgreSQL)**

```sql
-- Create a simple table to demonstrate MVCC
CREATE TABLE accounts (
    id serial PRIMARY KEY,
    balance int
);

-- Insert initial values
INSERT INTO accounts (balance) VALUES (100), (150), (200);

-- Start two concurrent transactions
BEGIN; -- Transaction 1
UPDATE accounts SET balance = balance - 50 WHERE id = 1;

BEGIN; -- Transaction 2
SELECT balance FROM accounts WHERE id = 1; -- This will read the old value

-- Transaction 1 commits
COMMIT;

-- Transaction 2 now commits after reading the outdated value
COMMIT;
```

In this scenario, Transaction 2 reads the balance before Transaction 1 commits, demonstrating the isolated snapshot Transaction 2 operates on.

