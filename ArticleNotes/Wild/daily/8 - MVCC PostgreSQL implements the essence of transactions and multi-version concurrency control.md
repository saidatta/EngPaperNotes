https://github-com.translate.goog/guevara/read-it-later/issues/1121?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en

> **Original**:  
> This article is forwarded from [Technology World](http://www.jasongj.com/sql/mvcc/), the original link is: [http://www.jasongj.com/sql/mvcc/](http://www.jasongj.com/sql/mvcc/)

---

## 1. PostgreSQL’s Mechanism for ACID

### 1.1 Database ACID

A **database transaction** has the following four properties:

1. **Atomicity**:  
   A transaction is an indivisible unit. Either *all operations* within it succeed, or none of them do. E.g., with an ATM withdrawal (card swipe + cash dispense), you can’t just swipe the card without dispensing cash.  
2. **Consistency**:  
   A transaction must transition the database from one **valid state** to another. Integrity constraints (like `a + b = 10`) must hold *before and after* the transaction completes.  
3. **Isolation**:  
   Concurrent transactions operate in isolation, preventing them from interfering with each other’s intermediate states.  
4. **Durability**:  
   Once a transaction successfully commits, its changes are **persisted** to disk and won’t be undone by subsequent failures.  

### 1.2 How ACID Is Implemented in PostgreSQL

| ACID        | Implementation Technology         |
|-------------|-----------------------------------|
| Atomicity   | MVCC                              |
| Consistency | Constraints (PK, FK, check, etc.) |
| Isolation   | MVCC                              |
| Durability  | WAL (Write-Ahead Logging)         |

- **MVCC** is used for both **Atomicity** and **Isolation** in PostgreSQL.
- **WAL** ensures **Durability**.
- Consistency is largely enforced by **constraints** (primary key, foreign key, etc.).

Below, we’ll focus on how **MVCC** works in PostgreSQL.

---

## 2. MVCC Principles in PostgreSQL

### 2.1 Transaction ID (XID)

In PostgreSQL:

- Every transaction has a **unique** **Transaction ID (XID)**.  
- Even if you don’t explicitly call `BEGIN`/`COMMIT`, a single SQL statement is a transaction with its own XID.  
- `txid_current()` can retrieve the current transaction ID.

### 2.2 Hidden Version Fields

Each **row** (a.k.a. *tuple* in PostgreSQL terminology) has **four hidden fields**:

1. **xmin**: The XID of the transaction that **inserted** this row.  
2. **xmax**: The XID of the transaction that **deleted** this row.  
3. **cmin**: The “command ID” (within a transaction) that created or changed the row.  
4. **cmax**: The “command ID” (within a transaction) that invalidated or deleted the row.

> **Note**: `cmin` and `cmax` reflect the **statement ordering** within the same transaction, starting from 0.

**Example**:  
```sql
CREATE TABLE test (
  id INTEGER,
  value TEXT
);

BEGIN;
SELECT txid_current(); -- Suppose it returns 3277
INSERT INTO test VALUES(1, 'a');
SELECT *, xmin, xmax, cmin, cmax FROM test;
-- id=1, value='a', xmin=3277, xmax=0, cmin=0, cmax=0
```
When you insert or update rows within the same transaction, you’ll see how `cmin` and `cmax` increment for each statement inside that transaction.

#### Updating a Row
PostgreSQL’s `UPDATE` under the hood does:
1. Mark the old row as **deleted** (setting `xmax` = current TxID).
2. Insert a **new row** with `xmin` = current TxID (the “updated” content).

---

## 3. MVCC Ensures Atomicity

Because each row’s changes (INSERT/DELETE/UPDATE) store the **current TxID** (either in `xmin` or `xmax`), PostgreSQL can:

- **Commit** a transaction → all those row changes become valid.  
- **Rollback** a transaction → those row changes are invalidated.

Hence, all operations in a transaction are “all-or-nothing,” thus providing **atomicity**.

---

## 4. MVCC Ensures Transaction Isolation

**Isolation** means concurrent transactions see consistent data without interfering with each other’s uncommitted changes.

### 4.1 SQL Standard Isolation Levels

| Isolation Level      | Dirty Read | Non-repeatable Read | Phantom Read |
|----------------------|-----------:|---------------------:|-------------:|
| Read Uncommitted     | possible   | possible            | possible     |
| Read Committed       | **no**     | possible            | possible     |
| Repeatable Read      | **no**     | **no**              | possible     |
| Serializable         | **no**     | **no**              | **no**       |

- PostgreSQL actually implements **three** levels:  
  1. **Read Uncommitted** → internally treated as **Read Committed**  
  2. **Read Committed**  
  3. **Repeatable Read** (which in PostgreSQL also avoids phantom reads, effectively providing a stronger Snapshot Isolation)

### 4.2 Read Committed with MVCC

**Read Committed** can only see rows that were committed *before* the current statement. If another transaction hasn’t committed, its changes aren’t visible. This is recorded in `pg_clog` (or `pg_xact` in newer versions) to indicate whether a TxID is committed or not.

### 4.3 Repeatable Read with MVCC

In PostgreSQL:
- A transaction sees only row versions committed *before* its own transaction start (Snapshot).
- Even if another transaction commits new changes later, the current transaction continues to see the data as it was at its start time.  
- This effectively prevents non-repeatable reads and even phantom reads in PostgreSQL’s implementation.

---

## 5. Advantages of PostgreSQL’s MVCC

1. **Readers do not block Writers**, and Writers do not block Readers. Concurrency is greatly improved.  
2. **Transaction rollback** can happen instantly by discarding uncommitted row versions (since old versions remain).  
3. Large updates are feasible without needing a huge rollback segment (as in Oracle or MySQL InnoDB).  
4. Each “row version” can be kept in the same table data structure. This simplifies reconstructing older snapshots.

---

## 6. Disadvantages of PostgreSQL’s MVCC

1. **Transaction ID is limited** (32-bit).  
   - Potential **wraparound** once the ID space is exhausted.  
   - **VACUUM** recycles old committed XIDs by marking them as “frozen” (XID=2). This ensures older data is always visible to new transactions.

2. **A large amount of “dead tuples”** can accumulate.  
   - **UPDATE** = “mark old row as deleted + insert new row.” Over time, many old row versions can bloat the table.  
   - **VACUUM** is needed to reclaim dead tuples and free disk space.  

### 6.1 VACUUM vs VACUUM FULL

- **VACUUM**:
  - Non-blocking, can run concurrently with reads/writes.  
  - Marks dead tuples as reusable but does not shrink the physical file size.  
- **VACUUM FULL**:
  - Exclusive lock required.  
  - Compacts live tuples into a new file, discarding old file → can return space to the operating system but blocks other operations.

---

## 7. Summary

PostgreSQL implements **ACID** using:

- **MVCC** for **Atomicity** and **Isolation**  
- **Constraints** for **Consistency**  
- **WAL** for **Durability**  

**MVCC** in PostgreSQL:

1. Each row has hidden fields (`xmin`, `xmax`, `cmin`, `cmax`).  
2. **INSERT** sets `xmin = current TxID`.  
3. **DELETE** sets `xmax = current TxID`.  
4. **UPDATE** is logically a **DELETE** of the old row + **INSERT** of a new row version.  

**VACUUM** is essential to:

- Prevent transaction ID wraparound.  
- Reclaim space from dead tuples.  

Hence, MVCC offers **high concurrency** (no read/write blocking), but requires periodic vacuum maintenance.

```mermaid
flowchart LR
    A((Transaction)) --> B[INSERT: xmin = txID, xmax=0]
    A --> C[DELETE: xmax = txID of the deleting transaction]
    A --> D[UPDATE = old row xmax=txID, new row xmin=txID]
    D -->|New row version| Table
    D -->|Old row version still stored| Dead Tuples
    E((VACUUM)) -->|Reclaims space & marks old XID| D
```

**Key Takeaways**:
- PostgreSQL’s **MVCC** helps scale read/write concurrency.  
- Understanding row versioning is crucial for performance (preventing table bloat) and correct behavior under different isolation levels.  

---

## 8. Further Reading

- [PostgreSQL Documentation – MVCC](https://www.postgresql.org/docs/current/mvcc.html)  
- [VACUUM in PostgreSQL](https://www.postgresql.org/docs/current/sql-vacuum.html)  
- [Transaction ID Wraparound Issues](https://www.postgresql.org/docs/current/routine-vacuuming.html#VACUUM-FOR-WRAPAROUND)  
- [SQL Standard Isolation Levels & PostgreSQL Implementation](https://www.postgresql.org/docs/current/transaction-iso.html)  

```