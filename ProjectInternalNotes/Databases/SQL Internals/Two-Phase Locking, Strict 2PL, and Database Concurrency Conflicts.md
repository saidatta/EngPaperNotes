https://www.youtube.com/watch?v=b25JiQyQLNs&list=PLwrbo0b_XxA8BaxKRHuGHAQsBrmhYBsh1
---

### **Introduction to Database Concurrency Control**

Databases must handle multiple **concurrent transactions**, which can result in conflicts. The goal of concurrency control is to ensure that transactions **adhere to ACID** properties: **Atomicity, Consistency, Isolation,** and **Durability**. These properties ensure that transactions are handled reliably even in the presence of concurrent execution.

Let's break down the essential concepts:

---

### **ACID Properties (Recap)**

1. **Atomicity**:
   - **Definition**: Transactions are "all or nothing"; either all operations within the transaction are completed, or none are.
   - **Example**: If a transaction updates multiple rows in a table, either all rows are updated, or if an error occurs, all updates are rolled back.
   
   \[
   \text{Transaction outcome} = 
   \begin{cases} 
   \text{Success: Commit all operations} \\
   \text{Failure: Rollback all operations}
   \end{cases}
   \]

2. **Consistency**:
   - **Definition**: The database remains in a consistent state before and after a transaction. Any rules (constraints) defined in the schema, like "balance cannot go negative," must hold true after every transaction.
   - **Example**: Ensuring that a transfer from one account to another does not result in money disappearing or appearing out of nowhere.

3. **Isolation**:
   - **Definition**: Concurrent transactions should not interfere with each other. Even if transactions run simultaneously, they should appear as if they were executed serially.
   - **Example**: If two transactions are updating the same row, isolation ensures that each transaction operates as though it had exclusive access.

4. **Durability**:
   - **Definition**: Once a transaction is committed, its effects are permanent, even in the event of a system crash.
   - **Example**: If a transaction updates a row and the transaction commits, that update will survive power outages or system failures.
---
### **Concurrency Problems and Conflicts**
With concurrency, several problems can arise:
1. **Unrepeatable Reads (Read-Write Conflict)**:
   - **Description**: A transaction reads a value, another transaction modifies it, and the original transaction reads the modified value, resulting in inconsistent reads within the same transaction.
   - **Example**:
     - T1 reads A = 0.
     - T2 updates A = 2.
     - T1 reads A again and gets 2 (instead of the original 0).
   - **Problem**: T1 observes different values for the same object in the same transaction.
   
   ```plaintext
   T1: Read(A) = 0
   T2: Write(A) = 2
   T1: Read(A) = 2  (Unrepeatable Read)
   ```

2. **Dirty Reads (Write-Read Conflict)**:
   - **Description**: A transaction reads uncommitted data written by another transaction. If the writing transaction is rolled back, the reading transaction has seen invalid data.
   - **Example**:
     - T1 writes A = 2 but later aborts.
     - T2 reads A = 2 while T1 is uncommitted.
     - T1 aborts, and A is reverted to its original value.
   - **Problem**: T2 read a value that was never committed.

   ```plaintext
   T1: Write(A) = 2 (Uncommitted)
   T2: Read(A) = 2 (Dirty Read)
   T1: Abort (A should be reverted)
   ```

3. **Write-Write Conflict (Lost Updates)**:
   - **Description**: Two transactions write to the same data item. The last write overwrites the first, leading to lost updates.
   - **Example**:
     - T1 writes A = 10.
     - T2 writes A = 9.
     - The final value of A depends on which transaction commits last, but the earlier update may be lost.
   
   ```plaintext
   T1: Write(A) = 10
   T2: Write(A) = 9 (Overwrites T1's change)
   ```

---
### **Conflict Serializable Schedules**
To avoid these conflicts, a **conflict serializable schedule** ensures that **concurrent transactions** behave as if they were executed **one after the other**. Even though operations are interleaved, the result must be **consistent** with some serial execution order.
A schedule is conflict-serializable if it can be transformed into a serial schedule by swapping non-conflicting operations.

---
### **Locking Mechanisms**
**Locks** prevent multiple transactions from accessing the same data simultaneously in conflicting ways.
1. **Shared Lock (S)**: Allows multiple transactions to **read** a data item, but no transaction can **write** to it.
2. **Exclusive Lock (X)**: Allows a transaction to **write** a data item, preventing others from reading or writing it.
---
### **Two-Phase Locking (2PL)**
**Two-Phase Locking (2PL)** ensures conflict-serializable schedules by breaking transaction execution into two distinct phases:
#### **1. Growing Phase**:
   - A transaction can acquire locks (shared or exclusive) on data items but **cannot release** any locks.
   \[
   \text{Growing Phase: Acquire Locks (S or X)}
   \]

#### **2. Shrinking Phase**:
   - After the first lock is released, the transaction cannot acquire any more locks; it can only release existing locks.

   \[
   \text{Shrinking Phase: Release Locks (No New Locks)}
   \]

#### **Example of Two-Phase Locking**:

```plaintext
T1: Lock(A)      (Growing Phase)
T1: Lock(B)      (Growing Phase)
T1: Unlock(A)    (Shrinking Phase)
T1: Unlock(B)    (Shrinking Phase)
```

This ensures that no transaction **releases a lock** and then **acquires another lock**, preventing certain concurrency issues.

---

### **Strict Two-Phase Locking (Strict 2PL)**

**Strict 2PL** is a stricter version where a transaction holds **all its exclusive locks** until it commits or aborts. This prevents **cascading aborts** by ensuring that **no uncommitted data** is read by other transactions.

- **Benefit**: Eliminates **dirty reads** and cascading aborts.

#### **Cascading Abort Example**:

1. T1 acquires exclusive lock on A, writes A = 10.
2. T2 reads A = 10 (dirty read).
3. T1 aborts, reverting A to its original value.
4. T2 must also abort due to dirty read.

With **strict 2PL**, T2 could not have read A before T1 committed.

---

### **Deadlocks in Locking Systems**

**Deadlocks** occur when two or more transactions hold locks and are waiting for each other to release locks, causing the system to halt.

#### **Deadlock Example**:

```plaintext
T1: Lock(A)         T2: Lock(B)
T1: Wait for B      T2: Wait for A (Deadlock)
```

- **Deadlock Prevention**: 
   - Use **timeout mechanisms**.
   - **Wait-Die** or **Wound-Wait** schemes can be used to avoid deadlocks by enforcing a strict ordering on lock acquisition.
   
- **Deadlock Detection**:
   - Use **wait-for graphs** to detect cycles in the lock waiting graph. If a cycle is detected, one transaction is aborted to resolve the deadlock.

---

### **Equations for Serializable Schedules**

To verify if a schedule is conflict-serializable, we can represent it using **precedence graphs**:

1. **Precedence Graph**: 
   - Nodes represent transactions.
   - Directed edges represent dependencies between transactions (e.g., T1 must precede T2 if T1 writes a value that T2 reads).

2. **Conflict-Serializable if No Cycles**:
   - If the precedence graph contains no cycles, the schedule is conflict-serializable.

---

### **Practical Applications in Databases**

- **PostgreSQL**: Uses **strict 2PL** for certain isolation levels and includes **deadlock detection** mechanisms that terminate a transaction when a deadlock is detected.
- **MySQL (InnoDB)**: Utilizes **row-level locking** and also provides **deadlock detection**.

---

### **Conclusion**

Two-phase locking (2PL) and its stricter version, strict 2PL, provide robust mechanisms for ensuring conflict-serializable schedules. While 2PL avoids many concurrency issues, it can lead to deadlocks, requiring deadlock detection or prevention strategies. Strict 2PL further enhances reliability by preventing cascading aborts, though at the cost of potential performance overhead.

By understanding these concepts and implementing appropriate locking strategies, databases can ensure that transactions execute concurrently without compromising consistency or isolation.

