## Overview
This note explores:
1. **What a transaction is** and the **ACID** properties.
2. **Distributed transactions**: why they arise, how they differ from local transactions.
3. **XA protocol** and **2PC (Two-Phase Commit)**.
4. **MySQL** implementing XA transactions.
5. **3PC (Three-Phase Commit)** as an improvement on 2PC.
6. **TCC** (Try-Confirm-Cancel) approach and its pros/cons.
7. **Final takeaway**: there is no perfect solution—choose based on business needs.
---
## 1. Transactions
### 1.1 Definition
A **transaction** is a program-execution unit that:
- Reads and/or updates data in a database.
- Must be *all or nothing*.
#### Example (Simple):
```sql
BEGIN TRANSACTION;
  -- 1) Add user "Zhang San"
  -- 2) Add permission for "Zhang San"
END TRANSACTION;
```
Either **both** operations succeed, or both fail.

### 1.2 ACID Properties
1. **Atomicity**:  
   All operations in a transaction succeed or all fail.
2. **Consistency**:  
   Ensures that the database transitions from one valid state to another.
3. **Isolation**:  
   Concurrent transactions do not interfere with one another’s intermediate states.
4. **Durability**:  
   Once a transaction commits, its effects persist even in case of failure.

---

## 2. Distributed Transactions

With the growth of **distributed systems**, an activity (transaction) may span multiple services on different hosts. E.g.:

```sql
BEGIN TRANSACTION;
  -- 1) Add user "Zhang San" (User Service, Host A)
  -- 2) Add "Zhang San's" permission (Permission Service, Host B)
END TRANSACTION;
```

### 2.1 Challenges
- **Network issues**: partial successes, partial failures.
- **Coordination** among multiple services or databases is needed to maintain atomicity.

---

## 3. XA Protocol

Originally from **Tuxedo** and standardized by X/Open.  
- All major databases (Oracle, DB2, MySQL, etc.) implement **XA** for distributed transactions.
- **XA** uses a **two-phase commit (2PC)** to coordinate distributed resources.

### 3.1 Key Concepts
- **TM** (Transaction Manager): Coordinates and manages transactions, often part of the application server.
- **RM** (Resource Manager): Manages local resources. In DB world, the DB is the RM (Oracle, MySQL, etc.).
- **AP** (Application Program): The actual business code.

These combine into the **DTP (Distributed Transaction Processing)** model from X/Open.

---

## 4. Two-Phase Commit (2PC)

**2PC** is the standard approach in XA. It has two phases:

1. **Prepare Phase**:
   - TM sends a “prepare” request to each RM.  
   - Each RM must decide if it can guarantee commit. If yes, it “prepares” by writing data to stable storage (undo/redo logs).
   - RMs respond with “ready” or “no.”
2. **Commit Phase**:
   - If **all** RMs respond “ready,” TM instructs them to **commit**.
   - If **any** RM cannot commit, TM instructs them to **rollback**.
   - Once an RM is told “commit,” it must not fail (theoretically).

---

## 5. XA in MySQL (Example)

MySQL’s InnoDB implements XA with statements:

```sql
XA START 'createuser';
-- (SQL statements)
XA END 'createuser';
XA PREPARE 'createuser';
XA COMMIT 'createuser';  -- or XA ROLLBACK 'createuser';
```

- `XA START`: Begin an XA transaction, put it in `ACTIVE` state.
- (Run SQL statements)
- `XA END`: End transaction, switch to `IDLE` state.
- `XA PREPARE`: Prepare transaction, switch to `PREPARED` state.
- `XA COMMIT` or `XA ROLLBACK`: finalize.

**2PC** means we do an extra step (`PREPARE`) between starting and committing the transaction.

---

## 6. Shortcomings of 2PC

- **Synchronous & Blocking**:
  - RMs lock resources during transaction execution, suitable only for short, quick transactions under moderate concurrency.
- **Deadlock Risk**:
  - If the transaction manager fails, or the final commit/rollback is lost, resources might stay locked indefinitely.
- Performance overhead under **high concurrency**.

---

## 7. Three-Phase Commit (3PC)

**3PC** is an improvement on 2PC, adding a **timeout** mechanism to reduce indefinite blocking:

1. **CanCommit**:  
   - Similar to 2PC’s prepare, but if the coordinator sees a negative or times out, it aborts early.
2. **PreCommit**:  
   - If all participants are positive, coordinator sends “pre-commit.”
   - Participants get ready to commit but not finalize.
3. **DoCommit**:  
   - On success, coordinator sends “do commit.”

**3PC** can still have complexities (e.g., data inconsistency from timeouts). More messages can reduce performance in certain scenarios.

---

## 8. TCC (Try-Confirm-Cancel)

**TCC** is effectively an application-level 2PC with compensation, typically:

1. **Try**:
   - Reserve resources, check if we can proceed, lock or hold them in a “tentative” state.
2. **Confirm**:
   - If all is well, confirm the usage of those resources. Should be idempotent (retries safe).
3. **Cancel**:
   - If any step fails, release or roll back the resources. Also must be idempotent.

### 8.1 Advantages

- **No global resource blocking** as with 2PC.  
- Potentially better **performance** than pure 2PC.  

### 8.2 Disadvantages

- Requires **custom business logic** to lock/reserve resources, handle compensation.  
- **Idempotency** in confirm/cancel is not trivial.  
- More complex to implement, longer dev time, and code is less reusable.

---

## 9. Final Thoughts

No universal silver bullet. The choice depends on:

- **Business** needs for consistency vs. performance.
- **Tolerance** for partial failures, compensation, or blocking.
- **Data criticality**: short transactions vs. batch processing.

**2PC** is simpler but can block.  
**3PC** adds timeouts but more complexity.  
**TCC** moves logic into the application, offering more concurrency but also more dev overhead.

**Conclusion**: Evaluate your scenario. If rigid strong consistency is essential and transactions are short, **XA/2PC** might suffice. If you can do business-level compensation, **TCC** could be better.  

**Key Takeaway**: **Use the right distributed transaction approach for your situation**—there’s no one-size-fits-all.

---
## 10. Related Questions

1. **How to handle distributed transactions?**  
   - Evaluate patterns: 2PC, 3PC, TCC, etc.  
   - Align with your concurrency, business data sensitivity, and performance constraints.  
   - Possibly use a combination of eventual consistency + compensation or rigid protocols if needed.

2. **What are the characteristics of TCC mode?**  
   - **Try**: Reserve resources, ensure feasibility.  
   - **Confirm**: Confirm operation using the reserved resources.  
   - **Cancel**: Rollback or compensate if any step fails.  
   - Minimizes blocking but requires custom logic and idempotency in all phases.

3. **What are the shortcomings of 2PC?**  
   - Blocking protocol: if coordinator or participant fails, it can lock resources indefinitely.  
   - High overhead in concurrency scenarios.  
   - Potential for **deadlocks** or timeouts if final commit/rollback is never received.

```