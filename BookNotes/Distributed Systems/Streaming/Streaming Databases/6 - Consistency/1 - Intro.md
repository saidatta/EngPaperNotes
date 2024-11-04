## Overview

In this chapter, we delve into the concept of **consistency** in the context of databases and streaming systems. While traditional databases offer strong consistency guarantees, streaming systems often provide weaker forms of consistency, such as **eventual consistency**. We'll explore the challenges this presents and how newer streaming systems aim to offer stronger consistency guarantees.

---

## Key Concepts

- **Consistency**: Ensuring that the data in a system remains accurate and reliable, adhering to predefined rules and constraints.
- **Eventual Consistency**: A model where, in the absence of new updates, all accesses eventually return the last updated value.
- **Internal Consistency**: A stronger consistency model where every output is the correct output for a subset of the inputs.

---

## The Challenge of Consistency in Streaming Systems

### Traditional Databases

- Provide strong consistency guarantees.
- Queries return results consistent with the input data.
- Transactions ensure atomicity, consistency, isolation, and durability (ACID properties).

### Streaming Systems

- Data arrives **late** and **out of order**.
- Emphasis on **low latency** and **high throughput**.
- Classical stream processors (e.g., Flink, ksqlDB) guarantee **eventual consistency**.

### Implications

- **Eventual consistency** can be confusing and counterintuitive, especially for those accustomed to database systems.
- Non-windowed data and synchronization requirements pose challenges in stream processing.

---

## A Toy Example: Bank Transactions

To illustrate the challenges of consistency in streaming systems, we'll use a toy example adapted from Jamie Brandon's blog on internal consistency.

### Scenario

- A bank with **10 accounts**.
- Accounts continuously transfer **$1** to other accounts.
- Goal: Ensure that the **sum of all account balances** remains **zero** at all times.

### Table Representation

| Account | Starting Value | Transaction 1 | Transaction 2 | Transaction 3 | Transaction 4 |
|---------|----------------|---------------|---------------|---------------|---------------|
| 1       | $0             | –$1           | –$2           | –$3           | –$2           |
| 2       | $0             | $1            | $1            | $2            | $2            |
| 3       | $0             | $0            | $1            | $1            | $0            |
| **Sum** | $0             | $0            | $0            | $0            | $0            |

- **Invariant**: The sum of all account balances should always be **zero**.
- **Consistency Test**: Check if the sum deviates from zero at any point.

---

## Implementing the Toy Example

### 1. Setting Up Transactions

#### Python Code to Generate Transactions

```python
import datetime
import json
import random
import time
from kafi.kafi import Cluster

# Initialize Kafka cluster
c = Cluster("local")
c.create("transactions", partitions=1)
p = c.producer("transactions")

# Seed random number generator for reproducibility
random.seed(42)

# Generate 10,000 transactions
for id_int in range(0, 10000):
    row_str = json.dumps({
        "id": id_int,
        "from_account": random.randint(0, 9),
        "to_account": random.randint(0, 9),
        "amount": 1,
        "ts": datetime.datetime.now().isoformat(sep=" ", timespec="milliseconds")
    })
    print(row_str)
    p.produce(row_str, key=str(id_int))
    time.sleep(0.01)  # Sleep for 10 milliseconds

p.close()
```

- **Explanation**:
  - Creates a Kafka topic named `transactions` with one partition.
  - Generates 10,000 transactions where each transaction transfers $1 from one random account to another.
  - Transactions are produced every 10 milliseconds.

#### Sample Transaction Message

```json
{
  "id": 42,
  "from_account": 3,
  "to_account": 0,
  "amount": 1,
  "ts": "2023-10-24 23:27:57.603"
}
```

- **Fields**:
  - `id`: Unique identifier for the transaction.
  - `from_account`: Account number initiating the transfer.
  - `to_account`: Account number receiving the transfer.
  - `amount`: Amount transferred ($1).
  - `ts`: Timestamp of the transaction.

---

### 2. Analyzing Transactions with SQL

#### Defining the Transactions Table

Assuming we're using a streaming database or a stream processor with SQL capabilities, we first define the `transactions` table.

```sql
CREATE TABLE transactions (
  id INT,
  from_account INT,
  to_account INT,
  amount INT,
  ts TIMESTAMP
) WITH (
  'connector' = 'kafka',
  'topic' = 'transactions',
  'properties.bootstrap.servers' = 'localhost:9092',
  'format' = 'json'
);
```

---

#### Creating Views for Credits and Debits

##### Credits View

```sql
CREATE VIEW credits AS
SELECT
  to_account AS account,
  SUM(amount) AS credits
FROM transactions
GROUP BY to_account;
```

- **Purpose**: Calculates the total amount credited to each account.

##### Debits View

```sql
CREATE VIEW debits AS
SELECT
  from_account AS account,
  SUM(amount) AS debits
FROM transactions
GROUP BY from_account;
```

- **Purpose**: Calculates the total amount debited from each account.

---

#### Calculating Account Balances

```sql
CREATE VIEW balance AS
SELECT
  credits.account AS account,
  credits.credits - debits.debits AS balance
FROM credits
FULL OUTER JOIN debits ON credits.account = debits.account;
```

- **Explanation**:
  - **Full Outer Join**: Ensures all accounts are included, even if they only have credits or debits.
  - **Balance Calculation**: Subtracts debits from credits for each account.

---

#### Calculating Total Balance

```sql
CREATE VIEW total AS
SELECT SUM(balance) AS total_balance FROM balance;
```

- **Invariant Check**: The `total_balance` should always be **zero**.

---

### 3. Testing Consistency in Different Systems

We'll examine how various stream processing systems handle this scenario.

---

## Understanding Consistency Models

### Eventual Consistency

- **Definition**: All updates will propagate through the system eventually, and all replicas will be consistent **if no new updates are made**.
- **Implications**:
  - Temporary inconsistencies are acceptable.
  - Suited for applications where immediate consistency is not critical.

### Internal Consistency

- **Definition**: Every output is the correct output for a subset of the inputs.
- **Benefits**:
  - Provides stronger guarantees than eventual consistency.
  - More predictable and aligns with traditional database consistency expectations.

---

## Challenges in Stream Processing Systems

### Late and Out-of-Order Data

- **Problem**: In streaming contexts, data may arrive late or out of sequence.
- **Impact**: Can lead to inconsistencies in aggregations and calculations.

### Synchronization

- **Issue**: Lack of synchronization mechanisms in classical stream processors makes it difficult to maintain invariants like the total balance.

### Non-Windowed Data

- **Complication**: Without windowing, aggregations over unbounded streams can be problematic.
- **Solution**: Need mechanisms to handle infinite data streams.

---

## Evaluating Different Stream Processing Systems

### Classical Stream Processors (e.g., Flink, ksqlDB)

- **Consistency Model**: Eventual consistency.
- **Behavior**:
  - May not maintain the invariant (`total_balance = 0`) at all times.
  - Suitable for windowed aggregations and approximate computations.

### Streaming Databases with Stronger Consistency (e.g., Materialize, RisingWave, Pathway)

- **Consistency Model**: Internal consistency.
- **Behavior**:
  - Aim to maintain invariants like `total_balance = 0` consistently.
  - Provide outputs that are correct for subsets of inputs.

---

## Mathematical Explanation

### Transaction Processing

- **Variables**:
  - \( A_i \): Balance of account \( i \).
  - \( C_i \): Total credits to account \( i \).
  - \( D_i \): Total debits from account \( i \).
- **Balance Calculation**:
  \[
  A_i = C_i - D_i
  \]
- **Invariant (Total Balance)**:
  \[
  \sum_{i=1}^{N} A_i = \sum_{i=1}^{N} (C_i - D_i) = 0
  \]
  - Since every debit is matched by a credit, the total balance should remain zero.

---

### Consistency Violation in Eventual Consistency

- **Scenario**:
  - Debits and credits are processed asynchronously.
  - Delays or out-of-order processing can lead to temporary imbalances.
- **Mathematical Representation**:
  - At time \( t \), the observed total balance:
    \[
    \text{Total Balance}_t = \sum_{i=1}^{N} (C_i(t) - D_i(t)) \neq 0
    \]
  - Over time, as all transactions are processed, the system converges to:
    \[
    \lim_{t \to \infty} \text{Total Balance}_t = 0
    \]

---

## Code Examples in Different Systems

### Example in Apache Flink SQL

```sql
-- Define the transactions table
CREATE TABLE transactions (
  id INT,
  from_account INT,
  to_account INT,
  amount INT,
  ts TIMESTAMP(3),
  WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH (
  'connector' = 'kafka',
  'topic' = 'transactions',
  'properties.bootstrap.servers' = 'localhost:9092',
  'format' = 'json'
);

-- Define the credits view
CREATE VIEW credits AS
SELECT
  to_account AS account,
  SUM(amount) AS credits
FROM transactions
GROUP BY to_account;

-- Define the debits view
CREATE VIEW debits AS
SELECT
  from_account AS account,
  SUM(amount) AS debits
FROM transactions
GROUP BY from_account;

-- Define the balance view
CREATE VIEW balance AS
SELECT
  COALESCE(credits.account, debits.account) AS account,
  COALESCE(credits, 0) - COALESCE(debits, 0) AS balance
FROM credits
FULL OUTER JOIN debits ON credits.account = debits.account;

-- Define the total balance view
CREATE VIEW total_balance AS
SELECT SUM(balance) AS total_balance FROM balance;
```

- **Note**: Due to eventual consistency, `total_balance` may not always be zero.

---

### Example in Materialize

```sql
-- Create source from Kafka topic
CREATE SOURCE transactions_raw
FROM KAFKA BROKER 'localhost:9092' TOPIC 'transactions'
FORMAT AVRO USING SCHEMA '...'
ENVELOPE NONE;

-- Create the transactions view
CREATE MATERIALIZED VIEW transactions AS
SELECT * FROM transactions_raw;

-- Create credits, debits, balance, and total_balance views as before
-- Materialize maintains internal consistency, so total_balance should remain zero
```

- **Benefit**: Materialize provides stronger consistency, maintaining the invariant.

---

## Summary

- **Classical Stream Processors**:
  - Provide eventual consistency.
  - Suitable for use cases where temporary inconsistencies are acceptable.
  - Challenges with maintaining invariants in non-windowed, synchronized scenarios.

- **Streaming Databases with Stronger Consistency**:
  - Offer internal consistency.
  - Better align with traditional database expectations.
  - More predictable behavior in maintaining invariants.

---

## Conclusion

- **Convergence of Streaming and Database Worlds**:
  - For stream processing systems to effectively handle non-typical, non-windowed use cases, they need to offer stronger consistency guarantees.
  - This convergence requires balancing low latency and high throughput with stronger consistency models.

- **Future Considerations**:
  - Assessing whether classical stream processors can support stronger consistency without significant performance trade-offs.
  - Exploring the potential benefits of adopting internal consistency in more streaming systems.

---

## References

- **Jamie Brandon's Blog**: [Internal Consistency in Streaming Systems](https://scattered-thoughts.net/writing/internal-consistency/)
- **Apache Flink Documentation**: [Stateful Stream Processing](https://ci.apache.org/projects/flink/flink-docs-stable/dev/stream/state/state.html)
- **Materialize Documentation**: [Streaming Consistency Model](https://materialize.com/docs/overview/architecture/#consistency-model)

---

## Tags

#Consistency #StreamingSystems #EventualConsistency #InternalConsistency #StreamProcessing #DataEngineering #StaffPlusNotes #DistributedSystems

---

## Additional Notes

- **Eventual Consistency is Not Always Enough**:
  - For critical applications like banking, stronger consistency is necessary.
  - Streaming databases that offer internal consistency are better suited for such use cases.

- **Trade-offs**:
  - Stronger consistency may come at the cost of increased latency or reduced throughput.
  - System designers need to balance these trade-offs based on application requirements.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.