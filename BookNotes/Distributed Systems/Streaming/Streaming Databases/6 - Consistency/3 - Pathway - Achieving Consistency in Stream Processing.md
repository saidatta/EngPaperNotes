## Overview

In this section, we explore **Pathway**, a stream processing library for Python, and how it addresses consistency in stream processing. Unlike full-fledged streaming databases, Pathway allows developers to author streaming pipelines using Python, providing SQL-like capabilities within Python code.

---

## Key Concepts

- **Consistency in Stream Processing**: Ensuring that the output of a stream processing system remains accurate and reliable, even when data arrives out-of-order or with delays.
- **Pathway**: A Python library for stream processing that emphasizes consistency and correctness, even in the face of out-of-order messages.

---

## Implementing the Toy Bank Example with Pathway

### Objective

- Simulate a banking system where transactions are processed, and ensure that the total balance across all accounts remains **zero** at all times.

### Setup

1. **Kafka Cluster**: We use Kafka as the messaging platform to simulate streaming data.
2. **Transactions**: Generate 10,000 transactions where each transaction transfers \$1 from one random account to another.

### Code Implementation

#### 1. Importing Libraries and Configuring Kafka

```python
#!/usr/bin/python

import pathway as pw

# Kafka connection settings
rdkafka_settings = {
    "bootstrap.servers": "localhost:56512",
    "group.id": "pw",
    "session.timeout.ms": "6000"
}
```

- **Explanation**:
  - Import the `pathway` library.
  - Define Kafka settings for connecting to the Kafka cluster.

#### 2. Defining the Input Schema

```python
class InputSchema(pw.Schema):
    id: int
    from_account: int
    to_account: int
    amount: int
    ts: str
```

- **Explanation**:
  - Define a schema `InputSchema` that matches the structure of the transaction data.

#### 3. Reading Transactions from Kafka

```python
# t represents the streaming transactions coming from Kafka
t = pw.io.kafka.read(
    rdkafka_settings,
    topic="transactions",
    schema=InputSchema,
    format="json",
    autocommit_duration_ms=1000
)
```

- **Explanation**:
  - Use `pw.io.kafka.read` to read streaming data from the `transactions` topic in Kafka.
  - Data is formatted as JSON and conforms to `InputSchema`.

#### 4. Computing Credits

```python
# The credits table
credits = pw.sql(
    """
    SELECT to_account, SUM(amount) AS credits
    FROM T
    GROUP BY to_account
    """,
    T=t
)
```

- **Explanation**:
  - Use SQL-like syntax to compute the total credits for each `to_account`.

#### 5. Computing Debits

```python
# The debits table
debits = pw.sql(
    """
    SELECT from_account, SUM(amount) AS debits
    FROM T
    GROUP BY from_account
    """,
    T=t
)
```

- **Explanation**:
  - Compute the total debits for each `from_account`.

#### 6. Calculating Balance per Account

```python
# The balance after the debits are subtracted from credits
balance = pw.sql(
    """
    SELECT CC.to_account AS account, credits - debits AS balance
    FROM CC
    JOIN DD ON CC.to_account = DD.from_account
    """,
    CC=credits,
    DD=debits
)
```

- **Explanation**:
  - Join `credits` and `debits` to compute the balance for each account.
  - Aligns `to_account` from credits with `from_account` from debits.

#### 7. Computing Total Balance

```python
# The total that will be used to emit to Kafka
total = pw.sql(
    """
    SELECT SUM(balance) AS total FROM BB
    """,
    BB=balance
)
```

- **Explanation**:
  - Sum all account balances to compute the total balance.
  - According to the invariant, this should always be **zero**.

#### 8. Writing the Result to Kafka

```python
# The result in a Kafka topic named total_pathway
pw.io.kafka.write(
    total,
    rdkafka_settings=rdkafka_settings,
    topic_name='total_pathway',
    format='json'
)
```

- **Explanation**:
  - Write the computed `total` balance back to Kafka in the `total_pathway` topic.

#### 9. Running the Dataflow

```python
# Run the dataflow asynchronously
pw.run()
```

- **Explanation**:
  - Execute the streaming dataflow defined above.

---

### Execution and Results

#### Running the Application

- When the application runs, it processes the streaming transactions in real-time.
- It computes the credits, debits, balances, and total balance as data flows through the pipeline.

#### Output

- The result is a **single record** written to the `total_pathway` Kafka topic.

##### Sample Output Message

```json
{
    "total": 0,
    "diff": 1,
    "time": 1698960910176
}
```

- **Fields**:
  - `"total"`: The total balance across all accounts (should be **0**).
  - `"diff"`: Indicates the difference since the last computation (may vary based on implementation).
  - `"time"`: Timestamp of the computation.

#### Visualization

- **Graph**: Since only one message is produced, the visualization shows a single data point at zero.
- This mirrors the behavior observed with **Materialize**, indicating strong consistency.

---

## Handling Out-of-Order Messages

### Scenario

- Modify the transaction generator to produce **out-of-order messages**.
- Approximately **1,000 out of 10,000** messages are out-of-order.

### Impact on Systems

- **Flink SQL, ksqlDB, Proton**:
  - Results remain similar to the original execution.
  - Inconsistencies persist due to eventual consistency and lack of synchronization.

- **RisingWave, Materialize, Pathway**:
  - Results remain consistent and correct.
  - Demonstrates robustness in handling out-of-order data.

---

## Understanding Why Pathway Maintains Consistency

### Synchronization in Joins

- **Pathway** synchronizes inputs before performing the JOIN operation.
- Ensures that corresponding credits and debits are matched correctly.

### Avoiding Race Conditions

- By synchronizing data streams, Pathway prevents race conditions that lead to incorrect intermediate results.
- Guarantees that the balance computation always reflects accurate account states.

### Mathematical Explanation

- **Invariant**: The sum of all account balances (\( \sum_{i} A_i = 0 \)).

#### Synchronization Mechanism

- **Credits and Debits Alignment**:
  - For each account \( i \), ensure that \( C_i \) and \( D_i \) are updated before computing \( A_i = C_i - D_i \).
- **Preventing Early Emission**:
  - Results are emitted only after both credits and debits have been accounted for each account.

#### Handling Out-of-Order Data

- **Timestamps**: Use timestamps to order transactions correctly.
- **Buffering**: Temporarily hold data until it can be correctly processed.
- **Deterministic Processing**: Ensures that the order of operations leads to consistent results.

---

## Key Takeaways

- **Consistency Achieved Without a Full-Fledged Streaming Database**:
  - Pathway demonstrates that stream processors can achieve strong consistency.
- **Python-Based Stream Processing**:
  - Provides flexibility and ease of use for developers familiar with Python.
- **Importance of Synchronization**:
  - Proper synchronization of data streams is critical in maintaining consistency.

---

## Comparison with Other Systems

### Why Eventual Consistency Systems Fail the Toy Example

- **Lack of Synchronization**:
  - JOIN operations combine inputs without ensuring alignment.
- **Early Emission from Non-Monotonic Operators**:
  - Results are emitted as soon as partial data is available, leading to incorrect intermediate results.

### How Internally Consistent Systems Succeed

- **Synchronization of Inputs**:
  - Align data streams before processing JOINs and aggregations.
- **Controlled Emission**:
  - Emit results only when sufficient information is available to guarantee correctness.

---

## Mathematical Modeling

### Race Conditions in Eventual Consistency Systems

- **Scenario**: Credits and debits are processed at different rates.
- **Result**: Incorrect balances due to mismatched data.

#### Example

- **Transactions**:
  - \( T_1 \): Account 0 transfers \$1 to Account 1.
  - \( T_2 \): Account 0 transfers \$1 to Account 2.
- **Possible Processing Sequence**:
  - Credits for \( T_1 \) and \( T_2 \) are processed before debits.
  - Balances for Accounts 1 and 2 show credits without corresponding debits from Account 0.
- **Incorrect Total**:
  - Total balance becomes positive, violating the invariant.

### Synchronization in Pathway

- **Ensures that for each transaction**:
  - The debit from the `from_account` and the credit to the `to_account` are considered together.
- **Mathematically**:
  - For each transaction \( T_k \):
    \[
    A_{from\_account} = A_{from\_account} - amount_k
    \]
    \[
    A_{to\_account} = A_{to\_account} + amount_k
    \]
- **Total Balance**:
  - Sum of all \( A_i \) remains zero:
    \[
    \sum_{i} A_i = 0
    \]

---

## Best Practices

1. **Synchronize Data Streams**:
   - Ensure that JOIN operations are performed on aligned datasets.
2. **Handle Out-of-Order Data**:
   - Use buffering and timestamp ordering to process data correctly.
3. **Control Emission of Results**:
   - Emit results only when they are accurate and consistent.
4. **Leverage Language Familiarity**:
   - Use languages like Python to simplify development and debugging.
5. **Test with Realistic Scenarios**:
   - Include out-of-order and delayed messages in testing to ensure robustness.

---

## Conclusion

- **Pathway** showcases that strong consistency in stream processing can be achieved without a full streaming database.
- Synchronization and careful handling of data streams are key to maintaining consistency.
- Developers can leverage Python's simplicity and Pathway's capabilities to build robust stream processing applications.

---

## References

- **Pathway Documentation**: [Pathway Official Documentation](https://pathway.com/docs/)
- **Kafka Documentation**: [Apache Kafka](https://kafka.apache.org/)
- **Streaming Consistency Models**: [Understanding Consistency in Stream Processing](https://example.com/streaming-consistency)

---

## Tags

#StreamProcessing #Consistency #Python #Pathway #Kafka #DataEngineering #RealTimeData #StaffPlusNotes

---

## Footnotes

1. **Race Condition**: A situation where the system's substantive behavior is dependent on the sequence or timing of uncontrollable events.
2. **Monotonic Operators**: Operations where the addition of new data cannot invalidate previous results.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.