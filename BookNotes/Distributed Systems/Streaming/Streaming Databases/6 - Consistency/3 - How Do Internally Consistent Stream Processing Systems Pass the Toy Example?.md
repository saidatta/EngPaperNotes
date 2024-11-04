## Overview

In this note, we delve into how internally consistent stream processing systems, such as **RisingWave**, **Materialize**, and **Pathway**, handle the toy bank example effectively. We will explore the mechanisms these systems use to ensure consistency, compare them with eventually consistent systems like Flink SQL and ksqlDB, and discuss potential fixes and trade-offs involved.

---

## Key Concepts

- **Eventual Consistency**: A model where updates to a distributed system may not be immediately visible to all nodes, but all nodes will eventually become consistent.
- **Internal Consistency**: A stronger consistency model where every output is correct for a subset of the inputs, ensuring consistent and correct results at all times.
- **Barriers**: Control records used in stream processing to synchronize data streams, ensuring that operators only combine data from the same version or epoch.
- **Differential Dataflow (DD)**: A programming model and system for performing incremental computations efficiently, used by systems like Materialize and Pathway.

---

## The Toy Bank Example Recap

- **Scenario**: Simulate a banking system with 10 accounts where each account continuously transfers \$1 to another random account.
- **Objective**: Ensure that the **sum of all account balances** remains **zero** at all times (invariant).
- **Challenge**: In eventually consistent systems, intermediate results can violate this invariant due to lack of synchronization.

---

## Mathematical Foundation

### Account Balance Calculation

Let:

- \( A_i \): Balance of account \( i \).
- \( C_i \): Total credits to account \( i \).
- \( D_i \): Total debits from account \( i \).

Balance for account \( i \):

\[
A_i = C_i - D_i
\]

Total balance across all accounts:

\[
\sum_{i=0}^{9} A_i = \sum_{i=0}^{9} (C_i - D_i) = 0
\]

This holds because every debit is matched by a corresponding credit.

---

## Understanding Internal Consistency Mechanisms

### Synchronization of Streams

- **Goal**: Combine streams in a way that ensures only related data is processed together.
- **Approach**: Use mechanisms like **barriers** or **versions** to synchronize inputs before processing them with non-monotonic operators (e.g., `JOIN`, `UNION`).

### Non-Monotonic Operators

- **Definition**: Operators where the inclusion of new data can invalidate previous results.
- **Example**: `JOIN` operations without proper synchronization can produce incorrect results due to race conditions.

---

## How RisingWave Ensures Consistency

### Barriers in RisingWave

- **Barriers**: Control records with **epochs** (timestamps) injected into all data sources periodically.
- **Function**:
  - Serve as version numbers for data.
  - Ensure operators only emit results for a specific version after receiving the same version from all inputs.
- **Impact**: Synchronizes data streams, preventing the processing of unmatched or out-of-sync data.

### Visualization

![Using Transaction Barriers to Ensure Consistency](risingwave_barriers.png)

*Figure: Using transaction barriers to ensure consistency in RisingWave.*

- **Explanation**:
  - Barriers are injected after each transaction.
  - Operators process data up to the current barrier, ensuring that only data with the same version is combined.
  - Prevents race conditions and maintains the invariant.

### Trade-offs

- **Barrier Frequency**:
  - **High Frequency**: Lower end-to-end latency but higher memory consumption.
  - **Low Frequency**: Reduced memory usage but increased latency.

---

## How Materialize Ensures Consistency

### Differential Dataflow (DD)

- **Core Principle**: Data is always versioned.
- **Operator Behavior**:
  - Operators respect versions and only combine data with the same version.
- **Implementation**:
  - No need for additional barriers; versions are inherent in the dataflow.
- **Snapshot Isolation**:
  - Ensures that operations see a consistent snapshot of the data at a given version.

### Visualization

![Differential Dataflow Versioning](materialize_dd.png)

*Figure: How Differential Dataflow in Materialize ensures consistency through data versions.*

---

## How Pathway Ensures Consistency

- **Based on Differential Dataflow**:
  - Similar to Materialize, uses versioning to synchronize inputs.
- **Python Implementation**:
  - Allows stream processing with strong consistency guarantees in Python.
- **Applicability**:
  - Demonstrates that consistency can be achieved without a full-fledged streaming database.

---

## Addressing Eventual Consistency in Systems Like Flink SQL

### The Challenge

- **Issue**: Non-monotonic operators (e.g., `JOIN`) without synchronization can lead to incorrect results due to race conditions.
- **Symptoms**:
  - Inconsistent intermediate results.
  - Violation of invariants (e.g., total balance not zero).

### Potential Fixes

#### Synchronizing Inputs Using Timestamps

- **Approach**:
  - Use a common field (e.g., timestamp) to synchronize data streams.
  - Modify the `JOIN` condition to include the timestamp.

#### Example: Flink SQL Fix

**Original JOIN in Flink SQL**:

```sql
CREATE VIEW balance AS
SELECT
  credits.account,
  credits.credits - debits.debits AS balance
FROM
  credits
JOIN
  debits
ON
  credits.account = debits.account;
```

**Modified JOIN with Timestamp Synchronization**:

```sql
CREATE VIEW balance AS
SELECT
  credits.account,
  credits.credits - debits.debits AS balance
FROM
  credits
JOIN
  debits
ON
  credits.account = debits.account
AND
  credits.ts = debits.ts;
```

- **Explanation**:
  - By including `credits.ts = debits.ts`, we ensure that only data with the same timestamp is joined.
  - This effectively synchronizes the inputs.

#### Limitations of the Fix

- **State Growth**: Including timestamps in `GROUP BY` can lead to unbounded state growth.
- **Brittleness**: Relies on exact timestamp matches, which may not be feasible in real-world scenarios.
- **Complexity**: Requires manual adjustments and may not generalize well to other use cases.
- **Maintenance**: Makes the SQL code more complex and harder to maintain.

---

## MiniBatch in Flink 1.19+

### Overview

- **MiniBatch**: A feature introduced in Flink 1.19 to buffer input records for optimization.
- **Configuration Parameters**:
  - `table.exec.mini-batch.enabled`: Enable MiniBatch.
  - `table.exec.mini-batch.allow-latency`: Maximum latency for buffering.
  - `table.exec.mini-batch.size`: Maximum number of records to buffer.

### Impact on Consistency

- **Findings**:
  - Increasing `mini-batch.size` reduces the number of incorrect intermediate results.
  - At higher batch sizes, Flink outputs only the correct final result.
- **Caveats**:
  - MiniBatch is primarily a performance optimization, not a consistency guarantee.
  - Requires careful tuning and does not provide internal consistency in all cases.

---

## Consistency Versus Latency

### Types of Latency

1. **Processing Time Latency**: Time taken to produce any result.
2. **End-to-End Latency**: Time taken to produce a consistent and correct result.

### Trade-offs

- **Internally Consistent Systems**:
  - May have higher processing time latency due to synchronization overhead.
  - Often achieve lower end-to-end latency for consistent results.
- **Eventually Consistent Systems**:
  - Lower processing time latency.
  - May never reach a consistent state without additional fixes.

### Implications

- **Use Case Considerations**:
  - For applications requiring consistent results (e.g., financial transactions), internal consistency is crucial.
  - For applications prioritizing ultra-low latency over consistency, eventual consistency may suffice.

---

## Recommendations for Stream Processing Systems

- **Provide Consistency Options**:
  - Allow users to toggle between internal and eventual consistency based on use case.
- **Improve Abstractions**:
  - Offer higher-level abstractions that handle synchronization transparently.
- **Educate Users**:
  - Highlight the importance of consistency and guide users on achieving it within the system.

---

## Summary

- **Eventual Consistency Limitations**:
  - Can lead to incorrect results in non-windowed, non-monotonic use cases.
  - Difficult for users from the database world to adapt to.

- **Internal Consistency Benefits**:
  - Ensures correctness without requiring complex workarounds.
  - Aligns with the expectations of database engineers.

- **Key Mechanisms**:
  - **Barriers** in RisingWave.
  - **Versioning** in Materialize and Pathway.

- **Potential Solutions for Eventually Consistent Systems**:
  - Synchronize inputs via timestamps (with limitations).
  - Use features like MiniBatch (does not fully guarantee consistency).

---

## Additional Notes

### Out-of-Order Messages

- Internally consistent systems handle out-of-order data effectively.
- Eventually consistent systems may continue to produce inconsistent results even with out-of-order data.

### Future Directions

- Stream processing systems should aim to provide flexibility in consistency models.
- Enhancing user experience by reducing the complexity of achieving consistency.

---

## Mathematical Appendix

### Race Condition Scenarios

#### Scenario 1: Credits Emit Faster

- **Events**:
  - Credits are emitted before corresponding debits.
- **Result**:
  - Balances show inflated credits, leading to positive total balances.

#### Scenario 2: Debits Emit Faster

- **Events**:
  - Debits are emitted before corresponding credits.
- **Result**:
  - Balances show inflated debits, leading to negative total balances.

### Synchronization via Barriers

- **Operators Process Data**:
  - Only when all inputs have reached the same barrier (version).
- **Ensures**:
  - Data from different stages are aligned temporally.

---

## Code Examples

### Flink SQL with Timestamp Synchronization

```sql
-- Credits View with Timestamp
CREATE VIEW credits AS
SELECT
  to_account AS account,
  SUM(amount) AS credits,
  ts
FROM
  transactions
GROUP BY
  to_account, ts;

-- Debits View with Timestamp
CREATE VIEW debits AS
SELECT
  from_account AS account,
  SUM(amount) AS debits,
  ts
FROM
  transactions
GROUP BY
  from_account, ts;

-- Balance View with Synchronized JOIN
CREATE VIEW balance AS
SELECT
  credits.account,
  credits.credits - debits.debits AS balance
FROM
  credits
JOIN
  debits
ON
  credits.account = debits.account
AND
  credits.ts = debits.ts;
```

---

## Best Practices

1. **Understand Consistency Needs**:
   - Determine if your application requires internal consistency or can tolerate eventual consistency.

2. **Use Built-in Synchronization Mechanisms**:
   - Leverage features like barriers or versioning if available.

3. **Avoid Manual Synchronization Hacks**:
   - Manual fixes may be brittle and hard to maintain.

4. **Test with Real-world Scenarios**:
   - Include out-of-order and delayed messages in testing.

5. **Consider Trade-offs**:
   - Balance between latency and consistency based on application requirements.

---

## References

1. **Jamie Brandon**. *Internal Consistency in Streaming Systems*. [Blog Post](https://scattered-thoughts.net/writing/internal-consistency/), 2021.

2. **Derek G. Murray et al.** *Naiad: A Timely Dataflow System*. Proceedings of SOSP’13.

---

## Tags

#StreamProcessing #Consistency #EventualConsistency #InternalConsistency #Barriers #DifferentialDataflow #FlinkSQL #RisingWave #Materialize #Pathway #DataEngineering #StaffPlusNotes

---

## Footnotes

1. **Eventual Consistency Systems**: Systems like Flink SQL and ksqlDB that prioritize low latency and may produce inconsistent intermediate results.

2. **Internal Consistency Systems**: Systems like RisingWave and Materialize that ensure consistent outputs by synchronizing data streams.

3. **Non-Monotonic Operators**: Operators where the addition of new data can invalidate previous results (e.g., `JOIN`, `UNION`).

4. **Barriers**: Control records used to synchronize streams by indicating versions or epochs.

5. **Differential Dataflow**: A computational model for incremental computations, providing versioned data and synchronization.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.