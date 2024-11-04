## Overview

In modern data processing, ensuring **consistency** across different stream processing systems is crucial, especially when dealing with real-time data and financial transactions. This document compares the consistency guarantees and behaviors of various stream processing systems using a **toy bank example**. The systems compared are:

- **Apache Flink SQL**
- **ksqlDB**
- **Proton (Timeplus)**
- **RisingWave**
- **Materialize**

We will delve into each system's setup, execution, and the results obtained, focusing on how they handle consistency in stream processing.

---

## The Toy Bank Example

### Scenario

- **Bank Accounts**: 10 accounts (numbered 0 to 9).
- **Transactions**: Each account continuously transfers \$1 to another random account.
- **Objective**: Ensure that the **sum of all account balances** remains **zero** at all times (invariant).

### Consistency Check

- **Invariant**: The total sum of all account balances should always be **zero**.
- **Test**: At any point, summing up the balances of all accounts should yield zero.
- **Challenge**: In streaming systems, data can arrive **out of order**, and computations may be **eventually consistent**, leading to temporary inconsistencies.

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

Total balance for all accounts:

\[
\sum_{i=0}^{9} A_i = \sum_{i=0}^{9} (C_i - D_i) = 0
\]

This holds because every debit is matched by a corresponding credit in another account.

---

## Implementing the Toy Example

### Transaction Data Generation (Python Code)

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

random.seed(42)
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
    time.sleep(0.01)  # Simulate 10ms delay between transactions

p.close()
```

- **Explanation**:
  - Generates **10,000 transactions**.
  - Each transaction transfers **\$1** from a random account to another.
  - Transactions are produced every **10 milliseconds**.
  - Uses **Kafka** as the messaging platform.

---

## System Comparisons

### 1. Apache Flink SQL

#### Setup

**Define Source Table (`transactions`):**

```sql
CREATE TABLE transactions (
    id BIGINT,
    from_account INT,
    to_account INT,
    amount DOUBLE,
    ts TIMESTAMP(3)
) WITH (
    'connector' = 'kafka',
    'topic' = 'transactions',
    'properties.bootstrap.servers' = 'localhost:9092',
    'properties.group.id' = 'transactions_flink',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json',
    'json.fail-on-missing-field' = 'true',
    'json.ignore-parse-errors' = 'false'
);
```

**Create Views:**

- **Credits View:**

  ```sql
  CREATE VIEW credits AS
  SELECT
    to_account AS account,
    SUM(amount) AS credits
  FROM transactions
  GROUP BY to_account;
  ```

- **Debits View:**

  ```sql
  CREATE VIEW debits AS
  SELECT
    from_account AS account,
    SUM(amount) AS debits
  FROM transactions
  GROUP BY from_account;
  ```

- **Balance View:**

  ```sql
  CREATE VIEW balance AS
  SELECT
    credits.account AS account,
    credits.credits - debits.debits AS balance
  FROM credits
  JOIN debits ON credits.account = debits.account;
  ```

- **Total View:**

  ```sql
  CREATE VIEW total AS
  SELECT SUM(balance) AS total FROM balance;
  ```

**Define Sink Table (`total_sink`):**

```sql
CREATE TABLE total_sink (
  total DOUBLE,
  PRIMARY KEY (total) NOT ENFORCED
) WITH (
  'connector' = 'upsert-kafka',
  'properties.bootstrap.servers' = 'localhost:9092',
  'topic' = 'total_flink',
  'key.format' = 'json',
  'value.format' = 'json',
  'properties.group.id' = 'total_flink'
);
```

**Insert into Sink:**

```sql
INSERT INTO total_sink SELECT * FROM total;
```

#### Execution and Results

- **Dataflow Graph**:

  - The data flows from `transactions` to `credits` and `debits`.
  - `credits` and `debits` are joined to compute `balance`.
  - The `balance` is summed to compute the `total`.

- **Observations**:

  - **Number of Messages**: ~80,000 messages in `total_flink`.
  - **Behavior**: The `total` fluctuates wildly between positive and negative values (e.g., +400 to -600).
  - **Convergence**: Once the input stream stops, the `total` eventually converges to **0**.

- **Reason**:

  - **Eventual Consistency**: Flink SQL provides eventual consistency, which means intermediate results may be inconsistent.
  - **Latency vs. Consistency Trade-off**: Prioritizes low latency over strong consistency.

#### Visual Representation

- **Graph**: A line graph showing fluctuations in `total` over time, converging to zero at the end.

---

### 2. ksqlDB

#### Setup

**Define Source Table (`transactions`):**

```sql
CREATE TABLE transactions (
  id VARCHAR PRIMARY KEY,
  from_account INT,
  to_account INT,
  amount DOUBLE,
  ts VARCHAR
) WITH (
  kafka_topic = 'transactions',
  value_format = 'json',
  partitions = 1,
  timestamp = 'ts',
  timestamp_format = 'yyyy-MM-dd HH:mm:ss.SSS'
);
```

**Create Tables (Views):**

- **Credits Table:**

  ```sql
  CREATE TABLE credits AS
  SELECT
    to_account AS account,
    SUM(amount) AS credits
  FROM transactions
  GROUP BY to_account EMIT CHANGES;
  ```

- **Debits Table:**

  ```sql
  CREATE TABLE debits AS
  SELECT
    from_account AS account,
    SUM(amount) AS debits
  FROM transactions
  GROUP BY from_account EMIT CHANGES;
  ```

- **Balance Table:**

  ```sql
  CREATE TABLE balance AS
  SELECT
    credits.account AS account,
    credits.credits - debits.debits AS balance
  FROM credits
  INNER JOIN debits ON credits.account = debits.account EMIT CHANGES;
  ```

- **Total Table:**

  ```sql
  CREATE TABLE total AS
  SELECT
    'foo' AS key,
    SUM(balance) AS total
  FROM balance
  GROUP BY 'foo' EMIT CHANGES;
  ```

**Note**: `EMIT CHANGES` indicates continuous query (push query).

#### Execution and Results

- **Sink Topic**: Results are written to Kafka topic `total_ksqldb`.

- **Observations**:

  - **Number of Messages**: ~40,000 messages in `total_ksqldb`.
  - **Behavior**: `total` fluctuates between positive and negative values (e.g., +100 to -100).
  - **Convergence**: Converges to **0** after the input stream stops.

- **Reason**:

  - **Eventual Consistency**: Similar to Flink SQL, ksqlDB operates on eventual consistency.
  - **Intermediate Inconsistencies**: Due to asynchronous processing and lack of synchronization.

#### Visual Representation

- **Graph**: Similar fluctuation pattern as Flink SQL, with convergence to zero at the end.

---

### 3. Proton (Timeplus)

#### Setup

**Define External Stream (`transactions`):**

```sql
CREATE EXTERNAL STREAM transactions(
    id INT,
    from_account INT,
    to_account INT,
    amount INT,
    ts DATETIME64
) SETTINGS
    type = 'kafka',
    brokers = 'broker:29092',
    topic = 'transactions',
    data_format = 'JSONEachRow';
```

**Create Views:**

- **Credits View:**

  ```sql
  CREATE VIEW credits AS
  SELECT
    now64() AS ts,
    to_account AS account,
    SUM(amount) AS credits
  FROM transactions
  GROUP BY to_account
  EMIT PERIODIC 100ms;
  ```

- **Debits View:**

  ```sql
  CREATE VIEW debits AS
  SELECT
    now64() AS ts,
    from_account AS account,
    SUM(amount) AS debits
  FROM transactions
  GROUP BY from_account
  EMIT PERIODIC 100ms;
  ```

- **Balance View:**

  ```sql
  CREATE VIEW balance AS
  SELECT
    c.account,
    credits - debits AS balance
  FROM
    changelog(credits, account, ts, true) AS c
    JOIN changelog(debits, account, ts, true) AS d ON c.account = d.account;
  ```

**Define External Stream for Output (`total_s`):**

```sql
CREATE EXTERNAL STREAM total_s(total INT) SETTINGS
    type = 'kafka',
    brokers = 'broker:29092',
    topic = 'total_proton',
    data_format = 'JSONEachRow';
```

**Create Materialized View (`total`):**

```sql
CREATE MATERIALIZED VIEW total INTO total_s AS
SELECT
    SUM(balance) AS total
FROM
    balance;
```

#### Execution and Results

- **Sink Topic**: Results are written to Kafka topic `total_proton`.

- **Observations**:

  - **Number of Messages**: 56 messages in `total_proton`.
  - **Behavior**: Fluctuations are less severe, ranging between -10 and +9.
  - **Convergence**: Converges to **0** after the input stream stops.

- **Reason**:

  - **Eventual Consistency with Periodic Emission**: The use of `EMIT PERIODIC` reduces the number of intermediate inconsistencies.

#### Visual Representation

- **Graph**: Fewer fluctuations, smoother convergence to zero.

---

### 4. RisingWave

#### Setup

**Define Source Table (`transactions`):**

```sql
CREATE TABLE IF NOT EXISTS transactions (
  id INT,
  from_account INT,
  to_account INT,
  amount INT,
  ts TIMESTAMP
) WITH (
  connector = 'kafka',
  topic = 'transactions',
  properties.bootstrap.server = 'broker:29092',
  scan.startup.mode = 'earliest',
  scan.startup.timestamp_millis = '140000000'
) ROW FORMAT JSON;
```

**Create Views:**

- **Accounts View:**

  ```sql
  CREATE VIEW accounts AS
  SELECT
    from_account AS account
  FROM transactions
  UNION
  SELECT
    to_account
  FROM transactions;
  ```

- **Credits View:**

  ```sql
  CREATE VIEW credits AS
  SELECT
    transactions.to_account AS account,
    SUM(transactions.amount) AS credits
  FROM transactions
  LEFT JOIN accounts ON transactions.to_account = accounts.account
  GROUP BY to_account;
  ```

- **Debits View:**

  ```sql
  CREATE VIEW debits AS
  SELECT
    transactions.from_account AS account,
    SUM(transactions.amount) AS debits
  FROM transactions
  LEFT JOIN accounts ON transactions.from_account = accounts.account
  GROUP BY from_account;
  ```

- **Balance View:**

  ```sql
  CREATE VIEW balance AS
  SELECT
    credits.account AS account,
    credits.credits - debits.debits AS balance
  FROM credits
  INNER JOIN debits ON credits.account = debits.account;
  ```

- **Total View:**

  ```sql
  CREATE VIEW total AS
  SELECT SUM(balance) AS total FROM balance;
  ```

**Define Sink (`total_sink`):**

```sql
CREATE SINK total_sink
FROM total
WITH (
  connector = 'kafka',
  properties.bootstrap.server = 'broker:29092',
  topic = 'total_risingwave',
  type = 'append-only',
  force_append_only = 'true'
);
```

#### Execution and Results

- **Sink Topic**: Results are written to Kafka topic `total_risingwave`.

- **Observations**:

  - **Number of Messages**: 105 messages in `total_risingwave`.
  - **Behavior**: All messages have the correct total of **0**.
  - **Consistency**: Maintains internal consistency throughout.

- **Reason**:

  - **Internal Consistency**: RisingWave provides stronger consistency guarantees, ensuring correct results at all times.

#### Visual Representation

- **Graph**: A flat line at zero, indicating consistent correct results.

---

### 5. Materialize

#### Setup

**Define Kafka Connection:**

```sql
CREATE CONNECTION kafka_connection TO KAFKA (BROKER 'broker:29092');
```

**Define Source (`transactions_source`):**

```sql
CREATE SOURCE transactions_source
FROM KAFKA CONNECTION kafka_connection (TOPIC 'transactions', START OFFSET (0))
KEY FORMAT TEXT
VALUE FORMAT TEXT
INCLUDE KEY
ENVELOPE UPSERT;
```

**Create View (`transactions`):**

```sql
CREATE VIEW transactions AS
SELECT
  ((text::jsonb) ->> 'id')::STRING AS id,
  ((text::jsonb) ->> 'from_account')::INT AS from_account,
  ((text::jsonb) ->> 'to_account')::INT AS to_account,
  ((text::jsonb) ->> 'amount')::INT AS amount,
  ((text::jsonb) ->> 'ts')::TIMESTAMP AS ts,
  key
FROM transactions_source;
```

**Create Views:**

- **Accounts View**: Same as RisingWave.

- **Credits View**: Same as RisingWave.

- **Debits View**: Same as RisingWave.

- **Balance View**: Same as RisingWave.

- **Total View**: Same as RisingWave.

**Define Sink (`total_sink`):**

```sql
CREATE SINK total_sink
FROM total
INTO KAFKA CONNECTION kafka_connection (TOPIC 'total_materialize')
FORMAT JSON
ENVELOPE DEBEZIUM;
```

#### Execution and Results

- **Sink Topic**: Results are written to Kafka topic `total_materialize`.

- **Observations**:

  - **Number of Messages**: Only **one message**.
  - **Behavior**: The message contains the correct total of **0**.
  - **Consistency**: Materialize ensures consistent results at all times.

- **Reason**:

  - **Strong Consistency Model**: Materialize provides internal consistency, with results always reflecting the correct state.

#### Visual Representation

- **Graph**: A single point at zero, indicating immediate correct result.

---

## Analysis and Conclusions

### Consistency Models

- **Eventual Consistency**: Flink SQL, ksqlDB, and Proton operate under eventual consistency.
  - **Implications**:
    - Intermediate results may be inconsistent.
    - Systems eventually converge to the correct result after the input stream stops.
    - Suitable for windowed aggregations and applications tolerant of temporary inconsistencies.

- **Internal Consistency**: RisingWave and Materialize provide internal consistency.
  - **Implications**:
    - Results are always consistent and correct.
    - No need to wait for the input stream to stop.
    - Better suited for applications requiring strong consistency (e.g., financial transactions).

### Trade-offs

- **Latency vs. Consistency**:
  - Systems favoring low latency may accept weaker consistency.
  - Systems providing strong consistency may incur higher latency or resource usage.

- **Throughput**:
  - Systems with strong consistency may have lower throughput due to synchronization overhead.

### Recommendations

- **Use Cases Requiring Strong Consistency**:
  - Choose systems like RisingWave or Materialize.
  - Critical for financial applications, inventory management, etc.

- **Use Cases Tolerant to Temporary Inconsistencies**:
  - Systems like Flink SQL, ksqlDB, or Proton may suffice.
  - Suitable for real-time analytics where eventual accuracy is acceptable.

---

## Mathematical Explanation of Inconsistencies

### Eventual Consistency Systems

- **Out-of-Order Processing**:
  - Transactions may be processed in an order different from their occurrence.
  - Leads to temporary imbalances in credits and debits.

- **Asynchronous Aggregations**:
  - Credits and debits are aggregated separately and joined later.
  - Lack of synchronization causes mismatches in intermediate results.

### Internal Consistency Systems

- **Synchronous Processing**:
  - Ensure that credits and debits are processed in a coordinated manner.
  - Joins and aggregations reflect all inputs up to a certain point consistently.

- **Incremental Updates**:
  - Maintain materialized views that are updated with each new transaction.
  - Use techniques like **delta processing** to efficiently update results.

---

## Best Practices

1. **Understand Consistency Requirements**:
   - Determine if your application can tolerate temporary inconsistencies.
   - Choose a system that aligns with your consistency needs.

2. **Leverage Materialized Views**:
   - Use materialized views to maintain up-to-date results efficiently.

3. **Monitor Latency and Throughput**:
   - Be aware of the trade-offs between consistency, latency, and throughput.
   - Optimize configurations based on your application's priorities.

4. **Test with Realistic Workloads**:
   - Simulate real-world scenarios to observe system behavior under load.

5. **Stay Updated on System Capabilities**:
   - Streaming systems are evolving; newer versions may offer improved consistency guarantees.

---

## Conclusion

- **Consistency in Stream Processing**:
  - Essential for applications where accuracy is critical.
  - Different systems offer varying levels of consistency guarantees.

- **System Selection**:
  - No one-size-fits-all solution.
  - Evaluate based on your specific use case, consistency needs, and performance requirements.

- **Future Directions**:
  - Exploration of hybrid models that balance consistency and performance.
  - Potential enhancements in classical stream processors to offer stronger consistency without significant performance penalties.

---

## References

- **Apache Flink Documentation**: [Flink SQL](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/dev/table/sql/overview/)
- **ksqlDB Documentation**: [ksqlDB](https://ksqldb.io/)
- **Proton (Timeplus) Documentation**: [Proton GitHub](https://github.com/timeplus-io/chameleon)
- **RisingWave Documentation**: [RisingWave](https://www.risingwave.dev/)
- **Materialize Documentation**: [Materialize](https://materialize.com/docs/)

---

## Tags

#Consistency #StreamProcessing #StreamingDatabases #ApacheFlink #ksqlDB #Timeplus #RisingWave #Materialize #DataEngineering #StaffPlusNotes

---

## Additional Notes

- **Dataflow Graphs**: Visual representations help in understanding the flow of data and transformations applied.

- **Kafka Topics**: Used extensively for both input (source) and output (sink) in these examples.

- **SQL Variations**: Each system has its own dialect or extensions of SQL; be mindful of syntax differences.

- **Emphasis on Correctness**: In financial applications, correctness trumps performance; choose systems accordingly.

---

## Footnotes

1. **Eventual Consistency**: A consistency model used in distributed computing to achieve high availability that allows for updates to be propagated to all nodes eventually.

2. **Internal Consistency**: A stronger consistency model where the system ensures that operations appear to execute atomically and in isolation.

3. **Materialized Views**: Database objects that contain the results of a query and can be updated incrementally.

---

## Appendices

### A. Sample Transaction Data

**First Few Transactions:**

```json
{
    "id": 0,
    "from_account": 1,
    "to_account": 3,
    "amount": 1,
    "ts": "2023-10-25 12:00:00.000"
}
{
    "id": 1,
    "from_account": 0,
    "to_account": 7,
    "amount": 1,
    "ts": "2023-10-25 12:00:00.010"
}
{
    "id": 2,
    "from_account": 4,
    "to_account": 2,
    "amount": 1,
    "ts": "2023-10-25 12:00:00.020"
}
```

### B. Kafka Configuration Tips

- Ensure that the Kafka cluster is properly configured with sufficient partitions and replication factors.
- Monitor Kafka topics to identify any bottlenecks or issues in data ingestion and consumption.

### C. Stream Processing Configuration

- **Flink SQL**:
  - Configure `state.backend` and `checkpointing` for fault tolerance.
  - Adjust `parallelism` based on workload.

- **ksqlDB**:
  - Use `EMIT CHANGES` wisely; understand its implications on resources.

- **Proton**:
  - Utilize `EMIT PERIODIC` to control the frequency of updates.

- **RisingWave and Materialize**:
  - Leverage their PostgreSQL compatibility for integration with existing tools.

---

Feel free to reach out if you have any questions or need further assistance with any of these topics.