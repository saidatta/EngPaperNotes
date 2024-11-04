## Overview

**Incremental View Maintenance (IVM)** is a method to keep **materialized views** in a relational database up-to-date incrementally. Instead of recomputing the entire view from scratch whenever the base tables change, IVM computes and applies only the changes (deltas) to the views. This approach is more efficient, especially when changes to the base tables are small compared to their overall size.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Timing of View Maintenance](#timing-of-view-maintenance)
   - [Immediate Maintenance](#immediate-maintenance)
   - [Deferred Maintenance](#deferred-maintenance)
3. [Implementations of IVM](#implementations-of-ivm)
   - [pg_ivm](#pg_ivm)
     - [Example with pg_ivm](#example-with-pg_ivm)
     - [Pros and Cons](#pg_ivm-pros-and-cons)
   - [Hydra](#hydra)
     - [Example with Hydra](#example-with-hydra)
     - [Pros and Cons](#hydra-pros-and-cons)
   - [Epsio](#epsio)
     - [Example with Epsio](#example-with-epsio)
     - [Pros and Cons](#epsio-pros-and-cons)
   - [Feldera](#feldera)
     - [Example with Feldera](#example-with-feldera)
     - [Pros and Cons](#feldera-pros-and-cons)
   - [PeerDB](#peerdb)
     - [Replication Modes](#replication-modes)
     - [Examples with PeerDB](#examples-with-peerdb)
     - [Pros and Cons](#peerdb-pros-and-cons)
4. [Mathematical Representation](#mathematical-representation)
   - [Delta Computation](#delta-computation)
   - [Efficiency Gains](#efficiency-gains)
5. [Summary](#summary)
6. [References](#references)
7. [Tags](#tags)

---

## Introduction

**Materialized Views** are precomputed views that store query results physically, improving query performance by avoiding expensive computations on large datasets. However, maintaining materialized views can be resource-intensive, especially when underlying data changes frequently.

**Incremental View Maintenance (IVM)** addresses this challenge by updating only the parts of the materialized view affected by changes in the base tables. This incremental approach is more efficient and can significantly reduce the overhead of view maintenance.

---

## Timing of View Maintenance

There are two primary approaches to view maintenance concerning timing:

### Immediate Maintenance

- **Definition**: Views are updated **immediately** in the same transaction that modifies the base tables.
- **Characteristics**:
  - Ensures that the materialized view is always consistent with the base tables.
  - May introduce overhead during write operations, as updates to views occur synchronously.
- **Use Cases**:
  - Scenarios where data consistency is critical.
  - Applications requiring real-time analytics or up-to-date reporting.

### Deferred Maintenance

- **Definition**: Views are updated **after** the transaction that modifies the base tables has been committed.
- **Methods**:
  - **Explicit Commands**: Using commands like `REFRESH MATERIALIZED VIEW`.
  - **Scheduled Updates**: Periodic background processes.
  - **Lazy Updates**: Views are updated when they are accessed.
- **Characteristics**:
  - Reduces overhead during write operations.
  - May result in views being out-of-sync with base tables until maintenance occurs.
- **Use Cases**:
  - Scenarios where slight delays in data consistency are acceptable.
  - Batch processing applications.

---

## Implementations of IVM

Several tools and extensions implement IVM, bridging the operational and streaming planes:

### pg_ivm

#### Introduction

- **pg_ivm** is a PostgreSQL extension that adds immediate maintenance capabilities for materialized views.
- **Method**: Uses **AFTER triggers** to update materialized views when base tables are modified.

#### Example with pg_ivm

**Step 1: Create an Immediately Maintained Materialized View**

```sql
SELECT create_immv('myview', 'SELECT * FROM mytab');
```

- **Explanation**:
  - `create_immv`: Function provided by pg_ivm to create an immediately maintained materialized view.
  - `'myview'`: Name of the materialized view.
  - `'SELECT * FROM mytab'`: Query defining the view.

**Step 2: Insert Data into Base Table**

```sql
INSERT INTO mytab (id, name) VALUES (1, 'Alice');
```

**Step 3: Query the Materialized View**

```sql
SELECT * FROM myview;
```

- **Result**:

| id | name  |
|----|-------|
| 1  | Alice |

- The materialized view `myview` is updated immediately after the insertion into `mytab`.

#### Pros and Cons

##### Pros

- **Integration**: Seamlessly integrates with PostgreSQL.
- **Consistency**: Ensures views are always up-to-date.
- **Simplicity**: No need for external systems.

##### Cons

- **Resource Sharing**: Shares compute and memory resources with the database.
- **Performance Impact**: Immediate maintenance may impact write performance.
- **Row-Based Storage**: May not be optimal for analytical workloads requiring columnar storage.

---

### Hydra

#### Introduction

- **Hydra** is a database that extends PostgreSQL with both row and columnar storage.
- **Features**:
  - Supports traditional and incremental materialized views.
  - Leverages pg_ivm for immediate view maintenance.
  - Allows tables to use either **heap** (row-based) or **columnar** storage.

#### Example with Hydra

**Step 1: Create Tables with Different Storage Types**

```sql
CREATE TABLE heap_table (
  id SERIAL PRIMARY KEY,
  name TEXT
) USING heap;

CREATE TABLE columnar_table (
  id SERIAL PRIMARY KEY,
  data JSONB
) USING columnar;
```

- **Explanation**:
  - `USING heap`: Creates a row-based table.
  - `USING columnar`: Creates a columnar table.

**Step 2: Create an Incrementally Maintained Materialized View**

```sql
SELECT create_immv('columnar_view', 'SELECT * FROM columnar_table');
```

#### Pros and Cons

##### Pros

- **Flexibility**: Supports both row-based and columnar tables.
- **Performance**: Columnar storage optimizes analytical queries.
- **Integration**: Leverages PostgreSQL ecosystem.

##### Cons

- **Complexity**: Managing different storage types may add complexity.
- **Resource Sharing**: Similar resource concerns as pg_ivm.

---

### Epsio

#### Introduction

- **Epsio** is a tool for incremental materialized view maintenance that works with existing PostgreSQL databases.
- **Method**: Constantly and incrementally updates query results as underlying data changes.
- **Features**:
  - Supports complex SQL syntax (JOINs, CTEs, subqueries, GROUP BY).
  - Offloads view maintenance from the source database.

#### Example with Epsio

**Step 1: Define a Materialized View**

```sql
CALL epsio.create_view('epsio_view',
  'SELECT SUM(salary) AS total_salary, d.name AS department_name
   FROM employee_salaries e
   JOIN departments d ON e.department_id = d.id
   GROUP BY d.name');
```

- **Explanation**:
  - `epsio.create_view`: Epsio function to create a materialized view.
  - `'epsio_view'`: Name of the view.
  - The query calculates the total salary per department.

**Step 2: Query the Materialized View**

```sql
SELECT * FROM epsio_view;
```

- **Result**:

| total_salary | department_name |
|--------------|-----------------|
| 250000       | Engineering     |
| 150000       | Marketing       |

#### Pros and Cons

##### Pros

- **Performance**: Offloads processing from the source database.
- **Efficiency**: Incremental updates avoid full recomputation.
- **Complex Queries**: Supports a wide range of SQL features.

##### Cons

- **Additional System**: Requires setting up and maintaining Epsio separately.
- **Data Sync**: Ensuring data consistency between source and Epsio.

---

### Feldera

#### Introduction

- **Feldera** is a continuous analytics platform based on the **Database Stream Processor (DBSP)** engine.
- **Method**: Processes queries continuously and incrementally, sending only changes to outputs.
- **Features**:
  - Real-time ETL capabilities.
  - Consistency guarantees similar to Materialize.
  - Supports complex, nested SQL queries.

#### Example with Feldera

**Step 1: Declare Input Tables**

```sql
CREATE TABLE VENDOR (
    id BIGINT NOT NULL PRIMARY KEY,
    name VARCHAR,
    address VARCHAR
);

CREATE TABLE PART (
    id BIGINT NOT NULL PRIMARY KEY,
    name VARCHAR
);

CREATE TABLE PRICE (
    part BIGINT NOT NULL,
    vendor BIGINT NOT NULL,
    price DECIMAL
);
```

**Step 2: Write Continuous Queries**

```sql
-- Lowest available price for each part
CREATE VIEW LOW_PRICE AS
SELECT part, MIN(price) AS price
FROM PRICE
GROUP BY part;

-- Preferred vendor details
CREATE VIEW PREFERRED_VENDOR AS
SELECT
    PART.id AS part_id,
    PART.name AS part_name,
    VENDOR.id AS vendor_id,
    VENDOR.name AS vendor_name,
    PRICE.price
FROM
    PRICE
    JOIN PART ON PART.id = PRICE.part
    JOIN VENDOR ON VENDOR.id = PRICE.vendor
    JOIN LOW_PRICE ON PRICE.price = LOW_PRICE.price AND PRICE.part = LOW_PRICE.part;
```

- **Explanation**:
  - **`LOW_PRICE` View**: Calculates the lowest price per part.
  - **`PREFERRED_VENDOR` View**: Retrieves details of parts and vendors offering the lowest price.

**Step 3: Define Data Sources and Outputs**

- Data sources (e.g., Kafka topics, databases) and outputs are configured in the Feldera UI or configuration files.

#### Pros and Cons

##### Pros

- **Real-Time Processing**: Processes data continuously.
- **Consistency**: Strong consistency guarantees.
- **Complex Queries**: Supports nested and complex SQL queries.

##### Cons

- **No Materialized Views**: Does not support materialized views directly.
- **Additional System**: Requires maintaining Feldera separately.

---

### PeerDB

#### Introduction

- **PeerDB** is an open-source tool for streaming data from PostgreSQL to various destinations (data warehouses, queues, storage).
- **Method**: Synchronizes data in real-time, acting as a CDC (Change Data Capture) tool.
- **Features**:
  - High performance (claims to be 10× faster than similar tools).
  - Supports complex transformations via streaming queries.

#### Replication Modes

1. **Log-Based**: Uses PostgreSQL's write-ahead log (WAL) for changes.
2. **Cursor-Based**: Based on timestamp or integer columns.
3. **XMIN-Based**: Uses transaction IDs.
4. **Streaming Query**: Allows complex transformations before replication.

#### Examples with PeerDB

**Example 1: Create Source and Target Peers**

```sql
CREATE PEER source FROM POSTGRES WITH (
  host = 'source_host',
  port = '5432',
  user = 'postgres',
  password = 'password',
  database = 'source_db'
);

CREATE PEER target FROM POSTGRES WITH (
  host = 'target_host',
  port = '5432',
  user = 'postgres',
  password = 'password',
  database = 'target_db'
);
```

**Example 2: Set Up Log-Based CDC**

```sql
CREATE MIRROR cdc_mirror FROM source TO target
WITH TABLE MAPPING (public.test_table:public.test_table);
```

- **Explanation**:
  - **`CREATE MIRROR`**: Sets up replication between source and target.
  - **`TABLE MAPPING`**: Specifies tables to replicate.

**Example 3: Set Up a Streaming Query**

```sql
CREATE MIRROR qrep_mirror FROM source TO target
FOR $$
    SELECT id, hashint4(c1) AS hash_c1, hashint4(c2) AS hash_c2, md5(t) AS hash_t
    FROM test_table WHERE id BETWEEN {{.start}} AND {{.end}}
$$ WITH (
    watermark_table_name = 'public.test_table',
    watermark_column = 'id',
    num_rows_per_partition = 10000,
    destination_table_name = 'public.test_transformed',
    mode = 'append'
);
```

- **Explanation**:
  - **Transformation**: Hashes sensitive columns before replication.
  - **Watermarking**: Controls data partitioning for streaming.

#### Pros and Cons

##### Pros

- **High Performance**: Efficient data replication.
- **Flexible Transformations**: Supports complex streaming queries.
- **Multiple Destinations**: Supports various data warehouses and storage systems.

##### Cons

- **No Direct Materialized Views**: Does not maintain materialized views; relies on target systems.
- **Setup Complexity**: Requires configuration of source, target, and replication modes.

---

## Mathematical Representation

### Delta Computation

Let:

- **\( V \)**: Materialized view.
- **\( T \)**: Base table.
- **\( \Delta T \)**: Change in the base table (insertions, deletions, updates).
- **\( \Delta V \)**: Change in the materialized view.

**IVM Principle**:

\[
V_{\text{new}} = V_{\text{old}} \oplus \Delta V
\]

Where:

- **\( \Delta V = f(\Delta T) \)**: Function **\( f \)** computes the changes to the view based on changes in the base table.
- **\( \oplus \)**: Represents the incremental update operation (e.g., union, aggregation).

### Efficiency Gains

- **Traditional Recalculation Cost**:

  \[
  C_{\text{full}} = O(n)
  \]

  Where **\( n \)** is the size of the base table.

- **IVM Update Cost**:

  \[
  C_{\text{ivm}} = O(|\Delta T|)
  \]

  Where **\( |\Delta T| \)** is the size of the change, typically **\( |\Delta T| \ll n \)**.

- **Efficiency Gain**:

  \[
  \text{Gain} = \frac{C_{\text{full}}}{C_{\text{ivm}}} = \frac{O(n)}{O(|\Delta T|)}
  \]

- **Interpretation**:

  - When changes are small, **IVM** significantly reduces computation time compared to full recomputation.

---

## Summary

- **Incremental View Maintenance (IVM)** improves efficiency by updating only the affected parts of materialized views.
- **Immediate Maintenance** ensures views are always consistent but may impact write performance.
- **Deferred Maintenance** reduces immediate overhead but may result in stale views.
- Tools like **pg_ivm**, **Hydra**, **Epsio**, **Feldera**, and **PeerDB** implement IVM or related functionalities.
- **Mathematical models** show that IVM reduces computational complexity when changes to base tables are small.
- Choosing the right tool depends on factors like performance requirements, resource constraints, and the need for data consistency.

---

## References

1. **pg_ivm**: [https://github.com/sraoss/pg_ivm](https://github.com/sraoss/pg_ivm)
2. **Hydra**: [https://www.hydrastorage.io](https://www.hydrastorage.io)
3. **Epsio**: [https://epsio.io](https://epsio.io)
4. **Feldera**: [https://feldera.com](https://feldera.com)
5. **PeerDB**: [https://peerdb.io](https://peerdb.io)
6. **Materialized Views in SQL**: [https://en.wikipedia.org/wiki/Materialized_view](https://en.wikipedia.org/wiki/Materialized_view)
7. **Incremental View Maintenance**: [https://en.wikipedia.org/wiki/View_maintenance](https://en.wikipedia.org/wiki/View_maintenance)

---

## Tags

#IncrementalViewMaintenance #IVM #MaterializedViews #PostgreSQL #pg_ivm #Hydra #Epsio #Feldera #PeerDB #DataEngineering #StreamingData #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.