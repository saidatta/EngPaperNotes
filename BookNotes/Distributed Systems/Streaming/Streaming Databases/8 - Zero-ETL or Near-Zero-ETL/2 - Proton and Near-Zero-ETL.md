# Bridging Operational and Analytical Data Planes

---

## Overview

In this note, we explore **Proton**, a next-generation **RTOLAP** (Real-Time Online Analytical Processing) database that serves as a streaming OLAP solution. We will delve into how Proton integrates with systems like **PeerDB** and **Kafka** to facilitate near-zero-ETL architectures. Additionally, we'll discuss the use of embedded OLAP databases like **DuckDB** and **chDB** to bring analytical workloads closer to the operational plane, thereby reducing data gravity and enhancing performance.

---

## Table of Contents

1. [Introduction to Proton](#introduction-to-proton)
2. [Integrating Proton with PeerDB and Kafka](#integrating-proton-with-peerdb-and-kafka)
   - [Creating a Kafka PEER in PeerDB](#creating-a-kafka-peer-in-peerdb)
   - [Mirroring Data to Kafka](#mirroring-data-to-kafka)
   - [Creating a Stream in Proton](#creating-a-stream-in-proton)
3. [Embedded OLAP Databases](#embedded-olap-databases)
   - [Why Embedded OLAP?](#why-embedded-olap)
   - [Data Flow with Proton and Embedded OLAP](#data-flow-with-proton-and-embedded-olap)
4. [Using DuckDB for Local Analytical Workloads](#using-duckdb-for-local-analytical-workloads)
   - [Installing DuckDB](#installing-duckdb)
   - [Microservice Example with DuckDB and FastAPI](#microservice-example-with-duckdb-and-fastapi)
   - [Implementing UPSERT in DuckDB](#implementing-upsert-in-duckdb)
5. [Using chDB (ClickHouse) for Local Analytical Workloads](#using-chdb-clickhouse-for-local-analytical-workloads)
   - [UPSERT in ClickHouse Using ReplacingMergeTree](#upsert-in-clickhouse-using-replacingmergetree)
   - [Microservice Example with chDB and Flask](#microservice-example-with-chdb-and-flask)
6. [Data Gravity and Replication](#data-gravity-and-replication)
   - [Understanding Data Gravity](#understanding-data-gravity)
   - [Replication Strategies](#replication-strategies)
7. [Analytical Data Reduction](#analytical-data-reduction)
   - [Reducing Data with Materialized Views](#reducing-data-with-materialized-views)
   - [Streaming Materialized View Changes](#streaming-materialized-view-changes)
8. [Conclusion](#conclusion)
9. [References](#references)

---

## Introduction to Proton

**Proton** is a next-generation **RTOLAP database** that fits into the category of **streaming OLAP databases**. It provides:

- **Stateful Streaming Ingestion**: Ability to ingest and process data streams with stateful transformations.
- **APIs for Real-Time Analytics**:
  - **Asynchronous Change Streams**: Push-based updates for real-time data propagation.
  - **Synchronous Pull Queries**: Traditional query model for on-demand data retrieval.
- **Complex Transformations**: Implemented at ingestion time to build **materialized views**.

---

## Integrating Proton with PeerDB and Kafka

To facilitate near-zero-ETL, we can integrate Proton with **PeerDB** and **Kafka**.

### Creating a Kafka PEER in PeerDB

First, we create a **PEER** in PeerDB to connect to Kafka.

**Example 8-3: Setting up a PEER to Kafka in PeerDB**

```sql
CREATE PEER kafka_peer FROM KAFKA WITH (
  bootstrap_server = '<bootstrap-servers>'
);
```

- **`kafka_peer`**: Name of the PEER.
- **`<bootstrap-servers>`**: Kafka bootstrap servers (e.g., `localhost:9092`).

### Mirroring Data to Kafka

Next, we create a **MIRROR** in PeerDB to replicate data from the source database to Kafka.

```sql
CREATE MIRROR mirror_to_kafka FROM
  source_peer TO kafka_peer FOR
$$
  SELECT * FROM source_table WHERE
  timestamp BETWEEN {{.start}} AND {{.end}}
$$
WITH (
  destination_table_name = 'my_topic'
);
```

- **`mirror_to_kafka`**: Name of the mirror.
- **`source_peer`**: The source database peer.
- **`kafka_peer`**: The Kafka peer we created.
- **`source_table`**: Table to mirror.
- **`timestamp`**: Column used for watermarking.
- **`'my_topic'`**: Kafka topic to write data to.

### Creating a Stream in Proton

In Proton, we create a stream to read data from Kafka.

**Example 8-4: Creating a stream from Kafka to Proton**

```sql
CREATE EXTERNAL STREAM frontend_events(raw string)
SETTINGS type='kafka',
         brokers='<bootstrap-servers>',
         topic='my_topic';
```

- **`frontend_events`**: Name of the external stream in Proton.
- **`raw string`**: Data type of the incoming messages.
- **`type='kafka'`**: Specifies the source as Kafka.
- **`brokers='<bootstrap-servers>'`**: Kafka brokers.
- **`topic='my_topic'`**: Topic to read from.

---

## Embedded OLAP Databases

### Why Embedded OLAP?

- **Trend**: Bringing smaller analytical workloads closer to the operational plane.
- **Limitations of HTAP Databases**:
  - Limited capacity to store large amounts of historical data.
  - Not optimized for extensive analytical queries compared to systems like Snowflake or ClickHouse.
- **Solution**: Reduce analytical data to fit the business domain and operational capacity using embedded OLAP databases.

### Data Flow with Proton and Embedded OLAP

![Streaming OLAP database reducing analytical data to be served in the operational plane](figure_8-4.png)

*Figure 8-4: Streaming OLAP database reducing analytical data to be served in the operational plane.*

- **PeerDB**: Sends real-time operational data to Kafka.
- **Proton**: Ingests and transforms data, builds materialized views.
- **Materialized View Changes**: Written to a Kafka topic.
- **Application**: Consumes changes and builds a local replica using an embedded OLAP database like **DuckDB** or **chDB**.

---

## Using DuckDB for Local Analytical Workloads

### Installing DuckDB

**DuckDB** is an embedded OLAP database suitable for analytical workloads within applications.

**Example 8-5: Installing DuckDB**

```bash
pip install duckdb
```

### Microservice Example with DuckDB and FastAPI

We can create a microservice that:

- Subscribes to a Kafka topic.
- Performs UPSERTs into a DuckDB table.
- Exposes analytical queries via a REST API.

**Example 8-6: Microservice that reads from Kafka and writes to DuckDB**

```python
import duckdb
from threading import Thread
from fastapi import FastAPI
from confluent_kafka import Consumer

app = FastAPI()
duckdb_con = duckdb.connect('my_persistent_db.duckdb')  # 1

def upsert(msg):  # 2
    # Deserialize the message to extract values
    primary_key, col1_value, col2_value = deserialize_message(msg)
    duckdb_con.execute("""
        INSERT OR REPLACE INTO my_table(id, col1, col2) VALUES(?, ?, ?)
    """, (primary_key, col1_value, col2_value))

def kafka2olap(conf):  # 3
    consumer = Consumer(conf)
    try:
        consumer.subscribe(["my_data"])
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            else:
                upsert(msg)
    finally:
        consumer.close()

@app.on_event("startup")
async def initialize():  # 4
    conf = {
        'bootstrap.servers': '<bootstrap-servers>',
        'group.id': 'my_group',
        'auto.offset.reset': 'earliest'
    }
    thread = Thread(target=kafka2olap, args=(conf,))
    thread.start()

@app.get("/my_data/")
async def read_item(id: int):  # 5
    results = duckdb_con.execute("""
        SELECT
            id,
            COUNT(*) AS row_counter,
            CURRENT_TIMESTAMP
        FROM my_table
        WHERE id = ?
    """, (id,)).fetchall()
    return results
```

**Explanation**:

1. **Connect to DuckDB**: Establish a connection to a persistent DuckDB database file.
2. **Define UPSERT Function**: Handles inserting or updating records in the DuckDB table.
3. **Kafka Consumer Function**: Consumes messages from Kafka and calls the `upsert` function.
4. **Initialize Kafka Consumer**: Starts the Kafka consumer thread when the application starts.
5. **REST API Endpoint**: Exposes an endpoint to query data from DuckDB.

### Implementing UPSERT in DuckDB

DuckDB supports `INSERT OR REPLACE` for UPSERT operations.

**Example 8-7: Implementing UPSERT in DuckDB**

```python
def upsert(msg):
    # Deserialize the message to get column values
    primary_key, col1_value, col2_value = deserialize_message(msg)
    duckdb_con.execute("""
        INSERT OR REPLACE INTO my_table(id, col1, col2) VALUES(?, ?, ?)
    """, (primary_key, col1_value, col2_value))
```

- **`INSERT OR REPLACE`**: Inserts a new record or replaces an existing one based on the primary key.

---

## Using chDB (ClickHouse) for Local Analytical Workloads

### UPSERT in ClickHouse Using ReplacingMergeTree

**chDB** is an embeddable OLAP database based on **ClickHouse**.

**Example 8-8: ClickHouse ENGINE that supports UPSERT**

```sql
CREATE TABLE hackernews_rmt (
    id UInt32,
    author String,
    comment String,
    views UInt64
)
ENGINE = ReplacingMergeTree()  -- 1
PRIMARY KEY (author, id);

SELECT *
FROM hackernews_rmt
FINAL;  -- 2
```

**Explanation**:

1. **`ReplacingMergeTree` Engine**: Used to emulate UPSERT behavior by replacing duplicate records during merges.
2. **`FINAL` Keyword**: Ensures the query returns the latest version of each record.

### Microservice Example with chDB and Flask

We can create a microservice using **Flask** and **chDB**.

**Example 8-9: chDB microservice wrapper**

```python
from flask import Flask, request
import chdb
import os

app = Flask(__name__)

@app.route('/', methods=["GET"])
def clickhouse():
    query = request.args.get('query', default="", type=str)
    format = request.args.get('default_format', default="JSONCompact", type=str)
    if not query:
        return "Query not found", 400
    res = chdb.query(query, format)
    return res.bytes()

@app.route('/', methods=["POST"])
def play():
    query = request.data.decode('utf-8')
    format = request.args.get('default_format', default="JSONCompact", type=str)
    if not query:
        return "Query not found", 400
    res = chdb.query(query, format)
    return res.bytes()

@app.errorhandler(404)
def handle_404(e):
    return "Not found", 404

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8123))
    app.run(host=host, port=port)
```

- **Flask Application**: Exposes endpoints to execute queries against the embedded chDB.
- **GET and POST Support**: Allows for both GET and POST requests to run queries.

---

## Data Gravity and Replication

### Understanding Data Gravity

- **Data Gravity**: Concept that data, like mass, attracts other data and services.
- **Implications**:
  - Large datasets are difficult to move or replicate.
  - Centralized analytical systems force operational systems to send data to a single location, increasing latency.

### Replication Strategies

- **Distribute Real-Time Analytics**: By streaming materialized view changes to operational systems, we can replicate data across regions.
- **Embedded OLAP Databases**: Use local replicas to serve analytical queries close to the user, reducing latency.

---

## Analytical Data Reduction

### Reducing Data with Materialized Views

- **Materialized Views**: Precomputed views that aggregate or transform data.
- **Purpose**: Reduce data volume and complexity.

### Streaming Materialized View Changes

- **Proton's Capability**: Streams changes to materialized views into Kafka topics.
- **Consumption**:
  - Applications consume these changes and update local embedded OLAP databases.
  - Ensures that operational systems have up-to-date analytical data without handling full datasets.

**Mathematical Representation**:

Let:

- **Total Data Volume (\( D_t \))**: The total size of analytical data.
- **Reduced Data Volume (\( D_r \))**: The size after applying materialized views.
- **Reduction Ratio (\( R \))**:

  \[
  R = \frac{D_r}{D_t}
  \]

Our goal is to minimize \( D_r \) while retaining necessary analytical insights, thus maximizing data reduction.

---

## Conclusion

By integrating Proton with PeerDB and leveraging embedded OLAP databases like DuckDB and chDB, we can:

- Achieve **near-zero-ETL** architectures.
- Reduce data gravity challenges by distributing analytical data efficiently.
- Bring analytical workloads closer to the operational plane, improving performance and reducing latency.
- Use materialized views and streaming to minimize data movement and maintain up-to-date analytics.

---

## References

- **Proton**: [Timeplus Official Website](https://www.timeplus.io/)
- **PeerDB**: [PeerDB GitHub Repository](https://github.com/peerdb-io/peerdb)
- **DuckDB**: [DuckDB Official Website](https://duckdb.org/)
- **chDB**: [chDB GitHub Repository](https://github.com/chdb-io/chdb)
- **ClickHouse ReplacingMergeTree**: [ClickHouse Documentation](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/replacingmergetree)

---

## Tags

#Proton #RTOLAP #StreamingOLAP #PeerDB #DuckDB #chDB #EmbeddedOLAP #DataGravity #MaterializedViews #NearZeroETL #DataEngineering #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.