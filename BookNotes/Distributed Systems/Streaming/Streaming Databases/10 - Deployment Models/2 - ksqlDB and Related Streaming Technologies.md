## Overview

In this note, we will explore various streaming technologies, focusing on **ksqlDB** and its role in stream processing. We'll also delve into alternative solutions like **Incremental View Maintenance (IVM)**, **Postgres Multicorn Foreign Data Wrapper**, code-based stream processors, lakehouse/streamhouse technologies, caching technologies, and general guidelines on where to process and query data.

---

## Table of Contents

1. [ksqlDB](#ksqldb)
   - [Introduction](#introduction)
   - [Use Cases](#use-cases)
   - [Pros and Cons](#pros-and-cons)
2. [Incremental View Maintenance (IVM)](#incremental-view-maintenance-ivm)
   - [Introduction](#ivm-introduction)
   - [Pros and Cons](#ivm-pros-and-cons)
3. [Postgres Multicorn Foreign Data Wrapper](#postgres-multicorn-foreign-data-wrapper)
   - [Introduction](#multicorn-introduction)
   - [Pros and Cons](#multicorn-pros-and-cons)
4. [Code-Based Stream Processors](#code-based-stream-processors)
   - [Introduction](#code-stream-introduction)
   - [Examples](#code-stream-examples)
5. [Lakehouse/Streamhouse Technologies](#lakehouse-streamhouse-technologies)
   - [Introduction](#lakehouse-introduction)
   - [Examples](#lakehouse-examples)
6. [Caching Technologies](#caching-technologies)
   - [Introduction](#caching-introduction)
   - [Examples](#caching-examples)
7. [General Guidelines: Where to Process and Query Data](#general-guidelines)
   - [The Four "Where" Questions](#the-four-where-questions)
   - [Analytical Use Case Example](#analytical-use-case-example)
   - [Consequences and Considerations](#consequences-and-considerations)
8. [Summary](#summary)
9. [References](#references)
10. [Tags](#tags)

---

## ksqlDB

### Introduction

**ksqlDB** is a streaming SQL engine for Apache Kafka. It allows for real-time data processing and stream transformations using SQL-like syntax. It's built on top of **Kafka Streams**, a Java library for building stream processing applications.

- **Consistency Guarantees**: Provides "continuous refinement," similar to **eventual consistency**.
- **Deployment**: Designed for deployment inside microservices, typically in the **operational plane** as part of an application backend.

### Use Cases

- **Simple Stream Processing Operations**: Best suited for straightforward transformations and filtering.
- **Preparing Data for Analytical Destinations**: Transforms data for ingestion into data warehouses or data lakes.
- **Point Queries Using Materialized Views**: Supports batch-only destinations that cannot run a full-fledged Kafka consumer.

### Limitations

- **Complexity in JOINs**: Difficult to implement correct JOIN operations, especially between streams and tables.
- **Limited SQL Support**: Only supports a subset of SQL syntax and semantics.
  - **No Self-JOINs**
  - **No Nested JOINs**
- **High Risk of Inconsistent Logic**: Requires a team of stream processing experts to mitigate risks.

### Pros and Cons

#### Pros

- **Data Freshness**: Real-time processing ensures up-to-date data.
- **Stream Processing Capabilities**: Leverages Kafka Streams for powerful transformations.
- **Materialized Views**: Enables point queries using TABLEs to support batch-only destinations.

#### Cons

- **Limited to Kafka**: Only supports Kafka as a data source and sink.
- **High Expertise Required**: Complex operations require deep stream processing knowledge.
- **Limited SQL Support**: Does not support full SQL syntax and semantics.
- **Complex Stream Processing**: Hard to implement complex operations correctly.

### Example: Simple Stream Transformation with ksqlDB

**Scenario**: Filter and transform a Kafka topic containing user events.

#### Step 1: Define the Source Stream

```sql
CREATE STREAM user_events (
  user_id VARCHAR,
  event_type VARCHAR,
  event_timestamp BIGINT
) WITH (
  KAFKA_TOPIC='user_events',
  VALUE_FORMAT='JSON'
);
```

#### Step 2: Create a Derived Stream

```sql
CREATE STREAM login_events AS
SELECT
  user_id,
  event_timestamp
FROM user_events
WHERE event_type = 'login';
```

#### Step 3: Materialize the View

```sql
CREATE TABLE user_login_counts AS
SELECT
  user_id,
  COUNT(*) AS login_count
FROM login_events
WINDOW TUMBLING (SIZE 1 HOUR)
GROUP BY user_id;
```

---

## Incremental View Maintenance (IVM)

### Introduction

**Incremental View Maintenance (IVM)** solutions like **Feldera**, **PeerDB**, and **Epsio** support batch-based point queries on preprocessed, fresh data.

- **Integration with Operational Databases**: Closely integrated with databases like PostgreSQL.
- **No Need for Kafka**: Does not require Kafka as an intermediary layer.
- **Complex Preprocessing**: Allows for complex data transformations using full SQL semantics.

### Pros and Cons

#### Pros

- **Data Freshness**: Provides up-to-date data.
- **Full SQL Support**: Supports full SQL syntax and semantics.
- **Consistency**: Ensures data consistency.
- **Enables Point Queries**: Suitable for batch-based querying.
- **Ease of Adoption**: Introduces streaming concepts within familiar database environments.

#### Cons

- **Limited Flexibility**: Restricted to supported data sources and sinks.
- **Vendor Lock-In**: Dependent on specific vendor capabilities.

### Example: Using PeerDB for Incremental View Maintenance

**Scenario**: Synchronize data from PostgreSQL to Snowflake with incremental updates.

#### Step 1: Configure PeerDB

```yaml
source:
  type: postgres
  config:
    host: localhost
    port: 5432
    database: mydb
    user: myuser
    password: mypassword

destination:
  type: snowflake
  config:
    account: myaccount
    user: myuser
    password: mypassword
    database: MY_DATABASE
    schema: PUBLIC
```

#### Step 2: Define the Incremental Sync

```sql
CREATE INCREMENTAL SYNC my_sync
FROM postgres.my_table
TO snowflake.my_table
WITH
  PRIMARY KEY (id),
  COLUMN MAPPING (id, name, updated_at);
```

---

## Postgres Multicorn Foreign Data Wrapper

### Introduction

**Multicorn** is a PostgreSQL extension that simplifies the development of Foreign Data Wrappers (FDWs) by allowing the use of Python.

- **Purpose**: Enables building FDWs for non-Postgres databases.
- **Use Case**: Access data from RTOLAP databases like Apache Pinot within PostgreSQL.
- **Location**: Operates within the operational plane.

### Pros and Cons

#### Pros

- **Full SQL Support**: Leverages PostgreSQL's SQL capabilities.
- **Consistency**: Ensures consistent data access.
- **Integration**: Access both OLTP (Postgres) and OLAP databases within PostgreSQL.

#### Cons

- **Limited to PostgreSQL**: Requires Postgres as the central database technology.
- **Impractical at Large Scale**: Difficult to standardize on Postgres in large organizations with diverse databases.

### Example: Creating a Multicorn FDW for Apache Pinot

#### Step 1: Install Multicorn

```shell
CREATE EXTENSION multicorn;
```

#### Step 2: Create a Foreign Data Wrapper

```sql
CREATE SERVER pinot_server
FOREIGN DATA WRAPPER multicorn
OPTIONS (
  wrapper 'pinot_fdw.PinotFDW',
  host 'pinot-host',
  port '8099'
);
```

#### Step 3: Create a Foreign Table

```sql
CREATE FOREIGN TABLE pinot_table (
  id BIGINT,
  name TEXT,
  value DOUBLE PRECISION
)
SERVER pinot_server
OPTIONS (
  table 'pinot_real_time_table'
);
```

---

## Code-Based Stream Processors

### Introduction

Code-based stream processors are frameworks that allow developers to write custom stream processing logic using programming languages like Python, Java, or C#.

- **Use Cases**: Ideal for complex or "hardcore" streaming use cases like fraud detection.
- **Location**: Typically located on the operational plane within microservices architectures.

### Examples

1. **Kafka Streams**: Java library for building stream processing applications.
2. **Apache Flink**: Open-source framework for stateful computations over unbounded and bounded data streams.
3. **Quix Streams**: Supports Python and C# for real-time data streaming.
4. **Bytewax**: Python library for building dataflows.
5. **Pathway**: Python-based, leveraging Timely Dataflow (also used in Materialize).

### Pros and Cons

#### Pros

- **Flexibility**: Allows for highly customized stream processing logic.
- **Language Support**: Supports multiple programming languages.
- **Integration**: Can integrate deeply with application logic.

#### Cons

- **Complexity**: Requires significant expertise in stream processing.
- **Maintenance**: Custom code can be harder to maintain and scale.
- **Consistency**: Ensuring data consistency can be challenging.

### Example: Stream Processing with Apache Flink

#### Word Count Example

```java
public class StreamingJob {

    public static void main(String[] args) throws Exception {

        // Set up execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Define data source
        DataStream<String> text = env.socketTextStream("localhost", 9999);

        // Define transformation
        DataStream<Tuple2<String, Integer>> wordCounts = text
            .flatMap(new FlatMapFunction<String, Tuple2<String, Integer>>() {
                public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
                    for (String word : value.split("\\s")) {
                        out.collect(new Tuple2<>(word, 1));
                    }
                }
            })
            .keyBy(0)
            .sum(1);

        // Define sink
        wordCounts.print();

        // Execute program
        env.execute("Streaming Word Count");
    }
}
```

---

## Lakehouse/Streamhouse Technologies

### Introduction

Lakehouse technologies are evolving to incorporate streaming capabilities, blurring the lines between batch and stream processing.

- **Lakehouse Examples**:
  - **Databricks Delta Tables**
  - **Apache Iceberg**
  - **Apache Hudi**
- **Streamhouse Architecture**: Proposed by Ververica with **Apache Paimon**, integrating stream processing closely with data lakes.

### Examples

1. **Databricks Spark Structured Streaming**: Streaming data processing with Apache Spark.
2. **Confluent's Tableflow**: Exposes data on Kafka-based streaming platforms as Iceberg tables.
3. **Streambased.io**: Allows querying Kafka topics with SQL, leveraging Trino and enhanced indexing.

### Pros and Cons

#### Pros

- **Unified Architecture**: Seamlessly integrates batch and stream processing.
- **Efficiency**: Reduces data duplication and processing latency.
- **Accessibility**: Makes streaming data more accessible through familiar SQL interfaces.

#### Cons

- **Complexity**: Can add complexity to the data architecture.
- **Vendor Lock-In**: May rely on proprietary solutions or specific platforms.
- **Performance Overhead**: Additional layers may introduce performance trade-offs.

### Example: Streaming Data with Apache Iceberg and Flink

#### Step 1: Define the Iceberg Table

```sql
CREATE TABLE iceberg_db.events (
  event_id BIGINT,
  event_type STRING,
  event_time TIMESTAMP(3)
) WITH (
  'connector' = 'iceberg',
  'catalog-name' = 'my_catalog',
  'catalog-type' = 'hadoop',
  'warehouse' = 'hdfs://namenode:8020/warehouse_path'
);
```

#### Step 2: Configure Flink Streaming Job

```sql
CREATE TABLE kafka_events (
  event_id BIGINT,
  event_type STRING,
  event_time TIMESTAMP(3),
  WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
  'connector' = 'kafka',
  'topic' = 'events_topic',
  'properties.bootstrap.servers' = 'kafka:9092',
  'format' = 'json'
);

INSERT INTO iceberg_db.events
SELECT * FROM kafka_events;
```

---

## Caching Technologies

### Introduction

Caching technologies like **Redis** and **Hazelcast** provide ultra-low latency data access and can be used for streaming data scenarios.

- **Use Cases**: Suitable for scenarios requiring sub-millisecond latency.
- **Location**: Can be deployed on the operational or streaming plane, depending on integration.

### Examples

1. **Redis**: In-memory data structure store, used as a database, cache, and message broker.
2. **Hazelcast**: In-memory data grid that offers stream processing capabilities with SQL-like syntax.

### Pros and Cons

#### Pros

- **Low Latency**: Provides extremely fast data access.
- **Stream Processing Capabilities**: Some caching solutions offer stream processing features.
- **Scalability**: Can scale horizontally to handle large volumes of data.

#### Cons

- **Data Persistence**: Primarily in-memory; may require strategies for data persistence.
- **Complexity**: Managing state and consistency can be challenging.
- **Limited Analytical Capabilities**: Not designed for complex analytical queries.

### Example: Stream Processing with Hazelcast Jet

#### Step 1: Define the Pipeline

```java
Pipeline pipeline = Pipeline.create();

pipeline.drawFrom(Sources.kafka(kafkaProps, "input-topic"))
    .withoutTimestamps()
    .flatMap(event -> transformEvent(event))
    .drainTo(Sinks.kafka(kafkaProps, "output-topic"));
```

#### Step 2: Submit the Job

```java
HazelcastInstance hazelcastInstance = Hazelcast.bootstrappedInstance();
JetService jet = hazelcastInstance.getJet();

JobConfig jobConfig = new JobConfig();
jobConfig.setName("stream-processing-job");

jet.newJob(pipeline, jobConfig);
```

---

## General Guidelines: Where to Process and Query Data

### The Four "Where" Questions

When designing data architectures, consider the following questions:

1. **Where is my use case located?**
   - Operational Plane, Analytical Plane, or Streaming Plane?
2. **Where is the data I need for my use case?**
   - Origin of the data.
3. **Where do I process it?**
   - Location where data transformations occur.
4. **Where do I query it?**
   - Location where data is accessed or consumed.

### Analytical Use Case Example

**Scenario**: An analytical dashboard requires data originally from operational systems.

- **Use Case Location**: Analytical Plane.
- **Data Origin**: Streaming Platform (Streaming Plane).

#### Possible Paths

1. **Process and Query on Analytical Plane**:

   ![Processing and Querying on Analytical Plane](processing_on_analytical_plane.png)

   - Pros:
     - Centralized processing.
     - Mature tooling and infrastructure.
   - Cons:
     - Higher latency due to data movement.
     - Inefficient incremental processing.

2. **Process on Streaming Plane, Query on Analytical Plane**:

   ![Processing on Streaming Plane, Querying on Analytical Plane](processing_on_streaming_plane.png)

   - Pros:
     - Lower latency.
     - Incremental processing.
   - Cons:
     - Requires streaming expertise.
     - Potentially higher complexity.

3. **Process and Query on Streaming Plane**:

   ![Processing and Querying on Streaming Plane](processing_and_querying_on_streaming_plane.png)

   - Pros:
     - Lowest latency.
     - Real-time data availability.
   - Cons:
     - Requires advanced streaming infrastructure.
     - Limited historical data retention.

### Consequences and Considerations

- **Data Freshness**: Closer processing to data origin leads to fresher data.
- **Incremental Processing**: Streaming allows for processing only new data.
- **Expertise Required**: Streaming technologies may require specialized skills.
- **Cost Implications**: Streaming can reduce processing costs but may increase infrastructure costs.
- **Data Gravity**: The tendency of data and applications to attract each other; processing closer to data can reduce latency.

### Mathematical Representation

**Latency (\( L \))**:

- **Total Latency**:
  \[
  L_{\text{total}} = L_{\text{ingest}} + L_{\text{process}} + L_{\text{query}}
  \]
- **Reducing \( L_{\text{process}} \) and \( L_{\text{query}} \)** by processing and querying closer to data origin.

---

## Summary

- **ksqlDB** is suitable for simple stream processing but has limitations in complex operations and consistency.
- **IVM Solutions** like PeerDB offer consistent, SQL-based transformations without Kafka.
- **Multicorn FDW** allows PostgreSQL to access data from other databases but is limited to Postgres ecosystems.
- **Code-Based Stream Processors** provide flexibility but require expertise and maintenance.
- **Lakehouse/Streamhouse Technologies** are bridging the gap between batch and streaming but may add complexity.
- **Caching Technologies** offer ultra-low latency but have limitations in analytical capabilities.
- **General Guidelines** emphasize considering the location of processing and querying to optimize for latency, cost, and data freshness.

---

## References

1. **ksqlDB Documentation**: [https://ksqldb.io](https://ksqldb.io)
2. **PeerDB**: [https://peerdb.io](https://peerdb.io)
3. **Apache Flink**: [https://flink.apache.org](https://flink.apache.org)
4. **Multicorn FDW**: [https://multicorn.org](https://multicorn.org)
5. **Hazelcast Jet**: [https://hazelcast.com/products/jet/](https://hazelcast.com/products/jet/)
6. **Streambased.io**: [https://www.streambased.io](https://www.streambased.io)
7. **Ververica Streamhouse**: [https://www.ververica.com](https://www.ververica.com)
8. **Confluent Tableflow**: [https://www.confluent.io/blog/confluent-tableflow](https://www.confluent.io/blog/confluent-tableflow)
9. **Databricks Delta Lake**: [https://delta.io](https://delta.io)

---

## Tags

#ksqlDB #StreamingDatabases #StreamProcessing #OperationalAnalytics #DataEngineering #IncrementalViewMaintenance #ForeignDataWrapper #Lakehouse #StreamingPlane #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.