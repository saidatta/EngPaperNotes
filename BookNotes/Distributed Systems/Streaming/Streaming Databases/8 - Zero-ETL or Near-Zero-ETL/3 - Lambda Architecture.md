## Overview

The **Lambda Architecture** is a data processing architecture designed to combine both **batch** and **real-time (streaming)** data processing. Introduced by **Nathan Marz** in his 2011 book *Big Data: Principles and Best Practices of Scalable Realtime Data Systems*, it addresses the challenges of providing robust and scalable data processing for big data applications. The term "lambda" is inspired by the Greek letter λ, representing the dual processing paths for batch and real-time data.

---
## Table of Contents

1. [Introduction](#introduction)
2. [Key Components](#key-components)
   - [Batch Layer](#batch-layer)
   - [Speed Layer](#speed-layer)
   - [Serving Layer](#serving-layer)
3. [Data Flow in Lambda Architecture](#data-flow-in-lambda-architecture)
4. [Advantages and Challenges](#advantages-and-challenges)
5. [Mathematical Modeling](#mathematical-modeling)
6. [Implementation Example](#implementation-example)
   - [Apache Pinot Hybrid Tables](#apache-pinot-hybrid-tables)
     - [REALTIME Table Definition](#realtime-table-definition)
     - [OFFLINE Table Definition](#offline-table-definition)
7. [Pipeline Configurations](#pipeline-configurations)
8. [Summary](#summary)
9. [References](#references)
10. [Tags](#tags)

---

## Introduction

- **Purpose**: To handle both **batch processing** (for historical data) and **real-time processing** (for immediate data needs) in a unified architecture.
- **Use Cases**: Ideal for applications requiring immediate insights as well as historical analytics, such as fraud detection, recommendation systems, and real-time analytics dashboards.

---

## Key Components

### 1. Batch Layer

- **Function**: Processes large volumes of data in a batch-oriented manner.
- **Responsibilities**:
  - Stores the **master dataset** (immutable, append-only raw data).
  - Performs batch processing to compute **batch views**.
- **Technologies**:
  - **Apache Hadoop**
  - **Apache Spark**

### 2. Speed Layer

- **Function**: Processes data in real-time to provide low-latency updates.
- **Responsibilities**:
  - Handles new data as it arrives.
  - Compensates for the high latency of the batch layer.
- **Technologies**:
  - **Apache Storm**
  - **Apache Flink**
  - **Apache Kafka Streams**

### 3. Serving Layer

- **Function**: Merges and exposes the results from both batch and speed layers to provide a comprehensive view.
- **Responsibilities**:
  - Indexes batch views for efficient querying.
  - Combines real-time and batch views for up-to-date results.
- **Technologies**:
  - **Apache Pinot**
  - **Apache Druid**
  - **Apache Cassandra**

---

## Data Flow in Lambda Architecture

![Lambda Architecture Diagram](lambda_architecture_diagram.png)

*Figure: Overview of the Lambda Architecture.*

- **Data Ingestion**: Data from various sources is ingested into both the batch and speed layers simultaneously.
- **Batch Processing**:
  - Processes data in large batches.
  - Generates comprehensive batch views.
- **Real-Time Processing**:
  - Processes data as it arrives.
  - Generates real-time views.
- **Serving Layer**:
  - Merges batch and real-time views.
  - Serves queries to applications and users.

---

## Advantages and Challenges

### Advantages

- **Scalability**: Can handle large volumes of data by scaling batch and speed layers independently.
- **Fault Tolerance**: Batch layer maintains the master dataset, ensuring data integrity.
- **Flexibility**: Supports both real-time and historical data processing.

### Challenges

- **Complexity**: Maintaining two separate codebases for batch and speed layers.
- **Consistency**: Ensuring that batch and real-time views are synchronized.
- **Operational Overhead**: Increased maintenance and monitoring efforts.

---

## Mathematical Modeling

Let's model the Lambda Architecture using mathematical expressions.

### Batch Layer

- **Master Dataset (\( D \))**: The immutable, append-only dataset.
- **Batch Function (\( f \))**: Computes batch views from \( D \).
- **Batch View (\( V_b \))**:

  \[
  V_b = f(D)
  \]

### Speed Layer

- **Incoming Data (\( d_t \))**: Data arriving at time \( t \).
- **Real-Time Function (\( g \))**: Processes \( d_t \) to produce real-time view.
- **Real-Time View (\( V_s \))**:

  \[
  V_s = g(d_t)
  \]

### Serving Layer

- **Unified View (\( V \))**: Combines batch and real-time views.

  \[
  V = V_b \cup V_s
  \]

- **Query Function (\( q \))**: Executes queries on \( V \).

  \[
  q(V) = \text{Result}
  \]

---

## Implementation Example

Let's explore an implementation using **Apache Pinot** as the serving layer and **Apache Flink** as the stream processor in the speed layer.

### Using a Separate Stream Processor and OLAP Database

- **Batch Layer**: Processes historical data and stores it in an OLAP database like Apache Pinot.
- **Speed Layer**: Uses Apache Flink to process streaming data, perform transformations, and write to Kafka.
- **Serving Layer**: Apache Pinot consumes data from both the batch layer (offline data) and speed layer (real-time data) to serve queries.

![Lambda Architecture with Apache Pinot](lambda_architecture_pinot.png)

*Figure 8-5: A more complex real-time data pipeline that can serve ad hoc queries on both streaming and historical data.*

---

## Apache Pinot Hybrid Tables

Apache Pinot supports **Hybrid Tables**, which consist of two internal tables:

- **REALTIME Table**: Ingests streaming data from sources like Kafka.
- **OFFLINE Table**: Stores batch-processed historical data.

These tables share the same name, allowing Pinot to provide a unified view.

### REALTIME Table Definition

**Example 8-10: Pinot REALTIME Table**

```json
{
  "tableName": "airlineStats",
  "tableType": "REALTIME",
  "segmentsConfig": {
    "timeColumnName": "DaysSinceEpoch",
    "retentionTimeUnit": "DAYS",
    "retentionTimeValue": "5",
    "replication": "1"
  },
  "ingestionConfig": {
    "streamIngestionConfig": {
      "streamConfigMaps": [
        {
          "streamType": "kafka",
          "stream.kafka.topic.name": "flights-realtime",
          "stream.kafka.decoder.class.name": "org.apache.pinot.plugin.stream.kafka.KafkaJSONMessageDecoder",
          "stream.kafka.consumer.factory.class.name": "org.apache.pinot.plugin.stream.kafka20.KafkaConsumerFactory",
          "stream.kafka.consumer.prop.auto.offset.reset": "smallest",
          "stream.kafka.zk.broker.url": "localhost:2191/kafka",
          "stream.kafka.broker.list": "localhost:19092",
          "realtime.segment.flush.threshold.time": "3600000",
          "realtime.segment.flush.threshold.size": "50000"
        }
      ]
    },
    "transformConfigs": [
      {
        "columnName": "ts",
        "transformFunction": "fromEpochDays(DaysSinceEpoch)"
      },
      {
        "columnName": "tsRaw",
        "transformFunction": "fromEpochDays(DaysSinceEpoch)"
      }
    ]
  },
  "fieldConfigList": [
    {
      "name": "ts",
      "encodingType": "DICTIONARY",
      "indexTypes": ["TIMESTAMP"],
      "timestampConfig": {
        "granularities": ["DAY", "WEEK", "MONTH"]
      }
    }
  ]
}
```

**Explanation**:

1. **Ingestion Configuration**: Specifies how to ingest data from Kafka.
2. **Transformations**: Applied during ingestion to process timestamps.
3. **Field Configurations**: Defines indexing and encoding for efficient querying.

### OFFLINE Table Definition

**Example 8-11: Pinot OFFLINE Table**

```json
{
  "tableName": "airlineStats",
  "tableType": "OFFLINE",
  "segmentsConfig": {
    "timeColumnName": "DaysSinceEpoch",
    "timeType": "DAYS",
    "segmentPushType": "APPEND",
    "segmentAssignmentStrategy": "BalanceNumSegmentAssignmentStrategy",
    "replication": "1"
  },
  "fieldConfigList": [
    {
      "name": "ts",
      "encodingType": "DICTIONARY",
      "indexTypes": ["TIMESTAMP"],
      "timestampConfig": {
        "granularities": ["DAY", "WEEK", "MONTH"]
      }
    },
    {
      "name": "ArrTimeBlk",
      "encodingType": "DICTIONARY",
      "indexes": {
        "inverted": {
          "enabled": "true"
        }
      },
      "tierOverwrites": {
        "hotTier": {
          "encodingType": "DICTIONARY",
          "indexes": {
            "bloom": {
              "enabled": "true"
            }
          }
        },
        "coldTier": {
          "encodingType": "RAW",
          "indexes": {
            "text": {
              "enabled": "true"
            }
          }
        }
      }
    }
  ],
  "tableIndexConfig": {
    "starTreeIndexConfigs": [
      {
        "dimensionsSplitOrder": ["AirlineID", "Origin", "Dest"],
        "functionColumnPairs": ["COUNT__*", "MAX__ArrDelay"],
        "maxLeafRecords": 10
      }
    ],
    "enableDynamicStarTreeCreation": true,
    "loadMode": "MMAP"
  },
  "tierConfigs": [
    {
      "name": "hotTier",
      "segmentSelectorType": "time",
      "segmentAge": "3130d",
      "storageType": "pinot_server",
      "serverTag": "DefaultTenant_OFFLINE"
    },
    {
      "name": "coldTier",
      "segmentSelectorType": "time",
      "segmentAge": "3140d",
      "storageType": "pinot_server",
      "serverTag": "DefaultTenant_OFFLINE"
    }
  ]
}
```

**Explanation**:

1. **Star-Tree Index**: Preaggregates data to improve query performance.
2. **Tiered Storage**: Configures hot and cold tiers for efficient storage management.
3. **Field Configurations**: Specifies indexing strategies for different fields.

### Tiered Storage in Pinot

![Pinot Tiered Storage](pinot_tiered_storage.png)

*Figure 8-6: Pinot tiered storage.*

- **Hot Tier**: Stores recent data for fast access.
- **Cold Tier**: Stores older data to optimize storage costs.

---

## Pipeline Configurations

### Data Ingestion from OLTP to Kafka

- **Use Case**: Capturing changes from a Postgres database using **Debezium**.

**Example 8-12: Debezium Postgres Configuration**

```json
{
  "name": "postgres",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "tasks.max": "1",
    "database.hostname": "0.0.0.0",
    "database.port": "5432",
    "database.user": "postgres",
    "database.password": "postgres",
    "database.dbname": "postgres",
    "topic.prefix": "dbserver1",
    "schema.include.list": "inventory"
  }
}
```

- **Explanation**: This configuration captures change data and writes it to Kafka topics.

### Data Transformation with Flink SQL

**Example 8-13: Creating a Kafka Source Connector in Flink**

```sql
CREATE TABLE KafkaSource (
  `id` BIGINT,
  `col1` STRING,
  `col2` STRING,
  `ts` TIMESTAMP(3) METADATA FROM 'timestamp'
) WITH (
  'connector' = 'kafka',
  'topic' = 'my_data',
  'properties.bootstrap.servers' = 'localhost:9092',
  'properties.group.id' = 'abc',
  'scan.startup.mode' = 'earliest-offset',
  'format' = 'json'
);
```

**Example 8-14: Creating a Kafka Sink in Flink SQL**

```sql
CREATE TABLE KafkaSink (
  `user_id` BIGINT,
  `col1` STRING,
  `col2` STRING,
  `ts` TIMESTAMP(3) METADATA FROM 'timestamp'
) WITH (
  'connector' = 'kafka',
  'topic' = 'my_data_transformed',
  'properties.bootstrap.servers' = 'localhost:9092',
  'properties.group.id' = 'testGroup',
  'scan.startup.mode' = 'earliest-offset',
  'format' = 'json'
);
```

- **Transformation Query**:

  ```sql
  INSERT INTO KafkaSink
  SELECT id, col1, UPPER(col2) AS col2_transformed, ts
  FROM KafkaSource
  WHERE col1 IS NOT NULL;
  ```

- **Explanation**: Transforms data from the source topic and writes it to a new topic for consumption by Pinot.

---

## Summary

- The **Lambda Architecture** effectively handles both batch and real-time data processing, providing a comprehensive solution for big data analytics.
- Using technologies like **Apache Pinot**, **Flink**, and **Debezium**, we can implement a scalable and efficient data pipeline that supports ad hoc queries on both streaming and historical data.
- While this architecture offers flexibility and robustness, it also introduces complexity and operational overhead.
- **Use Cases**:
  - Suitable for internal data analysis requiring full historical context.
  - Not ideal for external user-facing applications due to potential resource constraints.

---

## References

1. Nathan Marz, *Big Data: Principles and Best Practices of Scalable Realtime Data Systems*, Manning Publications, 2011.
2. Apache Pinot Documentation: [Hybrid Tables](https://docs.pinot.apache.org/basics/data-import/real-time)
3. Apache Flink Documentation: [Flink SQL](https://nightlies.apache.org/flink/flink-docs-release-1.13/docs/dev/table/sql/)
4. Debezium Documentation: [Debezium Postgres Connector](https://debezium.io/documentation/reference/connectors/postgresql.html)

---

## Tags

#LambdaArchitecture #BigData #RealTimeProcessing #BatchProcessing #ApachePinot #ApacheFlink #Debezium #DataEngineering #StaffPlusNotes #DataPipeline

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.