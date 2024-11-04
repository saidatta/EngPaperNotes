## Overview

In this chapter, we explore several **deployment models** for real-time analytics, focusing on how to leverage streaming databases and other streaming solutions effectively. We will examine where streaming databases are most advantageous and when alternative approaches might be more suitable. Key considerations include:

- **Consistency**: How data consistency impacts deployment choices.
- **Workload Types**: The nature of workloads (analytical vs. operational).
- **Storage Formats**: Row-based vs. columnar storage.
- **Streaming Plane**: Utilizing the streaming plane introduced in Chapter 9.

Our goal is to understand various architectural patterns that facilitate real-time analytics within the streaming plane and how they align with different use cases.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Consistency Spectrum in Real-Time Analytics](#consistency-spectrum-in-real-time-analytics)
3. [Deployment Models](#deployment-models)
   - [Consistent Streaming Database](#consistent-streaming-database)
   - [Consistent Stream Processor and RTOLAP](#consistent-stream-processor-and-rtolap)
   - [Eventually Consistent OLAP Streaming Database](#eventually-consistent-olap-streaming-database)
   - [Eventually Consistent Stream Processor and RTOLAP](#eventually-consistent-stream-processor-and-rtolap)
   - [Eventually Consistent Stream Processor and HTAP](#eventually-consistent-stream-processor-and-htap)
4. [Pros and Cons Analysis](#pros-and-cons-analysis)
5. [Mathematical Considerations](#mathematical-considerations)
6. [Code Examples](#code-examples)
7. [Summary](#summary)
8. [References](#references)
9. [Tags](#tags)

---

## Introduction

Real-time analytics spans a spectrum of use cases, each with unique requirements regarding consistency, access to historical data, and interaction with application logic.

- **Consistent Solutions**: Necessary when interacting directly with application logic and requiring immediate consistency.
- **Eventually Consistent Solutions**: Suitable for analytical workloads where eventual consistency is acceptable, and access to historical data is necessary.

We will explore deployment models ranging from consistent streaming databases to eventually consistent stream processors combined with various storage solutions.

---

## Consistency Spectrum in Real-Time Analytics

![Consistency Spectrum](consistency_spectrum.png)

*Figure: Real-Time Analytical Spectrum from Consistent to Eventually Consistent Solutions.*

- **Left End**: Consistent streaming solutions interacting closely with operational applications.
- **Right End**: Eventually consistent solutions focusing on analytical workloads without direct application interaction.

Understanding where your use case lies on this spectrum is crucial for selecting the appropriate deployment model.

---

## Deployment Models

### Consistent Streaming Database

#### Overview

- **Use Case**: When you need a database that runs complex asynchronous stream processing and directly participates in application logic.
- **Examples**: **RisingWave**, **Materialize**.
- **Characteristics**:
  - Provides strong consistency guarantees.
  - Simplifies infrastructure by combining streaming and database capabilities.
  - Suitable for applications requiring immediate consistency and limited historical data.

#### Architecture

![Consistent Streaming Database Architecture](consistent_streaming_database.png)

*Figure 10-1: Operational Analytics using a Consistent Streaming Database.*

- **Solid Arrows**: Represent streaming data flows.
- **Dashed Arrows**: Represent read/write interactions between the application and databases.
- **Components**:
  - **OLTP Database**: Traditional operational database (e.g., PostgreSQL).
  - **Consistent Streaming Database**: Consumes data from OLTP, performs stream processing, and stores results.
  - **Application**: Writes to OLTP and reads from the streaming database.

#### Pros and Cons

| Pros                                                                                                                                                  | Cons                                                                                                                                                 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Data freshness in milliseconds.                                                                                                                      | - Lacks columnar storage, which is better for faster analytical queries.                                                                             |
| - Separation of read and write resources, enabling independent scaling.                                                                                | - Difficulty in receiving historical data from the analytical plane due to data gravity.                                                             |
| - Ability to combine inputs from multiple OLTP databases (even different vendors) into a single consistent streaming database.                         | - May struggle with large data sizes.                                                                                                                |
| - Perform incremental transformations before sending data to the analytical plane.                                                                     |                                                                                                                                                      |
| - Supports both push and pull queries using the same query engine/interface.                                                                           |                                                                                                                                                      |

#### When to Use

- **Best For**: Use cases requiring consistent stream processing that is part of application logic but doesn't need extensive historical data.
- **Not Ideal For**: Scenarios requiring complex analytical queries over large historical datasets.

---

### Consistent Stream Processor and RTOLAP

#### Overview

- **Use Case**: When you prefer outputting streams to a columnar database and need consistency for application logic.
- **Examples**: **Pathway** (stream processor), **Apache Pinot** (RTOLAP).
- **Characteristics**:
  - Stream processor performs transformations and writes to Kafka.
  - RTOLAP database consumes from Kafka, joins with historical data, and serves low-latency queries.
  - Separation of push and pull queries.

#### Architecture

![Consistent Stream Processor and RTOLAP Architecture](consistent_stream_processor_rtolap.png)

*Figure 10-2: Consistent Stream Processor and RTOLAP Database.*

- **Components**:
  - **Consistent Stream Processor**: Performs real-time data transformations.
  - **Kafka**: Acts as the streaming platform.
  - **RTOLAP Database**: Consumes transformed data and serves analytical queries.
  - **Data Warehouse/Lakehouse**: Stores historical data (analytical plane).

#### Pros and Cons

| Pros                                                                                                                                      | Cons                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Provides data freshness in milliseconds to seconds.                                                                                      | - Separation of push and pull queries often requires separate engineering efforts and coordination.                                                       |
| - User-facing analytics can include all historical context without storing it on operational infrastructure.                               |                                                                                                                                                           |
| - Columnar format in RTOLAP provides fast analytical workloads to the application.                                                         |                                                                                                                                                           |
| - Consistent stream processor can participate in the application's business logic.                                                         |                                                                                                                                                           |

#### When to Use

- **Best For**: Use cases requiring most or all historical data for user-facing analytics and a consistent stream processor interacting with application logic.
- **Not Ideal For**: Scenarios where infrastructure complexity needs to be minimized.

---

### Eventually Consistent OLAP Streaming Database

#### Overview

- **Use Case**: When you want to consolidate analytical workloads into one solution and can accept eventual consistency.
- **Examples**: **Proton** (a streaming OLAP database).
- **Characteristics**:
  - Combines stream processing and OLAP capabilities.
  - Provides low-latency analytical queries with columnar storage.
  - Emits data to Kafka for further distribution if needed.

#### Architecture

![Eventually Consistent OLAP Streaming Database Architecture](eventually_consistent_olap_streaming_db.png)

*Figure 10-3: Eventually Consistent Streaming OLAP Database.*

- **Components**:
  - **OLTP Database**: Source of transactional data.
  - **Streaming OLAP Database**: Consumes data from OLTP, performs transformations, and stores results.
  - **Application**: Reads from the streaming OLAP database.
  - **Kafka**: Optional, for emitting analytical changes.

#### Pros and Cons

| Pros                                                                                                                                      | Cons                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Data freshness in milliseconds to seconds.                                                                                               | - Only eventually consistent; should not participate in application logic requiring strict consistency.                                                   |
| - Access to more or all historical data, providing richer context.                                                                         |                                                                                                                                                           |
| - Ability to emit analytical changes for building replicas or distributing data globally.                                                  |                                                                                                                                                           |
| - Simplifies infrastructure by converging stream processing and OLAP technologies.                                                         |                                                                                                                                                           |
| - Single SQL engine for both push and pull queries.                                                                                        |                                                                                                                                                           |

#### When to Use

- **Best For**: Reducing infrastructure complexity while providing comprehensive historical data for user-facing analytics.
- **Not Ideal For**: Applications requiring strict consistency in processing that directly impacts application logic.

---

### Eventually Consistent Stream Processor and RTOLAP

#### Overview

- **Use Case**: Commonly used model for providing real-time analytics, where eventual consistency is acceptable.
- **Examples**: **Apache Flink** (stream processor), **Apache Pinot** (RTOLAP).
- **Characteristics**:
  - Stream processor performs transformations and writes to Kafka.
  - RTOLAP database consumes data, combines with historical data, and serves queries.

#### Architecture

![Eventually Consistent Stream Processor and RTOLAP Architecture](eventually_consistent_stream_processor_rtolap.png)

*Figure 10-4: Eventually Consistent Stream Processor and RTOLAP.*

#### Pros and Cons

| Pros                                                                                                                                      | Cons                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Data freshness in milliseconds to seconds.                                                                                               | - Complex and bulky solution, potentially leading to higher costs.                                                                                         |
| - Combines historical and real-time data for a complete analytical view.                                                                   | - Stream processor should not participate in application logic due to eventual consistency.                                                                |
|                                                                                                                                            | - Separate engines for push and pull queries increase engineering complexity.                                                                              |

#### When to Use

- **Best For**: Use cases requiring comprehensive historical data for analytics where consistency is not critical.
- **Not Ideal For**: Scenarios where tight integration with application logic and strict consistency are required.

---

### Eventually Consistent Stream Processor and HTAP

#### Overview

- **Use Case**: When you want to keep analytical workloads near the operational plane with minimal infrastructure complexity.
- **Examples**: **Apache Flink** (stream processor), **SingleStore** or **Hydra** (HTAP databases).
- **Characteristics**:
  - HTAP database handles both transactional and analytical workloads.
  - Stream processor enriches data with limited historical context.

#### Architecture

![Eventually Consistent Stream Processor and HTAP Architecture](eventually_consistent_stream_processor_htap.png)

*Figure 10-5: Eventually Consistent Stream Processor and HTAP.*

#### Pros and Cons

| Pros                                                                                                                                      | Cons                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Data freshness in milliseconds.                                                                                                          | - Limited historical data due to storage constraints in HTAP databases.                                                                                    |
| - HTAP databases provide fast analytical queries with columnar storage.                                                                    | - Complexity increases when implementing data retention policies.                                                                                         |
| - Lower infrastructural complexity compared to separate OLAP systems.                                                                      | - Stream processor cannot participate in application logic due to eventual consistency.                                                                   |

#### When to Use

- **Best For**: Use cases requiring low-latency analytics with limited historical data and minimal infrastructure.
- **Not Ideal For**: Scenarios needing extensive historical data or strict consistency in application logic.

---

## Pros and Cons Analysis

### Summary Table

| Deployment Model                                    | Pros                                                                                                                                      | Cons                                                                                                                                                      |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Consistent Streaming Database**                   | - Millisecond data freshness<br>- Independent scaling of read/write resources<br>- Combines multiple OLTP sources<br>- Single SQL interface | - Lacks columnar storage<br>- Difficulty sourcing historical data<br>- Challenges with large data sizes                                                    |
| **Consistent Stream Processor and RTOLAP**          | - Millisecond to second data freshness<br>- Includes historical context<br>- Fast analytical queries                                       | - Separation of push/pull queries requires coordination<br>- Higher complexity                                                                             |
| **Eventually Consistent OLAP Streaming Database**   | - Millisecond to second data freshness<br>- Access to extensive historical data<br>- Simplifies infrastructure<br>- Single SQL engine       | - Not suitable for application logic requiring strict consistency                                                                                         |
| **Eventually Consistent Stream Processor and RTOLAP** | - Millisecond to second data freshness<br>- Complete analytical view with historical data                                                  | - Complex infrastructure<br>- Stream processor unsuitable for application logic<br>- Separate engines for push/pull queries increase complexity            |
| **Eventually Consistent Stream Processor and HTAP** | - Millisecond data freshness<br>- Fast analytical queries<br>- Lower infrastructure complexity                                             | - Limited historical data<br>- Complexity in data retention<br>- Stream processor unsuitable for application logic                                         |

---

## Mathematical Considerations

### Consistency and Latency

- **Consistency Models**:
  - **Strong Consistency**: Guarantees that all reads receive the most recent write.
  - **Eventual Consistency**: Guarantees that, given enough time without new updates, all reads will eventually return the last written value.

- **Latency (\( L \))**:
  - **Trade-off**: Strong consistency often comes with higher latency due to synchronization overhead.
  - **Equation**:
    \[
    L_{\text{strong}} > L_{\text{eventual}}
    \]
    Where \( L_{\text{strong}} \) is the latency under strong consistency, and \( L_{\text{eventual}} \) is the latency under eventual consistency.

### Data Freshness

- **Data Freshness (\( F \))**:
  - Defined as the time difference between when data is generated and when it is available for consumption.
  - **Equation**:
    \[
    F = T_{\text{available}} - T_{\text{generated}}
    \]
  - **Goal**: Minimize \( F \) to achieve real-time analytics.

### Storage Considerations

- **Row-Based vs. Columnar Storage**:
  - **Row-Based Storage**:
    - Optimized for write-heavy workloads.
    - Suitable for transactional operations.
  - **Columnar Storage**:
    - Optimized for read-heavy analytical queries.
    - Provides faster query performance for aggregations and scans.

---

## Code Examples

### Example: Setting Up a Consistent Streaming Database with Materialize

**Prerequisites**:

- **Materialize** installed.
- **PostgreSQL** as the OLTP database.
- **Kafka** for streaming data.

**1. Ingest Data from PostgreSQL Using Debezium**

```sql
-- Create source from PostgreSQL via Debezium connector
CREATE SOURCE users_source
FROM KAFKA BROKER 'localhost:9092' TOPIC 'dbserver1.public.users'
FORMAT AVRO USING CONFLUENT SCHEMA REGISTRY 'http://localhost:8081'
ENVELOPE DEBEZIUM;
```

**2. Create Materialized View**

```sql
-- Create a materialized view for real-time analytics
CREATE MATERIALIZED VIEW active_users AS
SELECT
  user_id,
  COUNT(*) AS login_count
FROM users_source
WHERE last_login > NOW() - INTERVAL '1 day'
GROUP BY user_id;
```

**3. Query the Materialized View**

```sql
-- Query the real-time analytics
SELECT * FROM active_users WHERE login_count > 5;
```

---

### Example: Consistent Stream Processing with Pathway and Apache Pinot

**Prerequisites**:

- **Pathway** installed.
- **Apache Pinot** set up.
- **Kafka** as the messaging system.

**1. Define Pathway Transformation**

```python
import pathway as pw

# Define the streaming table
users = pw.kafka.read(
    topic='user_events',
    bootstrap_servers='localhost:9092',
    value_deserializer=pw.kafka.json_deserializer
)

# Transformation logic
active_users = users \
    .filter(pw.col('event_type') == 'login') \
    .group_by('user_id') \
    .agg(login_count=pw.count())

# Write output to Kafka
active_users.to_kafka(
    topic='active_users',
    bootstrap_servers='localhost:9092',
    value_serializer=pw.kafka.json_serializer
)
```

**2. Configure Pinot to Consume from Kafka**

```json
{
  "tableName": "active_users",
  "tableType": "REALTIME",
  "ingestionConfig": {
    "streamIngestionConfig": {
      "streamConfigMaps": [
        {
          "streamType": "kafka",
          "stream.kafka.topic.name": "active_users",
          "stream.kafka.decoder.class.name": "org.apache.pinot.plugin.stream.kafka.KafkaJSONMessageDecoder",
          "stream.kafka.broker.list": "localhost:9092"
        }
      ]
    }
  }
}
```

---

## Summary

Selecting the appropriate deployment model for real-time analytics depends on:

- **Consistency Requirements**: Whether your application can tolerate eventual consistency or requires strong consistency.
- **Access to Historical Data**: The extent to which historical data is necessary for your analytics.
- **Interaction with Application Logic**: Whether the stream processing needs to participate directly in application logic.
- **Infrastructure Complexity**: Balancing the benefits of more complex architectures against the costs and maintenance overhead.

**Key Takeaways**:

- **Consistent Streaming Databases** are ideal for applications requiring immediate consistency and limited historical data.
- **Consistent Stream Processor with RTOLAP** provides historical context but increases infrastructure complexity.
- **Eventually Consistent Streaming OLAP Databases** simplify infrastructure and provide access to historical data but shouldn't be used for application logic requiring strict consistency.
- **Eventually Consistent Stream Processor with RTOLAP** is common but comes with higher complexity and should not be used for application logic.
- **Eventually Consistent Stream Processor with HTAP** keeps analytics near the operational plane but is limited in historical data and consistency for application logic.

---

## References

1. **RisingWave**: [RisingWave Official Website](https://www.risingwave.com/)
2. **Materialize**: [Materialize Official Website](https://materialize.com/)
3. **Apache Pinot**: [Apache Pinot Documentation](https://pinot.apache.org/)
4. **Proton**: [Proton Streaming OLAP Database](https://www.timeplus.io/)
5. **Apache Flink**: [Apache Flink Documentation](https://flink.apache.org/)
6. **Pathway**: [Pathway Official Website](https://pathway.com/)
7. **Apache Kafka**: [Apache Kafka Documentation](https://kafka.apache.org/)
8. **HTAP Databases**: [Hybrid Transactional/Analytical Processing Databases](https://en.wikipedia.org/wiki/Hybrid_transactional/analytical_processing)

---

## Tags

#RealTimeAnalytics #StreamingDatabases #StreamProcessing #OperationalAnalytics #DataEngineering #DeploymentModels #Consistency #HTAP #RTOLAP #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.