## Overview

In this note, we delve into the rapidly growing trend of streaming engines and stream processing systems integrating seamlessly with data lakes and lakehouses. We explore how technologies like **Apache Iceberg**, **Apache Paimon**, **Delta Lake**, and **Apache Hudi** are bridging the gap between streaming and batch processing. We also discuss the implications of this convergence for streaming databases and the future of data processing architectures.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Delta Lake](#delta-lake)
   - [Introduction to Delta Lake](#introduction-to-delta-lake)
   - [Using Delta Lake with Spark Structured Streaming](#using-delta-lake-with-spark-structured-streaming)
     - [Reading from Delta Tables](#reading-from-delta-tables)
     - [Writing to Delta Tables](#writing-to-delta-tables)
     - [Code Examples](#delta-lake-code-examples)
3. [Apache Paimon](#apache-paimon)
   - [Introduction to Apache Paimon](#introduction-to-apache-paimon)
   - [Creating and Querying Tables in Paimon](#creating-and-querying-tables-in-paimon)
     - [Code Examples](#apache-paimon-code-examples)
4. [Apache Iceberg](#apache-iceberg)
   - [Introduction to Apache Iceberg](#introduction-to-apache-iceberg)
   - [Integration with Streaming Platforms](#integration-with-streaming-platforms)
   - [Using Iceberg with RisingWave](#using-iceberg-with-risingwave)
     - [Code Example](#risingwave-iceberg-code-example)
   - [Using Iceberg with Spark Structured Streaming](#using-iceberg-with-spark-structured-streaming)
     - [Writing to Iceberg Tables](#writing-to-iceberg-tables)
     - [Reading from Iceberg Tables](#reading-from-iceberg-tables)
     - [Code Examples](#iceberg-spark-code-examples)
5. [Apache Hudi](#apache-hudi)
   - [Introduction to Apache Hudi](#introduction-to-apache-hudi)
   - [Integration with Spark Structured Streaming](#hudi-spark-integration)
     - [Writing to Hudi Tables](#writing-to-hudi-tables)
     - [Reading from Hudi Tables](#reading-from-hudi-tables)
     - [Code Examples](#hudi-spark-code-examples)
6. [OneTable or XTable](#onetable-or-xtable)
   - [Introduction](#introduction-to-onetable-xtable)
   - [Interoperability Between Table Formats](#interoperability-between-table-formats)
7. [The Relationship of Streaming and Lakehouses](#the-relationship-of-streaming-and-lakehouses)
   - [Trends and Future Directions](#trends-and-future-directions)
   - [Impact on Streaming Databases](#impact-on-streaming-databases)
8. [Conclusion](#conclusion)
9. [References](#references)
10. [Tags](#tags)

---

## Introduction

The data processing landscape is witnessing a significant convergence between streaming technologies and lakehouse architectures. Streaming engines like **Confluent**, **Redpanda**, and **WarpStream** are offering direct access to data stored in object storage layers using open table formats like **Apache Iceberg**. Similarly, **Apache Paimon** brings stream processing to the data lake, coining the term **"Streamhouse"**.

This convergence is bridging the gap between the **analytical plane** and the **streaming plane**, enabling more efficient and unified data processing workflows.

![Venn Diagram of Data Planes](venn_diagram_data_planes.png)

*Figure 11-1: Venn Diagram Illustrating the Overlapping of Analytical and Streaming Planes.*

---

## Delta Lake

### Introduction to Delta Lake

**Delta Lake** is an open-source storage layer that brings **ACID transactions** to data lakes. It builds upon the **Parquet** file format and adds a **transaction log** for implementing ACID properties and scalable metadata handling.

- **Key Features**:
  - ACID transactions
  - Scalable metadata handling
  - Time travel (data versioning)
  - Schema enforcement and evolution
  - Compatible with **Apache Spark** APIs

### Using Delta Lake with Spark Structured Streaming

**Spark Structured Streaming** integrates seamlessly with Delta Lake, allowing for both streaming and batch processing of data at scale. While Spark Structured Streaming uses micro-batching under the hood, it enables near-real-time processing of data in Delta Tables.

#### Reading from Delta Tables

You can read from a Delta Table as a streaming source using Spark Structured Streaming.

```scala
// Scala Example
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder.appName("DeltaLakeReadStream").getOrCreate()

val deltaStream = spark.readStream
  .format("delta")
  .load("/tmp/delta/events")
```

#### Writing to Delta Tables

Similarly, you can write streaming data into a Delta Table.

```scala
// Scala Example
deltaStream.writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", "/tmp/delta/events/_checkpoints/")
  .start("/tmp/delta/events")
```

#### Code Examples

##### Example: Reading from a Delta Table

```scala
import org.apache.spark.sql.SparkSession
import io.delta.implicits._

val spark = SparkSession.builder.appName("DeltaLakeReadStream").getOrCreate()

val deltaStream = spark.readStream
  .delta("/tmp/delta/events")

deltaStream.printSchema()

deltaStream.writeStream
  .format("console")
  .start()
  .awaitTermination()
```

##### Example: Writing to a Delta Table

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder.appName("DeltaLakeWriteStream").getOrCreate()

import spark.implicits._

// Create streaming DataFrame
val inputStream = spark.readStream
  .format("socket")
  .option("host", "localhost")
  .option("port", 9999)
  .load()

// Transform and write to Delta Table
inputStream.writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", "/tmp/delta/events/_checkpoints/")
  .start("/tmp/delta/events")
  .awaitTermination()
```

---

## Apache Paimon

### Introduction to Apache Paimon

**Apache Paimon** is an open-source data lake storage engine built on **Apache Flink**. It is designed for **streaming-first architectures** and provides capabilities similar to Delta Lake but with pure stream processing.

- **Key Features**:
  - Streaming and batch processing support
  - Built on Apache Flink
  - Supports multiple compute engines (Hive, Spark, Trino)
  - Uses its own file format based on **LSM trees**

### Creating and Querying Tables in Paimon

#### Code Examples

##### Example: Creating Tables

```sql
-- Create a primary table 'customers'
CREATE TABLE customers (
    id INT PRIMARY KEY NOT ENFORCED,
    name STRING,
    country STRING,
    zip STRING
);

-- Insert data into 'customers' table
INSERT INTO customers VALUES
  (1, 'Alice', 'USA', '10001'),
  (2, 'Bob', 'Canada', 'L5B4L4');

-- Create a temporary table 'Orders' based on a Kafka topic
CREATE TEMPORARY TABLE Orders (
    order_id INT,
    total DECIMAL(10,2),
    customer_id INT,
    proc_time AS PROCTIME()
) WITH (
    'connector' = 'kafka',
    'topic' = 'orders_topic',
    'properties.bootstrap.servers' = 'localhost:9092',
    'format' = 'csv'
);
```

##### Example: Querying with a Lookup JOIN

```sql
SELECT o.order_id, o.total, c.country, c.zip
FROM Orders AS o
JOIN customers FOR SYSTEM_TIME AS OF o.proc_time AS c
ON o.customer_id = c.id;
```

- **Explanation**:
  - **Temporal Join**: Uses `FOR SYSTEM_TIME AS OF` to perform a temporal table join.
  - **Real-Time Processing**: Enables real-time enrichment of streaming data with dimensional data.

---

## Apache Iceberg

### Introduction to Apache Iceberg

**Apache Iceberg** is an open table format for huge analytic datasets. It was originally developed by Netflix and is now widely adopted.

- **Key Features**:
  - Handles petabyte-scale tables
  - Supports schema evolution
  - ACID transactions
  - Partition evolution
  - Supports multiple compute engines (Spark, Flink, Trino, Hive, Presto, Snowflake)

### Integration with Streaming Platforms

Streaming platforms like **Redpanda**, **WarpStream**, and **Apache Kafka** are integrating Iceberg/Parquet support to directly access cold data through Iceberg APIs.

### Using Iceberg with RisingWave

**RisingWave** is a streaming database that supports Iceberg as a sink.

#### Code Example

```sql
CREATE SINK s1_sink FROM s1_table
WITH (
    connector = 'iceberg',
    warehouse.path = 's3a://my-iceberg-bucket/path/to/warehouse',
    s3.endpoint = 'https://s3.amazonaws.com',
    s3.access.key = '${ACCESS_KEY}',
    s3.secret.key = '${SECRET_KEY}',
    database.name = 'dev',
    table.name = 'table',
    primary_key = 'seq_id'
);
```

- **Explanation**:
  - **Sinking Data**: The sink `s1_sink` writes data from `s1_table` into an Iceberg table.
  - **Configuration**: Specifies S3 credentials and table details.
  - **Modes**: Supports `upsert` and `append-only` modes.

### Using Iceberg with Spark Structured Streaming

Iceberg integrates with Spark Structured Streaming for both reading and writing streaming data.

#### Writing to Iceberg Tables

```python
# Python Example
df.writeStream \
   .format("iceberg") \
   .outputMode("append") \
   .trigger(processingTime='10 seconds') \
   .option("path", "db.table_name") \
   .option("checkpointLocation", "/tmp/checkpoints") \
   .start()
```

#### Reading from Iceberg Tables

```python
# Python Example
df = spark.readStream \
    .format("iceberg") \
    .load("db.table_name")

df.writeStream \
    .format("console") \
    .start() \
    .awaitTermination()
```

#### Iceberg Spark Code Examples

##### Example: Writing Streaming Data to an Iceberg Table

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("IcebergWriteStream") \
    .getOrCreate()

df = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

df.writeStream \
   .format("iceberg") \
   .outputMode("append") \
   .trigger(processingTime='10 seconds') \
   .option("path", "db.table_name") \
   .option("checkpointLocation", "/tmp/checkpoints") \
   .start() \
   .awaitTermination()
```

##### Example: Reading Streaming Data from an Iceberg Table

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("IcebergReadStream") \
    .getOrCreate()

df = spark.readStream \
    .format("iceberg") \
    .load("db.table_name")

df.writeStream \
    .format("console") \
    .start() \
    .awaitTermination()
```

---

## Apache Hudi

### Introduction to Apache Hudi

**Apache Hudi** is an open-source data management framework used to simplify incremental data processing and data pipeline development.

- **Key Features**:
  - ACID transactions
  - Record-level updates and deletes
  - Time Travel queries
  - Efficient storage management
  - Supports multiple query engines (Spark, Impala, Hive, Presto, Trino)

### Integration with Spark Structured Streaming

Hudi integrates with Spark Structured Streaming for both reading and writing streaming data.

#### Writing to Hudi Tables

```python
df.writeStream \
    .format("hudi") \
    .options(**hudi_streaming_options) \
    .outputMode("append") \
    .option("path", "/path/to/hudi/table") \
    .option("checkpointLocation", "/path/to/checkpoints") \
    .start()
```

#### Reading from Hudi Tables

```python
df = spark.readStream \
    .format("hudi") \
    .load("/path/to/hudi/table")

df.writeStream \
    .format("console") \
    .start() \
    .awaitTermination()
```

#### Hudi Spark Code Examples

##### Example: Writing Streaming Data to a Hudi Table

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HudiWriteStream") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "input_topic") \
    .load()

hudi_options = {
    'hoodie.table.name': 'hudi_table',
    'hoodie.datasource.write.recordkey.field': 'id',
    'hoodie.datasource.write.partitionpath.field': 'partition',
    'hoodie.datasource.write.table.type': 'MERGE_ON_READ'
}

df.writeStream \
    .format("hudi") \
    .options(**hudi_options) \
    .outputMode("append") \
    .option("path", "/path/to/hudi/table") \
    .option("checkpointLocation", "/path/to/checkpoints") \
    .start() \
    .awaitTermination()
```

##### Example: Reading Streaming Data from a Hudi Table

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HudiReadStream") \
    .getOrCreate()

df = spark.readStream \
    .format("hudi") \
    .load("/path/to/hudi/table")

df.writeStream \
    .format("console") \
    .start() \
    .awaitTermination()
```

---

## OneTable or XTable

### Introduction to OneTable/XTable

**OneTable** or **XTable** is an open table format designed to make different open table formats interoperable. It serves as a meta table format, allowing for seamless integration between:

- **Apache Iceberg**
- **Apache Hudi**
- **Delta Lake**

### Interoperability Between Table Formats

- **Purpose**: To avoid data duplication and unnecessary data movement between different lakehouses.
- **Benefits**:
  - Simplifies data management across multiple open table formats.
  - Provides an abstraction layer for data storage and access.
- **Future Extensions**: Open-source nature allows for potential support of other formats like Apache Paimon's table format.

---

## The Relationship of Streaming and Lakehouses

### Trends and Future Directions

- **Integration of Streaming Platforms and Lakehouses**:
  - Streaming platforms are starting to offer direct access to data stored in open table formats.
  - **Redpanda**, **WarpStream**, and **Apache Kafka** are moving towards exposing data in formats like Iceberg/Parquet.
- **Streamhouse Concept**:
  - Coined by **Apache Paimon**, representing the convergence of streaming and data lake architectures.

### Impact on Streaming Databases

- **Shift of Single Source of Truth**:
  - Streaming platforms may become the primary storage layer, reducing the need for separate lakehouses.
- **Data Processing Pipelines**:
  - Processing pipelines currently in lakehouses may move towards streaming platforms.
- **Performance and Cost Benefits**:
  - Streaming-native systems offer lower end-to-end latency and cost savings compared to batch-based architectures.
- **Opportunity for Streaming Databases**:
  - Increased adoption of streaming platforms enhances the relevance of streaming databases.

---

## Conclusion

The convergence of streaming technologies and lakehouse architectures is reshaping the data processing landscape. The integration of open table formats with streaming platforms enables:

- **Unified Data Access**: Seamless access to both streaming and batch data.
- **Efficiency Gains**: Reduced data duplication and movement.
- **Simplified Architectures**: Potential for streaming platforms to become the single source of truth.

As this trend continues, streaming databases stand to gain significant traction, offering powerful solutions for both operational and analytical workloads with lower latency and cost efficiency.

---

## References

1. **Apache Iceberg Documentation**: [https://iceberg.apache.org/](https://iceberg.apache.org/)
2. **Apache Paimon Documentation**: [https://paimon.apache.org/](https://paimon.apache.org/)
3. **Delta Lake Documentation**: [https://docs.delta.io/](https://docs.delta.io/)
4. **Apache Hudi Documentation**: [https://hudi.apache.org/](https://hudi.apache.org/)
5. **RisingWave Documentation**: [https://www.risingwave.com/docs/](https://www.risingwave.com/docs/)
6. **Confluent Tableflow**: [https://www.confluent.io/blog/confluent-tableflow](https://www.confluent.io/blog/confluent-tableflow)
7. **Apache Spark Structured Streaming**: [https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
8. **OneTable/XTable Project**: [https://onetable.org/](https://onetable.org/)

---

## Tags

#Lakehouse #DataEngineering #StreamingData #ApacheIceberg #ApachePaimon #DeltaLake #ApacheHudi #SparkStructuredStreaming #StreamingDatabases #DataLakes #StreamProcessing #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.