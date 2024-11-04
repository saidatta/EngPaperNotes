## Overview

In this note, we explore how major data warehouses—**BigQuery**, **Redshift**, and **Snowflake**—are integrating streaming capabilities to support real-time analytics. This integration places them in the overlapping area between the **analytical plane** and the **streaming plane** in our data architecture Venn diagram.

---

## Table of Contents

1. [Introduction](#introduction)
2. [BigQuery](#bigquery)
   - [Streaming Ingestion](#bigquery-streaming-ingestion)
   - [Code Example](#bigquery-code-example)
   - [Integration with Cloud Dataflow](#integration-with-cloud-dataflow)
3. [Redshift](#redshift)
   - [Streaming Ingestion](#redshift-streaming-ingestion)
   - [Code Example](#redshift-code-example)
   - [Comparison with Streaming Databases](#comparison-with-streaming-databases)
4. [Snowflake](#snowflake)
   - [Streaming Capabilities](#snowflake-streaming-capabilities)
   - [Snowpipe Streaming](#snowpipe-streaming)
   - [Code Example](#snowpipe-code-example)
   - [Dynamic Tables](#dynamic-tables)
   - [Comparison with Streaming Databases](#snowflake-comparison-with-streaming-databases)
5. [Mathematical Considerations](#mathematical-considerations)
   - [Latency and Refresh Rates](#latency-and-refresh-rates)
   - [Batch vs. Streaming Architectures](#batch-vs-streaming-architectures)
6. [Summary](#summary)
7. [References](#references)
8. [Tags](#tags)

---

## Introduction

Data warehouses are increasingly incorporating streaming capabilities to handle real-time data ingestion and analytics. This convergence enhances their ability to support use cases where low latency is critical, such as IoT data processing, system telemetry, and clickstream analysis.

![Venn Diagram of Data Planes](venn_diagram_data_planes.png)

*Figure 11-1: Venn Diagram Illustrating the Overlapping of Analytical and Streaming Planes.*

---

## BigQuery

### Streaming Ingestion

**Google BigQuery** supports streaming ingestion through its SDKs for Java and Python. Data inserted via streaming ingestion is available for real-time analysis within seconds.

- **Key Features**:
  - Real-time data availability.
  - Supports near-real-time analytics use cases.
  - Data is immediately queryable upon insertion.

### Code Example

#### Streaming Ingestion Using Python SDK

```python
from google.cloud import bigquery
import json

def stream_data(dataset_name, table_name, json_data):
    client = bigquery.Client()
    table_id = f"{client.project}.{dataset_name}.{table_name}"
    table = client.get_table(table_id)
    
    # Convert JSON string to dictionary
    data = json.loads(json_data)
    rows_to_insert = [data]
    
    errors = client.insert_rows_json(table, rows_to_insert)
    
    if not errors:
        print(f'Loaded 1 row into {dataset_name}:{table_name}')
    else:
        print('Errors:')
        print(errors)
```

- **Explanation**:
  - **Client Initialization**: Creates a BigQuery client.
  - **Table Reference**: Specifies the dataset and table.
  - **Data Conversion**: Parses JSON data into a dictionary.
  - **Row Insertion**: Uses `insert_rows_json` to insert data.
  - **Error Handling**: Checks for insertion errors.

#### Usage Example

```python
json_data = '{"name": "Alice", "age": 30, "city": "New York"}'
stream_data('my_dataset', 'my_table', json_data)
```

### Integration with Cloud Dataflow

- **Apache Beam**: BigQuery integrates with **Google Cloud Dataflow**, which is based on Apache Beam.
- **Dataflow Capabilities**:
  - Read from and write to BigQuery.
  - Process streaming data in a serverless fashion.
- **Kafka Integration**: Google announced support for Apache Kafka for BigQuery, allowing for seamless deployment of Kafka in conjunction with BigQuery.

---

## Redshift

### Streaming Ingestion

**Amazon Redshift** supports streaming ingestion for low-latency data ingestion from:

- **Amazon Kinesis Data Streams**
- **Amazon Managed Streaming for Apache Kafka (MSK)**

Data is ingested into **materialized views**, which serve as landing areas for streaming data.

- **Key Features**:
  - Process data as it arrives.
  - Map JSON values from streams to materialized view columns.
  - Supports near-real-time analytics use cases.

### Code Example

#### Creating a Materialized View for Kafka Topic

```sql
CREATE MATERIALIZED VIEW MyView AUTO REFRESH YES AS
SELECT
    kafka_partition,
    kafka_offset,
    kafka_timestamp_type,
    kafka_timestamp,
    kafka_key,
    JSON_PARSE(kafka_value) AS Data,
    kafka_headers
FROM
    MySchema."mytopic"
WHERE
    CAN_JSON_PARSE(kafka_value);
```

- **Explanation**:
  - **Materialized View**: `MyView` is created to consume data from the Kafka topic `mytopic`.
  - **AUTO REFRESH YES**: Enables automatic refreshing of the view.
  - **Data Parsing**: Uses `JSON_PARSE` to parse Kafka messages.
  - **Filtering**: Ensures only messages that can be parsed as JSON are included.

#### Querying the Materialized View

```sql
SELECT Data->>'field1' AS field1, Data->>'field2' AS field2
FROM MyView;
```

### Comparison with Streaming Databases

- **Refresh Mechanism**:
  - **Redshift**: Materialized views refresh periodically, even with AUTO REFRESH, leading to some latency.
  - **Streaming Databases** (e.g., Materialize, RisingWave): Provide continuous, incremental updates with minimal latency.
- **Underlying Architecture**:
  - **Redshift**: Batch-oriented, which affects real-time processing capabilities.
  - **Streaming Databases**: Built on streaming architectures for true real-time processing.

---

## Snowflake

### Streaming Capabilities

Snowflake has expanded its streaming capabilities, offering:

1. **Continuous Data Loading**:
   - **Snowpipe Streaming**: For streaming ingestion.
   - **Kafka Connector**: Integrates Kafka data into Snowflake.
2. **Continuous Data Transformation**:
   - **Dynamic Tables**: Declaratively implement automated data pipelines using SQL.
3. **Change Data Tracking**:
   - **CDC**: Tracks changes in tables and dynamic tables.

### Snowpipe Streaming

#### Introduction

- **Purpose**: Enables low-latency data ingestion into Snowflake tables.
- **Implementation**: Provides an SDK (currently in Java) for developers to build streaming ingestion applications.

### Code Example

#### Streaming Ingestion Using Java SDK

```java
import net.snowflake.ingest.streaming.*;

public class SnowflakeStreamingIngestExample {
    public static void main(String[] args) throws Exception {
        // Create client configuration
        SnowflakeStreamingIngestClientFactory.Builder builder = 
            new SnowflakeStreamingIngestClientFactory.Builder("MY_CLIENT");

        // Set required properties (replace with your own)
        builder.setHost("myaccount.snowflakecomputing.com")
               .setUser("MY_USER")
               .setPrivateKey("MY_PRIVATE_KEY")
               .setRole("MY_ROLE")
               .setWarehouse("MY_WAREHOUSE");

        // Build the client
        SnowflakeStreamingIngestClient client = builder.build();

        // Create a channel
        OpenChannelRequest request = OpenChannelRequest.builder("MY_CHANNEL")
                .setDBName("MY_DB")
                .setSchemaName("MY_SCHEMA")
                .setTableName("MY_TABLE")
                .build();
        SnowflakeStreamingIngestChannel channel = client.openChannel(request);

        // Insert rows
        for (int val = 0; val < 1000; val++) {
            Map<String, Object> row = new HashMap<>();
            row.put("c1", val);
            InsertValidationResponse response = channel.insertRow(row, String.valueOf(val));

            if (response.hasErrors()) {
                throw response.getInsertErrors().get(0).getException();
            }
        }

        // Close the channel and client
        channel.close().get();
        client.close();
    }
}
```

- **Explanation**:
  - **Client Configuration**: Sets up connection properties.
  - **Channel Creation**: Opens a channel to a specific table.
  - **Row Insertion**: Inserts rows into the channel.
  - **Error Handling**: Checks for insertion errors.
  - **Closing Resources**: Closes the channel and client connections.

### Dynamic Tables

- **Purpose**: Simplify data ingestion and transformation using declarative SQL pipelines.
- **Functionality**:
  - Similar to materialized views.
  - Automatically refresh data based on a defined schedule or triggers.

#### Example: Creating a Dynamic Table

```sql
CREATE OR REPLACE DYNAMIC TABLE sales_aggregated
WAREHOUSE = 'COMPUTE_WH'
AS
SELECT
    date_trunc('day', timestamp) AS sales_date,
    SUM(amount) AS total_sales
FROM
    raw_sales_data
GROUP BY
    sales_date;
```

- **Explanation**:
  - **Dynamic Table**: `sales_aggregated` is updated automatically.
  - **Data Transformation**: Aggregates sales data on a daily basis.
  - **Scheduling**: The refresh schedule is defined by the warehouse settings.

### Comparison with Streaming Databases

- **Refresh Mechanism**:
  - **Snowflake**: Dynamic tables refresh periodically based on schedules.
  - **Streaming Databases**: Provide continuous, real-time updates.
- **Underlying Architecture**:
  - **Snowflake**: Batch-oriented architecture, which introduces latency.
  - **Streaming Databases**: Streaming architecture enables minimal latency.

---

## Mathematical Considerations

### Latency and Refresh Rates

- **Latency (\( L \))**: Time between data generation and data availability for query.

#### For Batch-Oriented Systems:

\[
L_{\text{batch}} = T_{\text{ingest}} + T_{\text{batch\_window}} + T_{\text{process}} + T_{\text{refresh}}
\]

- **\( T_{\text{ingest}} \)**: Time to ingest data.
- **\( T_{\text{batch\_window}} \)**: Duration of batch window.
- **\( T_{\text{process}} \)**: Time to process data.
- **\( T_{\text{refresh}} \)**: Time to refresh materialized views or dynamic tables.

#### For Streaming Systems:

\[
L_{\text{streaming}} = T_{\text{ingest}} + T_{\text{process}}
\]

- **No Batch Window**: Data is processed as it arrives.
- **Lower Latency**: \( L_{\text{streaming}} \ll L_{\text{batch}} \)

### Batch vs. Streaming Architectures

#### Batch Architecture:

- **Characteristics**:
  - Data is collected over a period.
  - Processing occurs after the batch window closes.
- **Implications**:
  - Higher latency due to batch windows.
  - Suitable for scenarios where real-time data is not critical.

#### Streaming Architecture:

- **Characteristics**:
  - Data is processed as soon as it arrives.
  - Continuous processing and updating.
- **Implications**:
  - Low latency, near real-time data availability.
  - Ideal for real-time analytics and time-sensitive applications.

---

## Summary

- **BigQuery**, **Redshift**, and **Snowflake** are integrating streaming capabilities to support real-time analytics.
- **BigQuery**:
  - Supports streaming ingestion via SDKs.
  - Integrates with Apache Beam and Cloud Dataflow.
- **Redshift**:
  - Uses materialized views for streaming ingestion from Kinesis and Kafka.
  - Materialized views refresh periodically, introducing some latency.
- **Snowflake**:
  - Offers Snowpipe Streaming for data ingestion.
  - Introduced dynamic tables for continuous data transformation.
  - Underlying batch architecture limits real-time capabilities.
- **Comparison with Streaming Databases**:
  - Streaming databases provide continuous, incremental updates with minimal latency.
  - Data warehouses have higher latency due to batch-oriented architectures.

---

## References

1. **Google BigQuery Documentation**: [Streaming Data into BigQuery](https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery)
2. **BigQuery Python Client Library**: [google-cloud-bigquery](https://googleapis.dev/python/bigquery/latest/index.html)
3. **Amazon Redshift Documentation**: [Streaming Ingestion with Amazon Redshift](https://docs.aws.amazon.com/redshift/latest/dg/streaming-ingestion.html)
4. **Amazon Redshift Materialized Views**: [Creating Materialized Views](https://docs.aws.amazon.com/redshift/latest/dg/materialized-view-create-sql.html)
5. **Snowflake Documentation**:
   - [Snowpipe Streaming](https://docs.snowflake.com/en/user-guide/data-load-snowpipe-streaming.html)
   - [Snowflake Connector for Kafka](https://docs.snowflake.com/en/user-guide/kafka-connector.html)
   - [Dynamic Tables](https://docs.snowflake.com/en/user-guide/dynamic-tables.html)

---

## Tags

#DataWarehouses #BigQuery #Redshift #Snowflake #StreamingIngestion #RealTimeAnalytics #MaterializedViews #DynamicTables #StreamingDatabases #DataEngineering #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.